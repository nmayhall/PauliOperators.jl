# ============================================================
# Windowed evolution driver for ShardedPauliSum.
#
# M1: single-threaded driver over the sharded structures. It follows the
# exact cursor/mark protocol the threaded driver (M2) uses, so the threaded
# version is a parallelization of these loops, not a rewrite.
# ============================================================

"""
    WindowCounters(nwindows)

Preallocated per-window instrumentation (design invariant: everything
measurable, nothing allocated in the hot path). `allocd[w]` is the
`Base.gc_num()` allocation delta across window `w` — any nonzero entry
after warm-up is a bug, enforced by the test suite. Early (capacity-forced)
merges accumulate into the window they occur in.
"""
struct WindowCounters
    terms_created::Vector{Int}
    cross_appends::Vector{Int}
    merge_in::Vector{Int}
    merge_out::Vector{Int}
    t_rotate::Vector{Float64}
    t_merge::Vector{Float64}
    allocd::Vector{Int64}
    max_shard_pop::Vector{Int}
    early_merges::Vector{Int}
end
WindowCounters(nw::Int) = WindowCounters(zeros(Int, nw), zeros(Int, nw),
                                         zeros(Int, nw), zeros(Int, nw),
                                         zeros(Float64, nw), zeros(Float64, nw),
                                         zeros(Int64, nw), zeros(Int, nw),
                                         zeros(Int, nw))

"""
    merge_shards!(S::ShardedPauliSum, f::MergeFilter; counters, w)

Window-boundary merge: per shard, gather + sort pending appends, merge with
the sorted live buffer under the strict filter `f`, reset append cursors.
This is the transport seam — a future distributed backend ships cross-node
segments here; the shared-memory engine merges in place.
"""
function merge_shards!(S::ShardedPauliSum{N,W,T}, f::MergeFilter;
                       counters::Union{Nothing,WindowCounters}=nothing,
                       w::Int=1) where {N,W,T}
    maxpop = 0
    for tid in 1:S.nthreads
        tin, tout, mp = _merge_owned!(S, tid, f)
        mp > maxpop && (maxpop = mp)
        if counters !== nothing
            counters.merge_in[w] += tin
            counters.merge_out[w] += tout
        end
    end
    for tid in 1:S.nthreads
        _reset_cursor_row!(S, tid)
    end
    if counters !== nothing
        counters.max_shard_pop[w] = max(counters.max_shard_pop[w], maxpop)
    end
    return S
end


"""
    evolve!(S::ShardedPauliSum, circ::CompiledCircuit;
            truncation, local_truncation, correction, counters)

Windowed sequence evolution on the sharded engine (the flat-storage
analogue of `evolve!(B::BinnedPauliSum, circ)`). Rotations sweep shards and
append sin branches under the loose `local_truncation` (applied per term at
append time — weight cutoffs are exact there, coefficient cutoffs act on
unmerged duplicates, the documented cadence semantics); every
`circ.window` rotations the appends are sort-merged into the live buffers
under the strict `truncation`.

If a rotation's worst-case appends cannot fit the destination segments, an
early merge is triggered (harmless: it only changes truncation cadence),
growing the append segments at the boundary if the population genuinely
needs more room.

With `nthreads > 1` (set at construction) the same windowed loop runs on a
pool of long-lived workers synchronized by a spin barrier; results agree
with the serial engine up to floating-point reduction order (bit-exact at
`window = 1`). `rebalance_threshold` (default `Inf` = off) triggers a
greedy reassignment of shard ownership at a window boundary whenever the
most loaded thread exceeds that multiple of the mean load.
"""
function evolve!(S::ShardedPauliSum{N,W,T}, circ::CompiledCircuit{N};
                 truncation::TruncationStrategy=NoTruncation(),
                 local_truncation::TruncationStrategy=NoTruncation(),
                 correction::CorrectionAccumulator=NoCorrection(),
                 counters::Union{Nothing,WindowCounters}=nothing,
                 rebalance_threshold::Real=Inf) where {N,W,T}
    circ.version == S.version ||
        error("CompiledCircuit was compiled against RankMap version $(circ.version), " *
              "but the ShardedPauliSum is at version $(S.version). Recompile with `compile`.")
    correction isa NoCorrection ||
        error("correction accumulators for the sharded engine land in milestone 3")
    fstrict = _compile_filter(truncation)
    flocal = _compile_filter(local_truncation)
    if S.nthreads > 1
        return _evolve_threaded!(S, circ, fstrict, flocal, counters,
                                 Float64(rebalance_threshold))
    end

    L = length(circ)
    gz = Vector{W}(undef, L)
    gx = Vector{W}(undef, L)
    ng = Vector{Int}(undef, L)
    cosv = Vector{Float64}(undef, L)
    sinv = Vector{Float64}(undef, L)
    for i in 1:L
        g = circ.generators[i]
        gz[i], gx[i] = _pack(W, g)
        ng[i] = count_ones(gz[i] & gx[i])
        cosv[i] = cos(circ.angles[i])
        sinv[i] = sin(circ.angles[i])
    end

    gcbase = Base.gc_num()
    @inbounds for i in 1:L
        w = cld(i, circ.window)
        s_G = circ.shifts[i]
        if !_snapshot_and_precheck_owned!(S, 1, s_G)
            merge_shards!(S, fstrict; counters, w)
            counters === nothing || (counters.early_merges[w] += 1)
            # appends are empty post-merge, so segments may be regrown here
            _grow_appends_owned!(S, 1, s_G)
            _reset_cursor_row!(S, 1)
            _snapshot_and_precheck_owned!(S, 1, s_G)   # refresh sweep bounds
        end
        t0 = time_ns()
        created, ovf = _rotate_owned!(S, 1, s_G, gz[i], gx[i], ng[i], cosv[i], sinv[i], flocal)
        ovf && error("append segment overflow despite precheck (rotation $i) — this is a bug")
        if counters !== nothing
            counters.t_rotate[w] += (time_ns() - t0) / 1e9
            counters.terms_created[w] += created
            s_G == 0 || (counters.cross_appends[w] += created)
        end
        if i % circ.window == 0 || i == L
            t1 = time_ns()
            merge_shards!(S, fstrict; counters, w)
            if counters !== nothing
                counters.t_merge[w] += (time_ns() - t1) / 1e9
                gcnow = Base.gc_num()
                counters.allocd[w] = Base.GC_Diff(gcnow, gcbase).allocd
                gcbase = gcnow
            end
        end
    end
    return S
end
