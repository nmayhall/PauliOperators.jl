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
    skip_clean = (f == NOFILTER)
    maxpop = 0
    @inbounds for j in 1:nshards(S)
        sh = S.shards[j]
        m = _gather_append!(sh, S.cur, j, S.nthreads)
        if !(skip_clean && m == 0)
            # chunked growth is permitted here (window boundary), never mid-rotation
            sh.n + m > length(sh.z) && _grow_live!(sh, sh.n + m)
            _sort_ws!(sh.ws, 1, m)
            n_in, n_out = _merge_shard!(sh, m, f)
            if counters !== nothing
                counters.merge_in[w] += n_in
                counters.merge_out[w] += n_out
            end
        end
        sh.n > maxpop && (maxpop = sh.n)
        for t in 1:S.nthreads
            S.cur[t][j] = sh.seg_lo[t]
            S.mark[t][j] = sh.seg_lo[t]
        end
    end
    if counters !== nothing
        counters.max_shard_pop[w] = max(counters.max_shard_pop[w], maxpop)
    end
    return S
end

# One rotation over all shards (serial M1 driver), respecting the cursor/
# mark protocol: sweeps see only entries below the rotation-start mark; all
# appends land at or above it. Returns terms created.
function _rotate_all_serial!(S::ShardedPauliSum{N,W,T}, s_G::Int,
                             gz::W, gx::W, n_g::Int, cosθ::Float64, sinθ::Float64,
                             flocal::MergeFilter) where {N,W,T}
    created = 0
    t = 1
    @inbounds for k in 1:nshards(S)
        cr, ov = _rotate_shard!(S, k, t, s_G, gz, gx, n_g, cosθ, sinθ, flocal)
        created += cr
        ov && error("append segment overflow in shard $k despite precheck — this is a bug")
    end
    # publish marks: appends from this rotation become sweepable next rotation
    mk = S.mark[t]
    ck = S.cur[t]
    @inbounds for j in 1:nshards(S)
        mk[j] = ck[j]
    end
    return created
end

function _precheck_all_serial(S::ShardedPauliSum, s_G::Int)
    @inbounds for k in 1:nshards(S)
        _precheck_shard(S, k, 1, s_G) || return false
    end
    return true
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
early merge is triggered (harmless: it only changes truncation cadence);
if capacity is still insufficient, that is a configuration error.
"""
function evolve!(S::ShardedPauliSum{N,W,T}, circ::CompiledCircuit{N};
                 truncation::TruncationStrategy=NoTruncation(),
                 local_truncation::TruncationStrategy=NoTruncation(),
                 correction::CorrectionAccumulator=NoCorrection(),
                 counters::Union{Nothing,WindowCounters}=nothing) where {N,W,T}
    circ.version == S.version ||
        error("CompiledCircuit was compiled against RankMap version $(circ.version), " *
              "but the ShardedPauliSum is at version $(S.version). Recompile with `compile`.")
    S.nthreads == 1 ||
        error("the multithreaded driver lands in milestone 2; construct with nthreads=1")
    correction isa NoCorrection ||
        error("correction accumulators for the sharded engine land in milestone 3")
    fstrict = _compile_filter(truncation)
    flocal = _compile_filter(local_truncation)

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
        if !_precheck_all_serial(S, s_G)
            merge_shards!(S, fstrict; counters, w)
            counters === nothing || (counters.early_merges[w] += 1)
            # appends are empty post-merge, so segments may be regrown here
            for k in 1:nshards(S)
                if !_precheck_shard(S, k, 1, s_G)
                    j = ((k - 1) ⊻ s_G) + 1
                    _grow_append!(S.shards[j], S.nthreads, S.shards[k].n)
                    for t in 1:S.nthreads
                        S.cur[t][j] = S.shards[j].seg_lo[t]
                        S.mark[t][j] = S.shards[j].seg_lo[t]
                    end
                end
            end
        end
        t0 = time_ns()
        created = _rotate_all_serial!(S, s_G, gz[i], gx[i], ng[i], cosv[i], sinv[i], flocal)
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
