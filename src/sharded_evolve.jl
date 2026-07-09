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


# ------------------------------------------------------------
# Observables and the truncation-strategy interface
# ------------------------------------------------------------

"""
    expectation_value(S::ShardedPauliSum{N}, ψ::Ket{N})

⟨ψ|S|ψ⟩ for a computational-basis ket. Includes any pending (unmerged)
appends — expectation is linear, so the pre-merge state evaluates exactly.
"""
function expectation_value(S::ShardedPauliSum{N,W,T}, ψ::Ket{N}) where {N,W,T}
    kv = ψ.v % W
    acc = zero(T)
    for tid in 1:S.nthreads
        acc += _expectation_owned(S, tid, kv)
    end
    return acc
end

"""
    inner_product(S1::ShardedPauliSum, S2::ShardedPauliSum)

Liouville inner product tr(S1†·S2)/2^N over the shared basis terms
(coefficient-vector dot product, matching `inner_product(::PauliSum, ...)`).
Both engines must be sharded by the same `RankMap` and in merged state.
"""
function inner_product(S1::ShardedPauliSum{N,W,T}, S2::ShardedPauliSum{N,W,T}) where {N,W,T}
    S1.A.rows == S2.A.rows ||
        error("inner_product requires both engines sharded by the same RankMap")
    out = zero(T)
    for j in 1:nshards(S1)
        a = S1.shards[j]
        b = S2.shards[j]
        i = 1
        k = 1
        @inbounds while i <= a.n && k <= b.n
            ka = (a.z[i], a.x[i])
            kb = (b.z[k], b.x[k])
            if _key_eq(ka, kb)
                out += conj(a.c[i]) * b.c[k]
                i += 1
                k += 1
            elseif _key_lt(ka, kb)
                i += 1
            else
                k += 1
            end
        end
    end
    return out
end

"""
    greedy_bisection_rankmap(S::ShardedPauliSum, r; protected, ncandidates, rng)

Draw a balance-optimized `RankMap` against the engine's live population
(see the `Vector{PauliBasis}` method). Use it to build a better-balanced
replacement engine: construct a new `ShardedPauliSum` with the returned map
(the sharded engine deliberately has no in-place `swap_row!` — ownership
rebalancing is the cheap knob; changing `A` is a reconstruction).
"""
function greedy_bisection_rankmap(S::ShardedPauliSum{N,W,T}, r::Int; kw...) where {N,W,T}
    terms = Vector{PauliBasis{N}}(undef, length(S))
    i = 0
    for sh in S.shards
        for k in 1:sh.n
            i += 1
            terms[i] = _unpack(PauliBasis{N}, sh.z[k], sh.x[k])
        end
    end
    return greedy_bisection_rankmap(terms, r; kw...)
end

_measure(S::ShardedPauliSum, ::NoCorrection) = nothing
_measure(S::ShardedPauliSum{N,W,T}, corr::EnergyCorrection{N}) where {N,W,T} =
    (energy = real(expectation_value(S, corr.ψ)),)
_measure(S::ShardedPauliSum{N,W,T}, corr::EnergyVarianceCorrection{N}) where {N,W,T} =
    error("EnergyVarianceCorrection is not supported by the sharded engine " *
          "(variance requires operator products across shards)")

_adaptive_filter(thresh::Float64) =
    MergeFilter(typemax(Int), typemax(Int), typemax(Int), thresh, 0.0, -1.0, 0.0, -1.0)

function _compact_all!(S::ShardedPauliSum, f::MergeFilter)
    for tid in 1:S.nthreads
        _compact_owned!(S, tid, f)
    end
    return S
end

function _apply!(S::ShardedPauliSum, s::TruncationStrategy)
    return _compact_all!(S, _compile_filter(s))
end

function _apply!(S::ShardedPauliSum, s::AdaptiveTruncation)
    if length(S) > s.max_terms
        hist = zeros(Int, _HIST_BINS)
        for sh in S.shards
            _hist_shard!(hist, sh)
        end
        th = max(_hist_threshold(hist, s.max_terms), s.min_thresh)
        _compact_all!(S, _adaptive_filter(th))
    else
        _compact_all!(S, _adaptive_filter(s.min_thresh))
    end
    return S
end

"""
    truncate!(S::ShardedPauliSum, strategy, corr=NoCorrection())

Apply `strategy` to the merged engine state in place, with optional
correction accumulation — the sharded analogue of
`truncate!(::PauliSum, ...)`. Deterministic (weight/coefficient) strategies
apply exactly; `AdaptiveTruncation` picks its global threshold from a
64-bin |c| exponent histogram, so the kept count lands within a factor-of-2
bin quantization of the serial top-k semantics. Stochastic strategies are
not supported.
"""
function truncate!(S::ShardedPauliSum, strategy::TruncationStrategy,
                   corr::CorrectionAccumulator=NoCorrection())
    before = _measure(S, corr)
    _apply!(S, strategy)
    after = _measure(S, corr)
    _accumulate!(corr, before, after)
    return S
end

# Serial window/early boundary: measure → merge (strict filter) →
# [adaptive threshold update + optional immediate re-clip] → measure →
# accumulate. The re-clip sits inside the measurement span, so corrections
# capture both merge-time and re-clip losses.
function _boundary_serial!(S::ShardedPauliSum, fref::Base.RefValue{MergeFilter},
                           adapt::Union{Nothing,AdaptiveTruncation},
                           correction::CorrectionAccumulator,
                           hist::Vector{Int},
                           counters::Union{Nothing,WindowCounters}, w::Int;
                           adaptive_update::Bool=true)
    before = _measure(S, correction)
    merge_shards!(S, fref[]; counters, w)
    if adapt !== nothing && adaptive_update
        total = length(S)
        if total > adapt.max_terms
            fill!(hist, 0)
            for sh in S.shards
                _hist_shard!(hist, sh)
            end
            th = max(_hist_threshold(hist, adapt.max_terms), adapt.min_thresh)
            fref[] = _adaptive_filter(th)
            total > 2 * adapt.max_terms && _compact_all!(S, fref[])
        else
            fref[] = _adaptive_filter(adapt.min_thresh)
        end
    end
    after = _measure(S, correction)
    _accumulate!(correction, before, after)
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
    adapt = truncation isa AdaptiveTruncation ? truncation : nothing
    fref = Ref(adapt === nothing ? _compile_filter(truncation)
                                 : _adaptive_filter(adapt.min_thresh))
    flocal = _compile_filter(local_truncation)
    if S.nthreads > 1
        correction isa Union{NoCorrection,EnergyCorrection} ||
            error("the threaded driver supports NoCorrection and EnergyCorrection " *
                  "(use nthreads=1 for custom correction accumulators)")
        return _evolve_threaded!(S, circ, fref, adapt, flocal, correction, counters,
                                 Float64(rebalance_threshold))
    end
    hist = zeros(Int, _HIST_BINS)

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
            # early merge: corrections still accumulate; the adaptive
            # threshold is only refreshed at regular window boundaries
            _boundary_serial!(S, fref, adapt, correction, hist, counters, w;
                              adaptive_update=false)
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
            _boundary_serial!(S, fref, adapt, correction, hist, counters, w)
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
