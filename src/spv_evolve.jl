# ============================================================
# SparsePauliVector: clip family, truncation-strategy integration, and the
# windowed evolution driver. Semantics contract: at `window = 1` with a
# deterministic truncation strategy, `evolve!` reproduces the Dict-backed
# `evolve!`/`truncate!` sequence exactly (dedup every rotation, identical
# drop decisions, identical correction accumulation).
# ============================================================

# ------------------------------------------------------------
# Clip family (parity with clip.jl; all in-place, zero-alloc)
# ------------------------------------------------------------

coeff_clip!(v::SparsePauliVector, thresh::Real) =
    _compact_spv!(v, _compile_filter(CoeffTruncation(Float64(thresh))))
clip!(v::SparsePauliVector; thresh=1e-16) = coeff_clip!(v, thresh)

weight_clip!(v::SparsePauliVector, max_weight::Int) =
    _compact_spv!(v, _compile_filter(WeightTruncation(max_weight)))
weight_damped_clip!(v::SparsePauliVector, alpha::Real, thresh::Real) =
    _compact_spv!(v, _compile_filter(WeightDampedTruncation(Float64(alpha), Float64(thresh))))
x_weight_clip!(v::SparsePauliVector, max_weight::Int) =
    _compact_spv!(v, _compile_filter(XWeightTruncation(max_weight)))
x_weight_damped_clip!(v::SparsePauliVector, alpha::Real, thresh::Real) =
    _compact_spv!(v, _compile_filter(XWeightDampedTruncation(Float64(alpha), Float64(thresh))))
majorana_weight_clip!(v::SparsePauliVector, max_weight::Int) =
    _compact_spv!(v, _compile_filter(MajoranaWeightTruncation(max_weight)))

"""
    stochastic_clip!(v::SparsePauliVector, ε::Real; rng=Random.default_rng())

Unbiased stochastic compression (Russian Roulette), the SparsePauliVector
analogue of `stochastic_clip!(::PauliSum, ...)`. In-place, order-preserving,
allocation-free (modulo RNG state).
"""
function stochastic_clip!(v::SparsePauliVector{N,W,T}, ε::Real;
                          rng::AbstractRNG=Random.default_rng()) where {N,W,T}
    out = 0
    @inbounds for i in 1:v.n
        c = v.c[i]
        ac = abs(c)
        if ac < ε
            rand(rng) < ac / ε || continue
            c = convert(T, ε * (c / ac))   # promote (preserves phase for complex T)
        end
        out += 1
        v.z[out] = v.z[i]
        v.x[out] = v.x[i]
        v.c[out] = c
    end
    v.n = out
    return v
end

# ------------------------------------------------------------
# Truncation-strategy protocol (_apply! per strategy; _measure/truncate!
# are shared with PauliSum via AnyPauliSum in truncation.jl)
# ------------------------------------------------------------

_apply!(v::SparsePauliVector, ::NoTruncation) = v
_apply!(v::SparsePauliVector, s::CoeffTruncation) = coeff_clip!(v, s.thresh)
_apply!(v::SparsePauliVector, s::WeightTruncation) = weight_clip!(v, s.max_weight)
_apply!(v::SparsePauliVector, s::XWeightTruncation) = x_weight_clip!(v, s.max_weight)
_apply!(v::SparsePauliVector, s::MajoranaWeightTruncation) = majorana_weight_clip!(v, s.max_weight)
_apply!(v::SparsePauliVector, s::WeightDampedTruncation) = weight_damped_clip!(v, s.alpha, s.thresh)
_apply!(v::SparsePauliVector, s::XWeightDampedTruncation) = x_weight_damped_clip!(v, s.alpha, s.thresh)
_apply!(v::SparsePauliVector, s::StochasticCoeffTruncation) =
    stochastic_clip!(v, s.epsilon; rng=s.rng)

function _apply!(v::SparsePauliVector{N,W,T}, s::StochasticSamplingTruncation) where {N,W,T}
    v.n <= s.n_keep && return v
    weights = [abs2(v.c[i]) for i in 1:v.n]
    norm_sq = sum(weights)
    sampling_keys = [rand(s.rng)^(1.0 / w) for w in weights]
    kept_idx = partialsortperm(sampling_keys, 1:s.n_keep, rev=true)
    keep = falses(v.n)
    for i in kept_idx
        keep[i] = true
    end
    kept_norm_sq = 0.0
    for i in eachindex(weights)
        keep[i] && (kept_norm_sq += weights[i])
    end
    out = 0
    @inbounds for i in 1:v.n
        keep[i] || continue
        out += 1
        v.z[out] = v.z[i]
        v.x[out] = v.x[i]
        v.c[out] = v.c[i]
    end
    v.n = out
    if kept_norm_sq > 0
        scale = sqrt(norm_sq / kept_norm_sq)
        @inbounds for i in 1:v.n
            v.c[i] *= scale
        end
    end
    return v
end

# Exact parity with _apply!(::PauliSum, ::AdaptiveTruncation): sort-based
# top-k threshold, not the histogram approximation.
function _apply!(v::SparsePauliVector, s::AdaptiveTruncation)
    if v.n > s.max_terms
        coeffs = sort!(abs.(view(v.c, 1:v.n)))
        if length(coeffs) > s.max_terms
            thresh = coeffs[end - s.max_terms]
            coeff_clip!(v, thresh)
        end
    else
        coeff_clip!(v, s.min_thresh)
    end
    return v
end

function _apply!(v::SparsePauliVector, s::CompositeTruncation)
    _apply_tup!(v, s.strategies)
    return v
end

# ------------------------------------------------------------
# Expectation value against computational-basis kets (hot; needed by the
# correction accumulators — the full observable set lives in spv_ops.jl)
# ------------------------------------------------------------

"""
    expectation_value(v::SparsePauliVector{N}, ψ::Ket{N})

⟨ψ|v|ψ⟩ for a computational-basis ket. Includes any pending (unmerged)
appends — expectation is linear, so the pre-merge state evaluates exactly.
Allocation-free.
"""
expectation_value(v::SparsePauliVector{N,W,T}, ψ::Ket{N,W}) where {N,W,T} =
    _expectation_spv(v, ψ.v)

# ------------------------------------------------------------
# Windowed evolution driver
# ------------------------------------------------------------

"""
    WindowCounters(nwindows)

Preallocated per-window instrumentation for `evolve!` (design invariant:
everything measurable, nothing allocated in the hot path). `allocd[w]` is
the `Base.gc_num()` allocation delta across window `w` — any nonzero entry
after warm-up is a bug, enforced by the test suite. Early (capacity-forced)
merges accumulate into the window they occur in.
"""
struct WindowCounters
    terms_created::Vector{Int}
    merge_in::Vector{Int}
    merge_out::Vector{Int}
    t_rotate::Vector{Float64}
    t_merge::Vector{Float64}
    allocd::Vector{Int64}
    early_merges::Vector{Int}
end
WindowCounters(nw::Int) = WindowCounters(zeros(Int, nw), zeros(Int, nw),
                                         zeros(Int, nw), zeros(Float64, nw),
                                         zeros(Float64, nw), zeros(Int64, nw),
                                         zeros(Int, nw))

"""
    merge_pending!(v::SparsePauliVector, f::MergeFilter=NOFILTER)

Sort-merge the pending appends into the sorted live buffer under filter
`f`, restoring the public merged-state invariant (`v.an == 0`).
Allocation-free in steady state.
"""
function merge_pending!(v::SparsePauliVector, f::MergeFilter=NOFILTER)
    m = _gather_append!(v)
    _sort_ws!(v.ws, 1, m)
    _merge_spv!(v, m, f)
    return v
end

"""
    merge_pending!(v, f, gz, gx)

`merge_pending!` for appends produced by a *single* rotation under the
generator mask `(gz, gx)`: the gathered triples are then sorted with
respect to `key ⊻ mask`, so `_unshuffle_ws!` restores natural order in
`weight(G)` linear passes instead of a comparison sort.
"""
function merge_pending!(v::SparsePauliVector{N,W}, f::MergeFilter, gz::W, gx::W) where {N,W}
    m = _gather_append!(v)
    _unshuffle_ws!(v, m, gz, gx)
    _merge_spv!(v, m, f)
    return v
end

# Sort the m gathered workspace triples: with a single-rotation mask, use
# the linear unshuffle cascade; otherwise the general comparison sort.
function _sort_pending!(O::SparsePauliVector{N,W}, m::Int,
                        mask::Union{Nothing,Tuple{W,W}}) where {N,W}
    if mask === nothing
        _sort_ws!(O.ws, 1, m)
    else
        _unshuffle_ws!(O, m, mask[1], mask[2])
    end
    return nothing
end

# Rotate the full pre-merge state (live + pending appends) under one
# generator, appending sin branches. Returns created count. The caller
# guarantees capacity (worst case: every swept term appends once).
function _rotate_spv!(v::SparsePauliVector{N,W,T}, gz::W, gx::W, ng::Int,
                      cosθ::Float64, sinθ::Float64, f::MergeFilter) where {N,W,T}
    hi = v.an            # snapshot: appends created below land above hi
    cur = v.an + 1
    cap = length(v.az)
    cur, cr1, ov1 = _rotate_range!(v.z, v.x, v.c, 1, v.n, gz, gx, ng,
                                   cosθ, sinθ, v.az, v.ax, v.ac, cur, cap, f)
    cur, cr2, ov2 = _rotate_range!(v.az, v.ax, v.ac, 1, hi, gz, gx, ng,
                                   cosθ, sinθ, v.az, v.ax, v.ac, cur, cap, f)
    v.an = cur - 1
    return cr1 + cr2, ov1 | ov2
end

"""
    evolve!(O::SparsePauliVector{N}, G::PauliBasis{N}, θ::Real)

In-place Heisenberg-picture evolution O ← exp(iθ/2 G) O exp(-iθ/2 G), the
flat-storage analogue of `evolve!(::PauliSum, G, θ)` (identical semantics:
sin branches are deduplicated into the sum immediately). Zero-allocation
once buffers have warmed up.
"""
function evolve!(O::SparsePauliVector{N,W,T}, G::PauliBasis{N}, θ::Real) where {N,W,T}
    gz, gx = _pack(W, G)
    ng = count_ones(gz & gx)
    O.n + 2 * O.an > length(O.az) && _grow_append!(O, O.n + 2 * O.an)
    _, ovf = _rotate_spv!(O, gz, gx, ng, cos(θ), sin(θ), NOFILTER)
    ovf && error("append overflow despite precheck — this is a bug")
    merge_pending!(O, NOFILTER, gz, gx)
    return O
end

"""
    evolve(O::SparsePauliVector{N}, G::PauliBasis{N}, θ::Real)

Non-mutating single-rotation evolution (see `evolve!`).
"""
evolve(O::SparsePauliVector{N}, G::PauliBasis{N}, θ::Real) where {N} =
    evolve!(copy(O), G, θ)

# Linear correction measures (expectation values) evaluate exactly on the
# pre-merge state (live + pending appends); non-linear ones (variance) need
# merged state first, so their boundary merges with NOFILTER, measures,
# then truncates in a separate compaction pass. Default `true` is the safe
# choice for user-defined accumulators.
_needs_merged_measure(::NoCorrection) = false
_needs_merged_measure(::EnergyCorrection) = false
_needs_merged_measure(::CorrectionAccumulator) = true

# Window/early boundary: measure → merge (strict filter) → generic _apply!
# for non-compilable strategies → measure → accumulate. Corrections capture
# exactly the truncation loss.
function _boundary!(O::SparsePauliVector{N,W}, f::MergeFilter, strategy::S,
                    compiled::Bool, correction::CorrectionAccumulator,
                    counters::Union{Nothing,WindowCounters},
                    w::Int,
                    mask::Union{Nothing,Tuple{W,W}}=nothing) where {N,W,S<:TruncationStrategy}
    local before, n_in, n_out
    if _needs_merged_measure(correction)
        m = _gather_append!(O)
        _sort_pending!(O, m, mask)
        n_in, n_out = _merge_spv!(O, m, NOFILTER)
        before = _measure(O, correction)
        compiled ? _compact_spv!(O, f) : _apply!(O, strategy)
    else
        before = _measure(O, correction)
        m = _gather_append!(O)
        _sort_pending!(O, m, mask)
        n_in, n_out = _merge_spv!(O, m, f)
        compiled || _apply!(O, strategy)
    end
    after = _measure(O, correction)
    _accumulate!(correction, before, after)
    if counters !== nothing
        counters.merge_in[w] += n_in
        counters.merge_out[w] += n_out
    end
    return O
end

"""
    evolve!(O::SparsePauliVector{N}, generators::Vector{PauliBasis{N}}, angles;
            window=1, truncation=NoTruncation(), local_truncation=NoTruncation(),
            correction=NoCorrection(), counters=nothing)

Heisenberg-picture sequence evolution on flat storage, applying generators
left to right (same convention as `evolve(::PauliSum, generators, angles)`;
sequences from `trotterize`/`qdrift` plug in directly).

Rotations append sin branches under the loose `local_truncation` (applied
per term at append time — weight cutoffs are exact there, coefficient
cutoffs act on unmerged duplicates); every `window` rotations the appends
are sort-merged into the live buffer under the strict `truncation`, and
`correction` accumulates the truncation losses.

At `window = 1` (the default) this reproduces the Dict path
`evolve!(O, g, θ); truncate!(O, truncation, correction)` per rotation
exactly, for every truncation strategy. `window > 1` trades truncation
cadence for speed: deduplication and truncation happen once per window.

If a rotation's worst-case appends cannot fit the append buffer, an early
merge is triggered (harmless: it only changes truncation cadence), growing
the buffer at the boundary if the population genuinely needs more room.
The steady-state hot path allocates zero bytes — pass a `WindowCounters`
to verify (`counters.allocd`).
"""
function evolve!(O::SparsePauliVector{N,W,T}, generators::Vector{PauliBasis{N,W}},
                 angles::Vector{<:Real};
                 window::Int=1,
                 truncation::TruncationStrategy=NoTruncation(),
                 local_truncation::TruncationStrategy=NoTruncation(),
                 correction::CorrectionAccumulator=NoCorrection(),
                 counters::Union{Nothing,WindowCounters}=nothing) where {N,W,T}
    length(generators) == length(angles) ||
        throw(DimensionMismatch("generators and angles must have same length"))
    window >= 1 || throw(ArgumentError("window must be >= 1"))
    L = length(generators)
    L == 0 && return O

    compiled = _is_compilable(truncation)
    f = compiled ? _compile_filter(truncation) : NOFILTER
    _is_compilable(local_truncation) ||
        throw(ArgumentError("local_truncation must be a deterministic " *
                            "(weight/coefficient) strategy — it runs per append"))
    flocal = _compile_filter(local_truncation)

    # Setup allocations (packed generators), outside the gc_num baseline.
    gz = Vector{W}(undef, L)
    gx = Vector{W}(undef, L)
    ng = Vector{Int}(undef, L)
    cosv = Vector{Float64}(undef, L)
    sinv = Vector{Float64}(undef, L)
    for i in 1:L
        gz[i], gx[i] = _pack(W, generators[i])
        ng[i] = count_ones(gz[i] & gx[i])
        cosv[i] = cos(angles[i])
        sinv[i] = sin(angles[i])
    end

    gcbase = Base.gc_num()
    # Appends from exactly one rotation are XOR-shuffled sorted keys, which
    # _boundary! can unshuffle in weight(G) linear passes instead of a full
    # sort; track how many rotations contributed since the last merge.
    since = 0
    lz = zero(W)
    lx = zero(W)
    @inbounds for i in 1:L
        w = cld(i, window)
        # Worst case: every live term AND every pending append anticommutes
        # and appends one sin branch.
        if 2 * O.an + O.n > length(O.az)
            _boundary!(O, f, truncation, compiled, correction, counters, w,
                       since == 1 ? (lz, lx) : nothing)
            since = 0
            counters === nothing || (counters.early_merges[w] += 1)
            O.n > length(O.az) && _grow_append!(O, O.n)
        end
        t0 = time_ns()
        created, ovf = _rotate_spv!(O, gz[i], gx[i], ng[i], cosv[i], sinv[i], flocal)
        ovf && error("append overflow despite precheck (rotation $i) — this is a bug")
        lz = gz[i]
        lx = gx[i]
        since += 1
        if counters !== nothing
            counters.t_rotate[w] += (time_ns() - t0) / 1e9
            counters.terms_created[w] += created
        end
        if i % window == 0 || i == L
            t1 = time_ns()
            _boundary!(O, f, truncation, compiled, correction, counters, w,
                       since == 1 ? (lz, lx) : nothing)
            since = 0
            if counters !== nothing
                counters.t_merge[w] += (time_ns() - t1) / 1e9
                gcnow = Base.gc_num()
                counters.allocd[w] = Base.GC_Diff(gcnow, gcbase).allocd
                gcbase = gcnow
            end
        end
    end
    return O
end

"""
    evolve(O::SparsePauliVector{N}, generators, angles; kwargs...)

Non-mutating sequence evolution (see `evolve!`).
"""
evolve(O::SparsePauliVector{N,W}, generators::Vector{PauliBasis{N,W}},
       angles::Vector{<:Real}; kwargs...) where {N,W} =
    evolve!(copy(O), generators, angles; kwargs...)
