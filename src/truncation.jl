using Random

# ============================================================
# Abstract Types
# ============================================================

"""
    TruncationStrategy

Abstract supertype for term-truncation strategies applied by `truncate!` and
the `truncation`/`local_truncation` keywords of `evolve!`. Define a new
strategy by subtyping and implementing `_apply!(O, s)`.
"""
abstract type TruncationStrategy end

"""
    CorrectionAccumulator

Abstract supertype for truncation-error trackers passed to `truncate!` and
`evolve!`: observables are measured before and after each truncation and the
differences accumulate. See `EnergyCorrection`, `EnergyVarianceCorrection`,
`NoCorrection`. Define a new accumulator by subtyping and implementing
`_measure(O, corr)` and `_accumulate!(corr, before, after)`.
"""
abstract type CorrectionAccumulator end


# ============================================================
# Truncation Strategy Types
# ============================================================

"""
    NoTruncation()

Identity truncation — does nothing.
"""
struct NoTruncation <: TruncationStrategy end

"""
    CoeffTruncation(thresh::Float64)

Remove Pauli terms with |coefficient| <= `thresh`.
"""
struct CoeffTruncation <: TruncationStrategy
    thresh::Float64
end
CoeffTruncation() = CoeffTruncation(1e-6)

"""
    WeightTruncation(max_weight::Int)

Remove Pauli terms with Pauli weight > `max_weight`.
"""
struct WeightTruncation <: TruncationStrategy
    max_weight::Int
end

"""
    XWeightTruncation(max_weight::Int)

Remove Pauli terms with X-weight (number of X/Y factors) > `max_weight`.
"""
struct XWeightTruncation <: TruncationStrategy
    max_weight::Int
end

"""
    MajoranaWeightTruncation(max_weight::Int)

Remove Pauli terms with Majorana weight > `max_weight`.
"""
struct MajoranaWeightTruncation <: TruncationStrategy
    max_weight::Int
end

"""
    WeightDampedTruncation(alpha::Float64, thresh::Float64)

Remove Pauli terms with |coefficient|·exp(-alpha·weight) <= `thresh`,
i.e. a coefficient threshold that grows exponentially with Pauli weight.
`alpha = 0` reduces to `CoeffTruncation(thresh)`; large `alpha` approaches
a hard weight cutoff.
"""
struct WeightDampedTruncation <: TruncationStrategy
    alpha::Float64
    thresh::Float64
end
WeightDampedTruncation(alpha::Real) = WeightDampedTruncation(alpha, 1e-6)

"""
    XWeightDampedTruncation(alpha::Float64, thresh::Float64)

Remove Pauli terms with |coefficient|·exp(-alpha·x_weight) <= `thresh`,
i.e. a coefficient threshold that grows exponentially with X-weight (the
number of X/Y factors). `alpha = 0` reduces to `CoeffTruncation(thresh)`;
large `alpha` approaches a hard X-weight cutoff.
"""
struct XWeightDampedTruncation <: TruncationStrategy
    alpha::Float64
    thresh::Float64
end
XWeightDampedTruncation(alpha::Real) = XWeightDampedTruncation(alpha, 1e-6)

"""
    StochasticCoeffTruncation(epsilon::Float64; rng=Random.default_rng())

Unbiased stochastic compression (Russian Roulette). Wraps `stochastic_clip!`.

For each term with |c| < epsilon:
- Keep with probability |c|/epsilon (promote to epsilon·sign(c))
- Delete with probability 1 - |c|/epsilon
"""
struct StochasticCoeffTruncation <: TruncationStrategy
    epsilon::Float64
    rng::AbstractRNG
end
StochasticCoeffTruncation(epsilon::Float64) = StochasticCoeffTruncation(epsilon, Random.default_rng())

"""
    StochasticSamplingTruncation(n_keep::Int; rng=Random.default_rng())

Stochastically sample `n_keep` terms via importance sampling with probabilities
proportional to |c_i|^2. Kept terms are rescaled to preserve norm.
"""
struct StochasticSamplingTruncation <: TruncationStrategy
    n_keep::Int
    rng::AbstractRNG
end
StochasticSamplingTruncation(n_keep::Int) = StochasticSamplingTruncation(n_keep, Random.default_rng())

"""
    AdaptiveTruncation(max_terms::Int, min_thresh::Float64)

If the number of terms exceeds `max_terms`, increase the clipping threshold
to reduce the operator size. Otherwise clip at `min_thresh`.
"""
struct AdaptiveTruncation <: TruncationStrategy
    max_terms::Int
    min_thresh::Float64
end
AdaptiveTruncation(; max_terms::Int=10000, min_thresh::Float64=1e-12) = AdaptiveTruncation(max_terms, min_thresh)

"""
    MeanFieldTruncation(max_weight::Int, reference::Ket{N})

Expectation-preserving order-`k` (`k = max_weight`) mean-field truncation around the
computational-basis `reference`. Every term with Pauli weight > `k` is replaced by its
order-`k` fluctuation factorization (`δP_j = P_j − ⟨P_j⟩I`), a sum of strings of weight
≤ `k`. Exact when `k ≥ weight`, and equal to the state-adapted covariance projection π_k
of arXiv:2609.12840. Preserves `⟨reference|O|reference⟩` for every `k`.

Unlike the drop strategies, this modifies survivors (it folds truncated weight back onto
lower-order terms), so it is not "pure-drop": `truncate!` routes it through the measured
before/after correction path rather than the fused delta. Pass the *same* `reference` to
any `EnergyCorrection`/`EnergyVarianceCorrection` so the energy delta registers as ≈ 0.

Currently supported on `PauliSum` only; on a `SparsePauliVector` it errors (convert to a
`PauliSum` first).
"""
struct MeanFieldTruncation{N} <: TruncationStrategy
    max_weight::Int
    reference::Ket{N}
end

"""
    CompositeTruncation(strategies...)

Apply multiple truncation strategies in sequence.

Strategies are stored as a typed `Tuple` rather than `Vector{TruncationStrategy}`,
so the per-element dispatches inside `_apply!` resolve at compile time and the
inner `coeff_clip!` / `weight_clip!` calls inline. Constructing via the variadic
form (`CompositeTruncation(CoeffTruncation(1e-4), WeightTruncation(5))`) is
the supported call style; an `AbstractVector` constructor is also provided
for convenience but converts to a tuple internally.
"""
struct CompositeTruncation{S<:Tuple} <: TruncationStrategy
    strategies::S
end
CompositeTruncation(s::TruncationStrategy...) = CompositeTruncation(s)
CompositeTruncation(v::AbstractVector{<:TruncationStrategy}) = CompositeTruncation(Tuple(v))


# ============================================================
# _apply! — raw truncation dispatch (internal)
# ============================================================

function _apply!(O::PauliSum{N}, ::NoTruncation) where N
    return O
end

function _apply!(O::PauliSum{N}, s::CoeffTruncation) where N
    return coeff_clip!(O, s.thresh)
end

function _apply!(O::PauliSum{N}, s::WeightTruncation) where N
    return weight_clip!(O, s.max_weight)
end

function _apply!(O::PauliSum{N}, s::XWeightTruncation) where N
    return x_weight_clip!(O, s.max_weight)
end

function _apply!(O::PauliSum{N}, s::MajoranaWeightTruncation) where N
    return majorana_weight_clip!(O, s.max_weight)
end

function _apply!(O::PauliSum{N}, s::WeightDampedTruncation) where N
    return weight_damped_clip!(O, s.alpha, s.thresh)
end

function _apply!(O::PauliSum{N}, s::XWeightDampedTruncation) where N
    return x_weight_damped_clip!(O, s.alpha, s.thresh)
end

function _apply!(O::PauliSum{N}, s::StochasticCoeffTruncation) where N
    return stochastic_clip!(O, s.epsilon; rng=s.rng)
end

function _apply!(O::PauliSum{N}, s::StochasticSamplingTruncation) where N
    length(O) <= s.n_keep && return O

    keys_vec = collect(keys(O))
    weights = [abs2(O[k]) for k in keys_vec]
    norm_sq = sum(weights)
    sampling_keys = [rand(s.rng)^(1.0/w) for w in weights]

    kept_idx = partialsortperm(sampling_keys, 1:s.n_keep, rev=true)
    kept_set = Set(keys_vec[i] for i in kept_idx)

    kept_norm_sq = sum(abs2(O[k]) for k in kept_set)
    filter!(p -> p.first in kept_set, O)

    if kept_norm_sq > 0
        scale = sqrt(norm_sq / kept_norm_sq)
        for k in keys(O)
            O[k] *= scale
        end
    end

    return O
end

function _apply!(O::PauliSum{N}, s::AdaptiveTruncation) where N
    if length(O) > s.max_terms
        coeffs = sort(abs.(collect(values(O))))
        if length(coeffs) > s.max_terms
            thresh = coeffs[end - s.max_terms]
            coeff_clip!(O, thresh)
        end
    else
        coeff_clip!(O, s.min_thresh)
    end
    return O
end

function _apply!(O::PauliSum{N}, s::MeanFieldTruncation{N}) where N
    return mean_field_factorize!(O, s.reference, s.max_weight)
end

# Recursive tail-pop iteration over the heterogeneous tuple of strategies so
# each `_apply!(O, strategy)` resolves at compile time and inlines.
@inline _apply_tup!(O, ::Tuple{}) = O
@inline _apply_tup!(O, s::Tuple)  = (_apply!(O, first(s)); _apply_tup!(O, Base.tail(s)))

function _apply!(O::PauliSum{N}, s::CompositeTruncation) where N
    _apply_tup!(O, s.strategies)
    return O
end


# ============================================================
# Correction Accumulator Types
# ============================================================

"""
    NoCorrection()

Track nothing during truncation. Zero overhead.
"""
struct NoCorrection <: CorrectionAccumulator end

"""
    EnergyCorrection(ψ::Ket{N})

Track accumulated change in ⟨ψ|O|ψ⟩ due to truncation.
"""
mutable struct EnergyCorrection{N} <: CorrectionAccumulator
    ψ::Ket{N}
    accumulated_energy::Float64
end
EnergyCorrection(ψ::Ket{N}) where N = EnergyCorrection{N}(ψ, 0.0)

"""
    EnergyVarianceCorrection(ψ::Ket{N})

Track accumulated changes in both ⟨ψ|O|ψ⟩ and Var(O,ψ) due to truncation.
"""
mutable struct EnergyVarianceCorrection{N} <: CorrectionAccumulator
    ψ::Ket{N}
    accumulated_energy::Float64
    accumulated_variance::Float64
end
EnergyVarianceCorrection(ψ::Ket{N}) where N = EnergyVarianceCorrection{N}(ψ, 0.0, 0.0)


# ============================================================
# Single-pass truncation deltas (fast corrections)
# ============================================================
# A truncation is "pure-drop" when every surviving coefficient is left
# unchanged: O_after = O_before - B, with B exactly the deleted terms.
# Any strategy compilable to a MergeFilter is pure-drop by construction --
# `should_drop` can only keep or discard a term, never modify it. For these,
# the corrections can be computed exactly from the dropped terms alone:
#
#     Δ⟨O⟩  = -⟨B⟩
#     ΔVar  = -( Var(B) + 2·cov(A,B) ),  cov(A,B) = Re⟨a|b⟩ - ⟨A⟩⟨B⟩
#
# where b = B|ψ⟩ lives on at most (#dropped) kets and a = A|ψ⟩ is only ever
# needed on b's support. So instead of the two full ⟨O²⟩ evaluations of the
# before/after `_measure` route (each builds an O(len(O)) KetSum), the exact
# correction costs O(#dropped) at the drop sites plus ONE sweep of the kept
# terms with lookups into a (#dropped)-entry dict.
#
# Everything NOT expressible as a keep/drop filter -- strategies that rescale
# survivors or choose data-dependent thresholds, user-defined strategies, and
# user-defined accumulators -- takes the measured before/after fallback in
# `truncate!` below, which is correct for arbitrary `_apply!` behavior and
# doubles as the reference implementation the delta path is tested against.
#
# `TruncationDelta` collects the dropped terms. It is passed into the
# merge/compact kernels, which call `_sink_drop!(Δ, z, x, c)` at the branch
# where they discard a term ("here is a term I am throwing away"); the kernel
# is otherwise unchanged, and passing `nothing` compiles the callback away.
# After the pass, `_finalize_delta!` folds the collected B into the
# correction accumulator. Ket bits are keyed as Int128 (`% Int128`
# reinterprets the SPV's unsigned words losslessly).

mutable struct TruncationDelta
    ψv::Int128                      # reference ket bits
    b::Dict{Int128,ComplexF64}      # B|ψ⟩ amplitudes, keyed by ket bits
end
TruncationDelta(ψ::Ket) = TruncationDelta(ψ.v, Dict{Int128,ComplexF64}())

# apply the (z,x) Pauli word to the ket bits kv: same math as
# Base.:*(::PauliBasis, ::Ket) in multiplication.jl, on raw words
@inline function _ket_action(z::Int128, x::Int128, kv::Int128)
    kv2 = x ⊻ kv
    idx = ((4 - count_ones(z & x) % 4) % 4 + 2 * (count_ones(z & kv2) % 2)) % 4 + 1
    return PHASE_TBL[idx], kv2
end

@inline _sink_drop!(::Nothing, z, x, c) = nothing
@inline function _sink_drop!(Δ::TruncationDelta, z, x, c)
    ph, kv2 = _ket_action(z % Int128, x % Int128, Δ.ψv)
    Δ.b[kv2] = get(Δ.b, kv2, zero(ComplexF64)) + ph * c
    return nothing
end

# ------------------------------------------------------------
# SparsePauliVector run sinks — exact corrections from the x-major walk
# ------------------------------------------------------------
# The SPV truncation kernels (_compact_spv!, _try_merge!) visit terms in
# x-major sorted order, so all terms sharing an x-string — i.e. all terms
# mapping the reference ket |ψ⟩ to the SAME target ket ±i^k|ψ ⊻ x⟩ — form a
# contiguous run. Within a run the A|ψ⟩ / B|ψ⟩ amplitudes on the run's
# target ket are plain sums of ph·c over kept / dropped terms, so the exact
# ingredients of the delta formula,
#     bb = ⟨Bψ|Bψ⟩ = Σ_runs |Σ_drop ph·c|²
#     ab = ⟨Aψ|Bψ⟩ = Σ_runs conj(Σ_keep ph·c)·(Σ_drop ph·c)
#     eA = ⟨ψ|A|ψ⟩,  eB = ⟨ψ|B|ψ⟩          (the x = 0 run's two sums)
# accumulate with O(1) state during the single pass the kernel already
# makes: no dict, and — unlike the Dict-backed TruncationDelta route — no
# second sweep over the kept terms. This is the reason the canonical SPV
# order is x-major (see _key_lt in spv_kernels.jl). Correctness relies on
# the kernel walk being x-monotone, which the sorted live buffer
# (_compact_spv!) and the sorted two-stream merge (_try_merge!) guarantee.

# phase of (z,x) acting on |kv⟩ — the phase half of _ket_action, on the
# SPV's packed word type (the target ket is kv ⊻ x, implicit in the run)
@inline function _ket_phase(z::W, x::W, kv::W) where {W<:Unsigned}
    kv2 = x ⊻ kv
    idx = ((4 - count_ones(z & x) % 4) % 4 + 2 * (count_ones(z & kv2) % 2)) % 4 + 1
    return PHASE_TBL[idx]
end

"""
    EnergyDropSink{W}

Drop sink for `EnergyCorrection` on SparsePauliVector kernels: Δ⟨O⟩ = −⟨B⟩
needs only the dropped diagonal (x = 0) terms, each contributing ±c by the
z/ψ parity. No run tracking, no phases beyond a sign, kept terms ignored.
"""
mutable struct EnergyDropSink{W<:Unsigned}
    ψv::W
    eB::ComplexF64
end

"""
    XRunDelta{W}

Drop/keep sink for `EnergyVarianceCorrection` on SparsePauliVector kernels.
Carries the current x-run's kept/dropped amplitude sums and the finished
cross-run accumulators (see the block comment above).
"""
mutable struct XRunDelta{W<:Unsigned}
    ψv::W
    started::Bool
    cur_x::W
    kept_run::ComplexF64
    drop_run::ComplexF64
    bb::Float64
    ab::ComplexF64
    eA::ComplexF64
    eB::ComplexF64
end
XRunDelta(ψv::W) where {W<:Unsigned} =
    XRunDelta{W}(ψv, false, zero(W), zero(ComplexF64), zero(ComplexF64),
                 0.0, zero(ComplexF64), zero(ComplexF64), zero(ComplexF64))

@inline function _flush_run!(Δ::XRunDelta{W}) where W
    Δ.started || return nothing
    Δ.bb += abs2(Δ.drop_run)
    Δ.ab += conj(Δ.kept_run) * Δ.drop_run
    if Δ.cur_x == zero(W)
        Δ.eA += Δ.kept_run
        Δ.eB += Δ.drop_run
    end
    Δ.kept_run = zero(ComplexF64)
    Δ.drop_run = zero(ComplexF64)
    return nothing
end

@inline function _run_advance!(Δ::XRunDelta{W}, x::W) where W
    if !Δ.started
        Δ.started = true
        Δ.cur_x = x
    elseif x != Δ.cur_x
        _flush_run!(Δ)
        Δ.cur_x = x
    end
    return nothing
end

# keep hooks (only XRunDelta needs to see survivors)
@inline _sink_keep!(::Nothing, z, x, c) = nothing
@inline _sink_keep!(::TruncationDelta, z, x, c) = nothing
@inline _sink_keep!(::EnergyDropSink, z, x, c) = nothing
@inline function _sink_keep!(Δ::XRunDelta{W}, z::W, x::W, c) where W
    _run_advance!(Δ, x)
    Δ.kept_run += _ket_phase(z, x, Δ.ψv) * c
    return nothing
end

@inline function _sink_drop!(Δ::EnergyDropSink{W}, z::W, x::W, c) where W
    x == zero(W) || return nothing
    Δ.eB += (1 - 2 * (count_ones(z & Δ.ψv) & 1)) * c
    return nothing
end
@inline function _sink_drop!(Δ::XRunDelta{W}, z::W, x::W, c) where W
    _run_advance!(Δ, x)
    Δ.drop_run += _ket_phase(z, x, Δ.ψv) * c
    return nothing
end

# reset hooks — _try_merge! restarts its walk after an overflow grow-and-retry
@inline _sink_reset!(::Nothing) = nothing
@inline _sink_reset!(Δ::TruncationDelta) = (empty!(Δ.b); nothing)
@inline _sink_reset!(Δ::EnergyDropSink) = (Δ.eB = zero(ComplexF64); nothing)
@inline function _sink_reset!(Δ::XRunDelta{W}) where W
    Δ.started = false
    Δ.cur_x = zero(W)
    Δ.kept_run = zero(ComplexF64)
    Δ.drop_run = zero(ComplexF64)
    Δ.bb = 0.0
    Δ.ab = zero(ComplexF64)
    Δ.eA = zero(ComplexF64)
    Δ.eB = zero(ComplexF64)
    return nothing
end

# The union the SPV kernels accept as a drop sink.
const SPVSink = Union{Nothing, TruncationDelta, EnergyDropSink, XRunDelta}

# sink selection: Dict-backed PauliSum has no term order, so it keeps the
# TruncationDelta dict route (with its one kept-terms sweep); the SPV
# kernels get the single-pass run sinks.
_make_sink(corr::Union{EnergyCorrection,EnergyVarianceCorrection}, ::PauliSum) =
    TruncationDelta(corr.ψ)
_make_sink(corr::EnergyCorrection, ::SparsePauliVector{N,W}) where {N,W} =
    EnergyDropSink{W}((corr.ψ.v % UInt128) % W, zero(ComplexF64))
_make_sink(corr::EnergyVarianceCorrection, ::SparsePauliVector{N,W}) where {N,W} =
    XRunDelta((corr.ψ.v % UInt128) % W)

# amplitudes of the KEPT operator on b's support (+ the reference ket)
function _sweep_kept!(adict::Dict{Int128,ComplexF64}, aψ::Base.RefValue{ComplexF64},
                      v::SparsePauliVector{N,W,T}, ψv::Int128) where {N,W,T}
    @inbounds for i in 1:v.n
        ph, kv2 = _ket_action(v.z[i] % Int128, v.x[i] % Int128, ψv)
        hit = haskey(adict, kv2)
        (hit || kv2 == ψv) || continue
        amp = ph * v.c[i]
        kv2 == ψv && (aψ[] += amp)
        hit && (adict[kv2] += amp)
    end
    return nothing
end

function _sweep_kept!(adict::Dict{Int128,ComplexF64}, aψ::Base.RefValue{ComplexF64},
                      O::PauliSum{N}, ψv::Int128) where {N}
    for (p, c) in O
        ph, kv2 = _ket_action(p.z, p.x, ψv)
        hit = haskey(adict, kv2)
        (hit || kv2 == ψv) || continue
        amp = ph * c
        kv2 == ψv && (aψ[] += amp)
        hit && (adict[kv2] += amp)
    end
    return nothing
end

_finalize_delta!(::NoCorrection, Δ::TruncationDelta, O) = nothing

function _finalize_delta!(corr::EnergyCorrection, Δ::EnergyDropSink, O)
    corr.accumulated_energy += -real(Δ.eB)
    return nothing
end

function _finalize_delta!(corr::EnergyVarianceCorrection, Δ::XRunDelta, O)
    _flush_run!(Δ)          # close the trailing run
    eA = Δ.eA
    eB = Δ.eB
    # identical formula to the TruncationDelta route below — only the
    # gathering of bb/ab/eA/eB differs (single ordered pass vs dict + sweep)
    corr.accumulated_energy   += -real(eB)
    corr.accumulated_variance += -(Δ.bb + 2 * real(Δ.ab)) + real(2 * eA * eB + eB * eB)
    return nothing
end

function _finalize_delta!(corr::EnergyCorrection, Δ::TruncationDelta, O)
    corr.accumulated_energy += -real(get(Δ.b, Δ.ψv, zero(ComplexF64)))
    return nothing
end

function _finalize_delta!(corr::EnergyVarianceCorrection, Δ::TruncationDelta, O)
    isempty(Δ.b) && return nothing
    eB = get(Δ.b, Δ.ψv, zero(ComplexF64))       # ⟨ψ|B|ψ⟩ (complex-safe)
    bb = 0.0                                    # ⟨Bψ|Bψ⟩
    for amp in values(Δ.b)
        bb += abs2(amp)
    end
    adict = Dict{Int128,ComplexF64}()
    for k in keys(Δ.b)
        adict[k] = zero(ComplexF64)
    end
    aψ = Ref(zero(ComplexF64))
    _sweep_kept!(adict, aψ, O, Δ.ψv)
    ab = zero(ComplexF64)                       # ⟨Aψ|Bψ⟩ (only b's support)
    for (k, bv) in Δ.b
        ab += conj(adict[k]) * bv
    end
    eA = aψ[]                                   # ⟨ψ|A|ψ⟩
    # matches variance() = real(‖Oψ‖² − ⟨ψ|O|ψ⟩²) exactly:
    #   Δ‖Oψ‖² = −(bb + 2Re⟨a|b⟩),  Δ⟨O⟩² = −(2·eA·eB + eB²)
    corr.accumulated_energy   += -real(eB)
    corr.accumulated_variance += -(bb + 2 * real(ab)) + real(2 * eA * eB + eB * eB)
    return nothing
end


# ============================================================
# measure — snapshot quantities before/after truncation
# ============================================================

_measure(::AnyPauliSum, ::NoCorrection) = nothing

function _measure(O::AnyPauliSum{N}, corr::EnergyCorrection{N}) where N
    return (energy = real(expectation_value(O, corr.ψ)),)
end

function _measure(O::AnyPauliSum{N}, corr::EnergyVarianceCorrection{N}) where N
    return (energy = real(expectation_value(O, corr.ψ)),
            variance = real(variance(O, corr.ψ)))
end


# ============================================================
# _accumulate! — update accumulator with before/after diffs
# ============================================================

_accumulate!(::NoCorrection, before, after) = nothing

function _accumulate!(corr::EnergyCorrection, before, after)
    corr.accumulated_energy += after.energy - before.energy
end

function _accumulate!(corr::EnergyVarianceCorrection, before, after)
    corr.accumulated_energy += after.energy - before.energy
    corr.accumulated_variance += after.variance - before.variance
end


# ============================================================
# truncate! — unified entry point
# ============================================================

"""
    truncate!(O::PauliSum, strategy::TruncationStrategy,
              corr::CorrectionAccumulator=NoCorrection())

Apply `strategy` to truncate `O` in-place. If a `CorrectionAccumulator` is
provided, measure quantities before and after truncation and accumulate the
differences.

Users can define new strategies by subtyping `TruncationStrategy` and
implementing `_apply!(O, s)`. New correction types are defined by subtyping
`CorrectionAccumulator` and implementing `_measure(O, corr)` and
`_accumulate!(corr, before, after)`.
"""
function truncate!(O::AnyPauliSum, strategy::TruncationStrategy,
                   corr::CorrectionAccumulator=NoCorrection())
    # Fast path: filter-compilable strategies (pure-drop by construction)
    # with the built-in accumulators use the single-pass delta formula
    # instead of full before/after measurements. Everything else -- rescaling
    # strategies (survivors change, e.g. StochasticSamplingTruncation),
    # data-dependent thresholds (AdaptiveTruncation), user-defined strategies
    # or accumulators -- takes the measured fallback below, which assumes
    # nothing about what `_apply!` does.
    if corr isa Union{EnergyCorrection,EnergyVarianceCorrection} &&
       !(strategy isa NoTruncation) && _is_compilable(strategy)
        Δ = _make_sink(corr, O)
        _apply_dropping!(O, _compile_filter(strategy), Δ)
        _finalize_delta!(corr, Δ, O)
        return O
    end
    before = _measure(O, corr)
    _apply!(O, strategy)
    after = _measure(O, corr)
    _accumulate!(corr, before, after)
    return O
end

# filter with drop-sink observation (pure-drop strategies only)
function _apply_dropping!(O::PauliSum{N}, f, Δ::TruncationDelta) where {N}
    filter!(O) do pr
        p, c = pr
        if should_drop(f, p.z % UInt128, p.x % UInt128, abs(c))
            _sink_drop!(Δ, p.z, p.x, c)
            return false
        end
        return true
    end
    return O
end
_apply_dropping!(v::SparsePauliVector, f, Δ::SPVSink) = _compact_spv!(v, f, Δ)
