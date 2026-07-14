"""
    AbstractQuantumChannel

Forward-compatibility supertype for quantum channels acting on `PauliSum`s.
The current implementation provides only top-level functions for single-qubit
Pauli channels (depolarizing, dephasing/phase-flip, bit-flip, bit-phase-flip,
plus a general `pauli_channel!`). Future channel types (generic Kraus,
amplitude damping, Lindblad, …) will subtype this and dispatch on a uniform
`apply_channel!` method.
"""
abstract type AbstractQuantumChannel end


@inline _qubit_mask(::Type{W}, N::Int, ::Nothing) where {W<:Unsigned} =
    N == 0 ? zero(W) : _nbit_mask(W, N)

@inline function _qubit_mask(::Type{W}, N::Int, q::Integer) where {W<:Unsigned}
    1 ≤ q ≤ N || throw(ArgumentError("qubit index $q out of range 1:$N"))
    return one(W) << (q - 1)
end

function _qubit_mask(::Type{W}, N::Int, qs) where {W<:Unsigned}
    m = zero(W)
    for q in qs
        1 ≤ q ≤ N || throw(ArgumentError("qubit index $q out of range 1:$N"))
        m |= one(W) << (q - 1)
    end
    return m
end

@inline function _check_inplace_eltype(::Type{T}) where {T}
    if T <: Integer
        throw(ArgumentError(
            "In-place channel application requires a non-Integer coefficient " *
            "type (got $T). Use the allocating variant or convert the PauliSum first."))
    end
    return nothing
end


"""
    pauli_channel!(O::PauliSum{N}, pX, pY, pZ; qubits=nothing) -> O

Apply (in place) the Heisenberg-picture single-qubit Pauli channel
`E(ρ) = pI·ρ + pX·XρX + pY·YρY + pZ·ZρZ` independently to each qubit in
`qubits` (default: all `1:N`), where `pI = 1 - pX - pY - pZ`.

The channel is diagonal in the Pauli basis: each `PauliBasis` term `Q` is
scaled by `∏_{i ∈ qubits} λ_{Q_i}`, with

    λ_I = 1
    λ_X = 1 - 2(pY + pZ)
    λ_Y = 1 - 2(pX + pZ)
    λ_Z = 1 - 2(pX + pY)

`qubits` accepts `nothing`, an `Integer`, or any iterable of integer qubit
indices (e.g. `Vector`, `Tuple`, `UnitRange`).

Throws `ArgumentError` if any of `pX,pY,pZ` is negative or if their sum
exceeds 1.
"""
function pauli_channel!(O::PauliSum{N,W,T}, pX::Real, pY::Real, pZ::Real;
                        qubits=nothing) where {N,W,T}
    _check_inplace_eltype(T)
    (pX ≥ 0 && pY ≥ 0 && pZ ≥ 0) || throw(ArgumentError("probabilities must be non-negative (got pX=$pX, pY=$pY, pZ=$pZ)"))
    pX + pY + pZ ≤ 1 + 4*eps(Float64) || throw(ArgumentError("pX+pY+pZ must be ≤ 1 (got $(pX+pY+pZ))"))

    M = _qubit_mask(W, N, qubits)
    λX = 1 - 2*(pY + pZ)
    λY = 1 - 2*(pX + pZ)
    λZ = 1 - 2*(pX + pY)

    for (P, c) in O
        xM = P.x & M
        zM = P.z & M
        nX = count_ones(xM & ~P.z)
        nY = count_ones(xM &  P.z)
        nZ = count_ones(zM & ~P.x)
        O[P] = c * λX^nX * λY^nY * λZ^nZ
    end
    return O
end

pauli_channel(O::AnyPauliSum, pX::Real, pY::Real, pZ::Real; kwargs...) =
    pauli_channel!(deepcopy(O), pX, pY, pZ; kwargs...)


"""
    depolarizing_channel!(O::PauliSum{N}, p; qubits=nothing) -> O

Apply (in place) the i.i.d. single-qubit depolarizing channel with
parameter `p ∈ [0,1]`, in the Heisenberg picture. Convention:
`pI = 1-p`, `pX = pY = pZ = p/3`. Equivalent to scaling each Pauli `P`
by `(1 - 4p/3)^w`, where `w` is the number of non-identity qubits of `P`
within `qubits`.

# Weight-decay equivalence

Applied to all qubits, this realizes the CPTP map
`O ↦ Σ_P exp(-γΔt·w(P)) ⟨P,O⟩ P / d` for any `γ,Δt ≥ 0` by choosing
`p = (3/4)·(1 - exp(-γΔt))` (see `depolarizing_p_for_weight_decay`). A
*thresholded* weight damper of the form "only damp when w > lmax" is
**not CPTP in general** and is intentionally not provided here.
"""
function depolarizing_channel!(O::PauliSum{N,W,T}, p::Real; qubits=nothing) where {N,W,T}
    _check_inplace_eltype(T)
    0 ≤ p ≤ 1 || throw(ArgumentError("depolarizing parameter p=$p must be in [0,1]"))
    M = _qubit_mask(W, N, qubits)
    λ = 1 - 4p/3
    for (P, c) in O
        k = count_ones((P.z | P.x) & M)
        O[P] = c * λ^k
    end
    return O
end

depolarizing_channel(O::AnyPauliSum, p::Real; kwargs...) =
    depolarizing_channel!(deepcopy(O), p; kwargs...)


"""
    dephasing_channel!(O::PauliSum{N}, p; qubits=nothing) -> O

Apply (in place) the i.i.d. single-qubit dephasing (phase-flip) channel
with probability `p ∈ [0,1]`: `ρ ↦ (1-p)ρ + p·ZρZ`. Suppresses `X` and
`Y` Pauli letters by `(1 - 2p)` per qubit.

Aliased as `phase_flip_channel!`.
"""
function dephasing_channel!(O::PauliSum{N,W,T}, p::Real; qubits=nothing) where {N,W,T}
    _check_inplace_eltype(T)
    0 ≤ p ≤ 1 || throw(ArgumentError("dephasing parameter p=$p must be in [0,1]"))
    M = _qubit_mask(W, N, qubits)
    λ = 1 - 2p
    for (P, c) in O
        k = count_ones(P.x & M)
        O[P] = c * λ^k
    end
    return O
end

dephasing_channel(O::AnyPauliSum, p::Real; kwargs...) =
    dephasing_channel!(deepcopy(O), p; kwargs...)

const phase_flip_channel!  = dephasing_channel!
const phase_flip_channel   = dephasing_channel


"""
    bit_flip_channel!(O::PauliSum{N}, p; qubits=nothing) -> O

Apply (in place) the i.i.d. single-qubit bit-flip channel with probability
`p ∈ [0,1]`: `ρ ↦ (1-p)ρ + p·XρX`. Suppresses `Y` and `Z` Pauli letters
by `(1 - 2p)` per qubit.
"""
function bit_flip_channel!(O::PauliSum{N,W,T}, p::Real; qubits=nothing) where {N,W,T}
    _check_inplace_eltype(T)
    0 ≤ p ≤ 1 || throw(ArgumentError("bit_flip parameter p=$p must be in [0,1]"))
    M = _qubit_mask(W, N, qubits)
    λ = 1 - 2p
    for (P, c) in O
        k = count_ones(P.z & M)
        O[P] = c * λ^k
    end
    return O
end

bit_flip_channel(O::AnyPauliSum, p::Real; kwargs...) =
    bit_flip_channel!(deepcopy(O), p; kwargs...)


"""
    bit_phase_flip_channel!(O::PauliSum{N}, p; qubits=nothing) -> O

Apply (in place) the i.i.d. single-qubit bit-phase-flip channel with
probability `p ∈ [0,1]`: `ρ ↦ (1-p)ρ + p·YρY`. Suppresses `X` and `Z`
Pauli letters by `(1 - 2p)` per qubit.
"""
function bit_phase_flip_channel!(O::PauliSum{N,W,T}, p::Real; qubits=nothing) where {N,W,T}
    _check_inplace_eltype(T)
    0 ≤ p ≤ 1 || throw(ArgumentError("bit_phase_flip parameter p=$p must be in [0,1]"))
    M = _qubit_mask(W, N, qubits)
    λ = 1 - 2p
    for (P, c) in O
        k = count_ones((P.x ⊻ P.z) & M)
        O[P] = c * λ^k
    end
    return O
end

bit_phase_flip_channel(O::AnyPauliSum, p::Real; kwargs...) =
    bit_phase_flip_channel!(deepcopy(O), p; kwargs...)


# ------------------------------------------------------------
# SparsePauliVector fast paths: in-place coefficient scaling on packed
# words (order-preserving, allocation-free). Same semantics as the
# PauliSum methods above.
# ------------------------------------------------------------

function pauli_channel!(O::SparsePauliVector{N,W,T}, pX::Real, pY::Real, pZ::Real;
                        qubits=nothing) where {N,W,T}
    _check_inplace_eltype(T)
    (pX ≥ 0 && pY ≥ 0 && pZ ≥ 0) || throw(ArgumentError("probabilities must be non-negative (got pX=$pX, pY=$pY, pZ=$pZ)"))
    pX + pY + pZ ≤ 1 + 4*eps(Float64) || throw(ArgumentError("pX+pY+pZ must be ≤ 1 (got $(pX+pY+pZ))"))
    M = _qubit_mask(W, N, qubits)
    λX = 1 - 2*(pY + pZ)
    λY = 1 - 2*(pX + pZ)
    λZ = 1 - 2*(pX + pY)
    @inbounds for i in 1:O.n
        xM = O.x[i] & M
        zM = O.z[i] & M
        nX = count_ones(xM & ~O.z[i])
        nY = count_ones(xM &  O.z[i])
        nZ = count_ones(zM & ~O.x[i])
        O.c[i] *= λX^nX * λY^nY * λZ^nZ
    end
    return O
end

function depolarizing_channel!(O::SparsePauliVector{N,W,T}, p::Real; qubits=nothing) where {N,W,T}
    _check_inplace_eltype(T)
    0 ≤ p ≤ 1 || throw(ArgumentError("depolarizing parameter p=$p must be in [0,1]"))
    M = _qubit_mask(W, N, qubits)
    λ = 1 - 4p/3
    @inbounds for i in 1:O.n
        O.c[i] *= λ^count_ones((O.z[i] | O.x[i]) & M)
    end
    return O
end

function dephasing_channel!(O::SparsePauliVector{N,W,T}, p::Real; qubits=nothing) where {N,W,T}
    _check_inplace_eltype(T)
    0 ≤ p ≤ 1 || throw(ArgumentError("dephasing parameter p=$p must be in [0,1]"))
    M = _qubit_mask(W, N, qubits)
    λ = 1 - 2p
    @inbounds for i in 1:O.n
        O.c[i] *= λ^count_ones(O.x[i] & M)
    end
    return O
end

function bit_flip_channel!(O::SparsePauliVector{N,W,T}, p::Real; qubits=nothing) where {N,W,T}
    _check_inplace_eltype(T)
    0 ≤ p ≤ 1 || throw(ArgumentError("bit_flip parameter p=$p must be in [0,1]"))
    M = _qubit_mask(W, N, qubits)
    λ = 1 - 2p
    @inbounds for i in 1:O.n
        O.c[i] *= λ^count_ones(O.z[i] & M)
    end
    return O
end

function bit_phase_flip_channel!(O::SparsePauliVector{N,W,T}, p::Real; qubits=nothing) where {N,W,T}
    _check_inplace_eltype(T)
    0 ≤ p ≤ 1 || throw(ArgumentError("bit_phase_flip parameter p=$p must be in [0,1]"))
    M = _qubit_mask(W, N, qubits)
    λ = 1 - 2p
    @inbounds for i in 1:O.n
        O.c[i] *= λ^count_ones((O.x[i] ⊻ O.z[i]) & M)
    end
    return O
end


"""
    depolarizing_p_for_weight_decay(rate, dt) -> Float64

Return the depolarizing parameter `p` such that the i.i.d. depolarizing
channel `depolarizing_channel!(O, p)` (applied to all qubits) scales each
Pauli term `P` by `exp(-rate*dt · w(P))`, where `w(P)` is the number of
non-identity qubits of `P`.

This is the CPTP realization of exponential-in-weight damping. A
*thresholded* weight damper (damp only when `w(P) > lmax`) is not CPTP in
general — its implied multi-qubit Pauli probabilities have negative
entries — and is intentionally not provided in this module.
"""
@inline depolarizing_p_for_weight_decay(rate::Real, dt::Real) =
    (3/4) * (1 - exp(-rate*dt))
