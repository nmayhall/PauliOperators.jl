"""
    evolve(O::PauliSum{N, W, T}, G::PauliBasis{N}, θ::Real)

Heisenberg-picture evolution: O(θ) = exp(iθ/2 G) O exp(-iθ/2 G)

Commuting terms pass through unchanged. Non-commuting terms branch:
    O(θ) = cos(θ)·O - i·sin(θ)·G·O
"""
function evolve(O::PauliSum{N, W, T}, G::PauliBasis{N}, θ::Real) where {N,W,T}
    _cos = cos(θ)
    _sin = 1im*sin(θ)
    cos_branch = deepcopy(O)
    sin_branch = PauliSum(N)
    for (p,c) in O
        if commute(p,G) == false
            cos_branch[p] *= _cos
            tmp = c*_sin*G*p
            curr = get(sin_branch, PauliBasis(tmp), 0.0) + coeff(tmp)
            sin_branch[PauliBasis(tmp)] = curr
        end
    end
    sum!(cos_branch, sin_branch)
    return cos_branch
end


"""
    evolve!(O::PauliSum{N, W, T}, G::PauliBasis{N}, θ::Real)

In-place Heisenberg-picture evolution: O(θ) = exp(iθ/2 G) O exp(-iθ/2 G)
"""
function evolve!(O::PauliSum{N, W, T}, G::PauliBasis{N}, θ::Real) where {N,W,T}
    _cos = cos(θ)
    _sin = 1im*sin(θ)
    sin_branch = PauliSum(N)
    for (p,c) in O
        if commute(p,G) == false
            tmp = c*_sin*G*p
            curr = get(sin_branch, PauliBasis(tmp), 0.0) + coeff(tmp)
            sin_branch[PauliBasis(tmp)] = curr
            O[p] *= _cos
        end
    end
    sum!(O, sin_branch)
    return O
end

"""
    evolve(K::KetSum{N, W, T}, G::PauliBasis{N}, θ::Real)

Schrödinger-picture evolution: K(θ) = exp(-iθ/2 G) K

Applies the unitary exp(-iθ/2 G) to a KetSum state vector.
"""
function evolve(K::KetSum{N, W, T}, G::PauliBasis{N}, θ::Real) where {N,W,T}
    _cos = cos(θ/2)
    _sin = -1im*sin(θ/2)
    K2 = KetSum(N, T=ComplexF64)
    GK = KetSum(N, T=ComplexF64)
    for (k, c) in K
        K2[k] = c * _cos
        ci, ki = G * k
        tmp = get(GK, ki, 0)
        GK[ki] = tmp + _sin * c * ci
    end
    for (k, c) in GK
        tmp = get(K2, k, 0)
        K2[k] = c + tmp
    end
    return K2
end

"""
    evolve!(K::KetSum{N, W, ComplexF64}, G::PauliBasis{N}, θ::Real)

In-place Schrödinger-picture evolution: K → exp(-iθ/2 G) K

Modifies `K` in place. Element type must be `ComplexF64` so that the
imaginary contribution from the sine branch can be stored back in `K`.
"""
function evolve!(K::KetSum{N, W, ComplexF64}, G::PauliBasis{N}, θ::Real) where {N,W}
    _cos = cos(θ/2)
    _sin = -1im*sin(θ/2)
    GK = KetSum(N, T=ComplexF64)
    for (k, c) in K
        K[k] *= _cos
        ci, ki = G * k
        tmp = get(GK, ki, 0)
        GK[ki] = tmp + _sin * c * ci
    end
    for (k, c) in GK
        tmp = get(K, k, 0)
        K[k] = c + tmp
    end
    return K
end

"""
    evolve(O::PauliSum{N,W,T}, generators::Vector{PauliBasis{N}}, angles::Vector{<:Real};
           truncation::TruncationStrategy=NoTruncation(),
           correction::CorrectionAccumulator=NoCorrection())

Heisenberg-picture sequence evolution: applies generators left to right, producing

    U_n† ⋯ U_1† O U_1 ⋯ U_n

where Uₖ = exp(-iθₖ/2 Gₖ). The effective right-side unitary is U₁U₂⋯Uₙ.

Sequences from `trotterize` and `qdrift` are designed for this convention.
"""
function evolve(O::PauliSum{N,W,T}, generators::Vector{PauliBasis{N,W}}, angles::Vector{<:Real};
                truncation::TruncationStrategy=NoTruncation(),
                correction::CorrectionAccumulator=NoCorrection()) where {N,W,T}
    length(generators) == length(angles) || throw(DimensionMismatch("generators and angles must have same length"))
    Ot = deepcopy(O)
    for (gi, θi) in zip(generators, angles)
        evolve!(Ot, gi, θi)
        truncate!(Ot, truncation, correction)
    end
    return Ot
end

"""
    evolve(K::KetSum{N,W,T}, generators::Vector{PauliBasis{N}}, angles::Vector{<:Real})

Schrödinger-picture sequence evolution: applies generators left to right, producing

    Uₙ ⋯ U₂ U₁ |K⟩

where Uₖ = exp(-iθₖ/2 Gₖ). The effective unitary is Uₙ⋯U₁ (reversed order).

Note: sequences from `trotterize` and `qdrift` use the Heisenberg convention.
To use them with KetSum, reverse the sequence:

    evolve(K, reverse(generators), reverse(angles))
"""
function evolve(K::KetSum{N,W,T}, generators::Vector{PauliBasis{N,W}}, angles::Vector{<:Real}) where {N,W,T}
    length(generators) == length(angles) || throw(DimensionMismatch("generators and angles must have same length"))
    Kt = deepcopy(K)
    for (gi, θi) in zip(generators, angles)
        Kt = evolve(Kt, gi, θi)
    end
    return Kt
end
