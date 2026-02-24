"""
    evolve(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real)

Heisenberg-picture evolution: O(θ) = exp(iθ/2 G) O exp(-iθ/2 G)

Commuting terms pass through unchanged. Non-commuting terms branch:
    O(θ) = cos(θ)·O - i·sin(θ)·G·O
"""
function evolve(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real) where {N,T}
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
    evolve!(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real)

In-place Heisenberg-picture evolution: O(θ) = exp(iθ/2 G) O exp(-iθ/2 G)
"""
function evolve!(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real) where {N,T}
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
