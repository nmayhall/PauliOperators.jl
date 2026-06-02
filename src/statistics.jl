"""
    variance(O::PauliSum{N}, ψ::Ket{N}) where N

Compute the variance of observable `O` in state `ψ`: `<O²> - <O>²`.
"""
function variance(O::PauliSum{N}, ψ::Ket{N}) where N
    σ = KetSum(N, ComplexF64)
    for (p, ci) in O
        cj, ki = p * ψ
        curr = get(σ, ki, 0.0) + cj * ci
        σ[ki] = curr
    end

    e2 = 0.0
    for (_, v) in σ
        e2 += v' * v
    end

    e1 = get(σ, ψ, 0.0)

    return real(e2 - e1 * e1)
end

"""
    covariance(A::PauliSum{N}, B::PauliSum{N}, ψ::Ket{N}) where N

Compute the covariance of observables `A` and `B` in state `ψ`: `<A†B> - <A†><B>`.
"""
function covariance(A::PauliSum{N}, B::PauliSum{N}, ψ::Ket{N}) where N
    σA = KetSum(N, ComplexF64)
    for (p, ci) in A
        cj, ki = p' * ψ
        curr = get(σA, ki, 0.0) + cj * ci'
        σA[ki] = curr
    end

    σB = KetSum(N, ComplexF64)
    for (p, ci) in B
        cj, ki = p * ψ
        curr = get(σB, ki, 0.0) + cj * ci
        σB[ki] = curr
    end

    eA = get(σA, ψ, 0.0)'
    eB = get(σB, ψ, 0.0)
    return inner_product(σA, σB) - eA * eB
end
