"""
    LinearAlgebra.norm(ps::PauliSum{N,T}, p::Real=2) where {N,T}

Compute the p-norm of the coefficient vector of a `PauliSum`.
- `p=2` (default): Frobenius norm `sqrt(sum(|c_i|^2))`
- `p=1`: L1 norm `sum(|c_i|)`
- `p=Inf`: max norm `max(|c_i|)`
"""
function LinearAlgebra.norm(ps::PauliSum{N,T}, p::Real=2) where {N,T}
    if p == 2
        s = zero(real(T))
        for (_, c) in ps
            s += abs2(c)
        end
        return sqrt(s)
    elseif p == 1
        s = zero(real(T))
        for (_, c) in ps
            s += abs(c)
        end
        return s
    elseif p == Inf
        s = zero(real(T))
        for (_, c) in ps
            a = abs(c)
            if a > s
                s = a
            end
        end
        return s
    else
        s = zero(real(T))
        for (_, c) in ps
            s += abs(c)^p
        end
        return s^(1/p)
    end
end

"""
    LinearAlgebra.norm(ks::KetSum{N,T}, p::Real=2) where {N,T}

Compute the p-norm of the coefficient vector of a `KetSum`.
"""
function LinearAlgebra.norm(ks::KetSum{N,T}, p::Real=2) where {N,T}
    if p == 2
        s = zero(real(T))
        for (_, c) in ks
            s += abs2(c)
        end
        return sqrt(s)
    elseif p == 1
        s = zero(real(T))
        for (_, c) in ks
            s += abs(c)
        end
        return s
    elseif p == Inf
        s = zero(real(T))
        for (_, c) in ks
            a = abs(c)
            if a > s
                s = a
            end
        end
        return s
    else
        s = zero(real(T))
        for (_, c) in ks
            s += abs(c)^p
        end
        return s^(1/p)
    end
end

"""
    Base.isapprox(ps1::PauliSum{N}, ps2::PauliSum{N}; atol=1e-14, rtol=0) where {N}

Compare two `PauliSum`s for approximate equality based on the norm of their difference.
"""
function Base.isapprox(ps1::PauliSum{N}, ps2::PauliSum{N}; atol=1e-14, rtol=0) where {N}
    diff = ps1 - ps2
    clip!(diff; thresh=0.0)
    return norm(diff) <= atol + rtol * max(norm(ps1), norm(ps2))
end
