"""
    weight(p::PauliBasis)

Number of non-identity single-qubit Pauli factors.
"""
function weight(p::PauliBasis)
    return count_ones(p.x | p.z)
end

"""
    coeff_clip!(ps::PauliSum{N}; thresh=1e-16)

Remove Pauli terms with |coefficient| <= `thresh`.
"""
function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

"""
    clip!(ps::PauliSum{N}; thresh=1e-16)

!!! warning "Deprecated"
    Use `coeff_clip!` instead for consistency with `weight_clip!`, `majorana_weight_clip!`, etc.
"""
clip!(ps::PauliSum; thresh=1e-16) = coeff_clip!(ps; thresh=thresh)

"""
    weight_clip!(ps::PauliSum{N}, max_weight::Int)

Remove terms with Pauli weight above max_weight.
"""
function weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}
    return filter!(p->weight(p.first) <= max_weight, ps)
end

"""
    majorana_weight(p::Union{PauliBasis{N}, Pauli{N}}) where N

Compute the Majorana weight of a Pauli string. The Majorana weight counts the number
of Majorana operators needed to represent the Pauli string in the Jordan-Wigner encoding.
"""
function majorana_weight(p::Union{PauliBasis{N}, Pauli{N}}) where N
    w = 0
    control = true
    Ibits = ~(p.z | p.x)
    Zbits = p.z & ~p.x

    for i in reverse(1:N)
        xbit = (p.x >> (i - 1)) & 1 != 0
        Zbit = (Zbits >> (i - 1)) & 1 != 0
        Ibit = (Ibits >> (i - 1)) & 1 != 0
        if Zbit && control || Ibit && !control
            w += 2
        elseif xbit
            control = !control
            w += 1
        end
    end
    return w
end

"""
    majorana_weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}

Remove terms with Majorana weight above `max_weight`.
"""
function majorana_weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}
    return filter!(p->majorana_weight(p.first) <= max_weight, ps)
end

"""
    coeff_clip!(ks::KetSum{N}; thresh=1e-16) where {N}

Remove Ket terms with |coefficient| <= `thresh`.
"""
function coeff_clip!(ks::KetSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ks)
end

"""
    clip!(ks::KetSum{N}; thresh=1e-16) where {N}

!!! warning "Deprecated"
    Use `coeff_clip!` instead.
"""
clip!(ks::KetSum; thresh=1e-16) = coeff_clip!(ks; thresh=thresh)

"""
    offdiag(ps::PauliSum{N,T}) where {N,T}

Return a new `PauliSum` containing only the off-diagonal terms (those with `x != 0`).
"""
function offdiag(ps::PauliSum{N,T}) where {N,T}
    return filter(p->p.first.x != 0, ps)
end

"""
    LinearAlgebra.diag(ps::PauliSum{N,T}) where {N,T}

Return a new `PauliSum` containing only the diagonal terms (those with `x == 0`,
i.e., only Z and I factors).
"""
function LinearAlgebra.diag(ps::PauliSum{N,T}) where {N,T}
    return filter(p->p.first.x == 0, ps)
end
