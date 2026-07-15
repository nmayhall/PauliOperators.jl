Base.:*(p::Pauli, pb::PauliBasis) = p*Pauli(pb)
Base.:*(pb::PauliBasis, p::Pauli) = Pauli(pb)*p

"""
    Base.:*(d1::Union{Dyad{N}, DyadBasis{N}}, d2::Union{Dyad{N}, DyadBasis{N}})

Dyad product: `|i⟩⟨j| ⋅ |k⟩⟨l| = δ_jk |i⟩⟨l|`, with coefficients multiplied through.
"""
function Base.:*(d1::Union{Dyad{N}, DyadBasis{N}}, d2::Union{Dyad{N}, DyadBasis{N}}) where N
    return Dyad{N}(coeff(d1) * coeff(d2) * (d1.bra.v==d2.ket.v), d1.ket, d2.bra)
end

Base.:*(k::Ket{N}, b::Bra{N}) where N = DyadBasis{N}(k,b)
Base.:*(k::Bra{N}, b::Ket{N}) where N = k.v == b.v ? 1 : 0 

"""
    Base.:*(b::Bra{N}, p::Union{Pauli{N}, PauliBasis{N}})

Apply a Pauli to a bra from the right: `⟨b|P = c ⟨b'|` with `b' = b ⊻ x`.
Returns a `(coefficient, Bra)` tuple — a Pauli connects each basis bra to
exactly one basis bra, so no sum type is needed.
"""
function Base.:*(b::Bra{N}, p::Pauli{N}) where N
    new_bra = Bra{N}(b.v ⊻ p.x)
    sign = count_ones(p.z & b.v)%2
    return PHASE_TBL[(2*sign)%4 + 1]*p.s, new_bra
end

function Base.:*(b::Bra{N}, p::PauliBasis{N}) where N
    new_bra = Bra{N}(b.v ⊻ p.x)
    sign = count_ones(b.v & p.z)%2
    return PHASE_TBL[(symplectic_phase(p) + 2*sign)%4 + 1], new_bra
end

"""
    Base.:*(p::Union{Pauli{N}, PauliBasis{N}}, k::Ket{N})

Apply a Pauli to a computational-basis ket: `P|k⟩ = c |k'⟩` with `k' = k ⊻ x`.
Returns a `(coefficient, Ket)` tuple — a Pauli maps each basis state to
exactly one basis state, so no sum type is needed.
"""
function Base.:*(p::Pauli{N}, k::Ket{N}) where N
    new_ket = Ket{N}(p.x ⊻ k.v)
    sign = count_ones(p.z & new_ket.v)%2
    return PHASE_TBL[(2*sign)%4 + 1]*p.s, new_ket
end

function Base.:*(p::PauliBasis{N}, k::Ket{N}) where N
    new_ket = Ket{N}(p.x ⊻ k.v)
    sign = count_ones(p.z & new_ket.v)%2
    return PHASE_TBL[(symplectic_phase(p) + 2*sign)%4 + 1], new_ket
end

function Base.:*(p::Union{Pauli{N}, PauliBasis{N}}, d::Union{Dyad{N}, DyadBasis{N}}) where N
    new_coeff, new_ket = p*d.ket
    return Dyad{N}(new_coeff * coeff(d) , new_ket, d.bra)
end 
function Base.:*(d::Union{Dyad{N}, DyadBasis{N}}, p::Union{Pauli{N}, PauliBasis{N}}) where N
    new_coeff, new_bra = d.bra*p
    return Dyad{N}(new_coeff * coeff(d) , d.ket, new_bra)
end 

function Base.:*(p::Union{Pauli{N}, PauliBasis{N}}, ks::KetSum{N,W,T}) where {N,W,T}
    out = KetSum(N, T=T)
    for (k,c) in ks
        c2,k2 = p*k
        tmp = get(out, k2, 0.0)
        out[k2] = tmp + c2*c
    end
    return out 
end



"""
    Base.:*(O::PauliSum{N,W,T}, k::Ket{N})

Apply a sum of Paulis to a basis state, returning a `KetSum` with one entry
per distinct `x` bitstring in `O`.
"""
function Base.:*(O::PauliSum{N,W,T}, k::Ket{N}) where {N,W,T}
    out = KetSum(N)
    for (p,c) in O
        c2,k2 = p*k
        tmp = get(out, k2, 0.0)
        out[k2] = tmp + c2*c
    end
    return out 
end

"""
    Base.:*(O::AnyPauliSum{N}, ks::KetSum{N})

Apply a sum of Paulis to a linear combination of basis states:
`(Σᵢ cᵢ Pᵢ)(Σₖ vₖ |k⟩)`. Returns a `ComplexF64` `KetSum`.
"""
function Base.:*(O::AnyPauliSum{N}, ks::KetSum{N}) where {N}
    out = KetSum(N, T=ComplexF64)
    for (p,c) in O
        for (k,ck) in ks
            c2,k2 = p*k
            tmp = get(out, k2, zero(ComplexF64))
            out[k2] = tmp + c2*c*ck
        end
    end
    return out
end

function Base.:*(d::DyadSum{N,W,T}, p::PauliSum{N,W,T}) where {N,W,T}
    out = DyadSum(N,T)
    for (dyad, coeff_d) in d
        for (pauli, coeff_p) in p
            new_dyad = dyad*pauli
            sum!(out, new_dyad * coeff_d * coeff_p)
        end   
    end
    return out 
end 

function Base.:*(d::DyadSum{N,W,T}, p::Adjoint{<:Any, PauliSum{N,W,T}}) where {N,W,T}
    out = DyadSum(N,T)
    for (dyad, coeff_d) in d
        for (pauli, coeff_p) in p.parent
            new_dyad = dyad*pauli
            sum!(out, new_dyad * coeff_d * coeff_p')
        end   
    end
    return out 
end 
function Base.:*(d::Adjoint{<:Any, DyadSum{N,W,T}}, p::PauliSum{N,W,T}) where {N,W,T}
    out = DyadSum(N,T)
    for (dyad, coeff_d) in d.parent
        for (pauli, coeff_p) in p
            new_dyad = dyad'*pauli
            sum!(out, new_dyad * coeff_d' * coeff_p)
        end   
    end
    return out 
end 

function Base.:*(d::Adjoint{<:Any, DyadSum{N,W,T}}, p::Adjoint{<:Any, PauliSum{N, W, T}}) where {N,W,T}
    out = DyadSum(N,T)
    for (dyad, coeff_d) in d.parent
        for (pauli, coeff_p) in p.parent
            new_dyad = dyad'*pauli
            sum!(out, new_dyad * coeff_d' * coeff_p')
        end   
    end
    return out 
end 



function Base.:*(p::PauliSum{N,W,T}, d::DyadSum{N,W,T}) where {N,W,T}
    out = DyadSum(N,T)
    for (dyad, coeff_d) in d
        for (pauli, coeff_p) in p
            new_dyad = pauli*dyad
            sum!(out, new_dyad * coeff_d * coeff_p)
        end   
    end
    return out 
end 

function Base.:*(p::PauliSum{N,W,T}, d::Adjoint{<:Any, DyadSum{N,W,T}}) where {N,W,T}
    out = DyadSum(N,T)
    for (dyad, coeff_d) in d.parent
        for (pauli, coeff_p) in p
            new_dyad = pauli*dyad'
            sum!(out, new_dyad * coeff_d' * coeff_p)
        end   
    end
    return out 
end 

function Base.:*(p::Adjoint{<:Any, PauliSum{N,W,T}}, d::DyadSum{N,W,T}) where {N,W,T}
    out = DyadSum(N,T)
    for (dyad, coeff_d) in d
        for (pauli, coeff_p) in p.parent
            new_dyad = pauli*dyad
            sum!(out, new_dyad * coeff_d * coeff_p')
        end   
    end
    return out 
end 

function Base.:*(p::Adjoint{<:Any, PauliSum{N,W,T}}, d::Adjoint{<:Any, DyadSum{N,W,T}}) where {N,W,T}
    out = DyadSum(N,T)
    for (dyad, coeff_d) in d.parent
        for (pauli, coeff_p) in p.parent
            new_dyad = pauli*dyad'
            sum!(out, new_dyad * coeff_d' * coeff_p')
        end   
    end
    return out 
end 


"""
    promote_to_sum(x)

Wrap a single basis element (`Pauli`, `PauliBasis`, `Dyad`, `DyadBasis`, `Ket`)
in its corresponding one-term sum type. Mixed products between single
operators and sums promote the single operand first, so only sum × sum
methods need bespoke implementations.
"""
promote_to_sum(d::Union{Dyad, DyadBasis}) = DyadSum(d)
promote_to_sum(d::Union{Pauli, PauliBasis}) = PauliSum(d)
promote_to_sum(d::Ket) = KetSum(d)

Singles{N} = Union{Dyad{N}, DyadBasis{N}, Pauli{N}, PauliBasis{N}}
Sums{N,T} = Union{DyadSum{N,W,T} where W, PauliSum{N,W,T} where W, Adjoint{<:Any, <:(DyadSum{N,W,T} where W)}, Adjoint{<:Any, <:(PauliSum{N,W,T} where W)}}

Base.:*(s::Singles{N}, p::Sums{N,T}) where {N,T} = promote_to_sum(s) * p
Base.:*(p::Sums{N,T}, s::Singles{N}) where {N,T} = p * promote_to_sum(s)

