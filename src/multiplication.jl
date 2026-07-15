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
    TT = promote_type(T, ComplexF64)
    out = KetSum(N, T=TT)
    sizehint!(out, length(ks))
    for (k,c) in ks
        c2,k2 = p*k
        tmp = get(out, k2, zero(TT))
        out[k2] = tmp + c2*c
    end
    return out
end



"""
    Base.:*(O::PauliSum{N,W,T}, k::Ket{N})

Apply a sum of Paulis to a basis state, returning a `KetSum` with one entry
per distinct `x` bitstring in `O`.
"""
function Base.:*(O::PauliSum{N,W,T}, k::Ket{N,W}) where {N,W,T}
    TT = promote_type(T, ComplexF64)
    out = KetSum(N, T=TT)
    for (p,c) in O
        c2,k2 = p*k
        tmp = get(out, k2, zero(TT))
        out[k2] = tmp + c2*c
    end
    return out
end

"""
    Base.:*(O::AnyPauliSum{N,W}, ks::KetSum{N,W})

Apply a sum of Paulis to a linear combination of basis states:
`(Σᵢ cᵢ Pᵢ)(Σₖ vₖ |k⟩)`. Returns a complex `KetSum`.

Terms are grouped by x-string: every Pauli in an x-group maps `|k⟩` to the
same `|k ⊻ x⟩`, and its phase factors as
`[i^sp(P)·(-1)^parity(z&x)] · (-1)^parity(z&k)` with the bracket a per-Pauli
constant, so each group's contribution to a slot is accumulated in a register
and the output Dict is touched once per (x-group, ket) instead of per
(Pauli, ket).
"""
function Base.:*(O::AnyPauliSum{N,W,TO}, ks::KetSum{N,W,TK}) where {N,W,TO,TK}
    TT = promote_type(TO, TK, ComplexF64)
    groups = Dict{W,Tuple{Vector{W},Vector{TT}}}()
    for (p,c) in O
        zs, cs = get!(() -> (W[], TT[]), groups, p.x)
        s = count_ones(p.z & p.x) % 2
        push!(zs, p.z)
        push!(cs, PHASE_TBL[(symplectic_phase(p) + 2*s)%4 + 1] * c)
    end
    kv = collect(keys(ks))
    cv = collect(values(ks))
    out = KetSum(N, T=TT)
    sizehint!(out, min(length(groups)*length(ks), 1 << 20))
    for (x, (zs, cs)) in groups
        m = length(zs)
        for j in eachindex(kv)
            v = kv[j].v
            acc = zero(TT)
            @inbounds @simd for i in 1:m
                acc += ifelse(isodd(count_ones(zs[i] & v)), -cs[i], cs[i])
            end
            k2 = Ket{N}(x ⊻ v)
            out[k2] = get(out, k2, zero(TT)) + acc*cv[j]
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

