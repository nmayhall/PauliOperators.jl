
function expectation_value(p::Union{PauliBasis{N}, Pauli{N}}, ket::Ket{N}) where N
    if count_ones(p.z & ket.v)%2 == 0
        return (p.x == 0) * coeff(p)
    else
        return -1*(p.x == 0) * coeff(p)
    end
    # return (-1)^count_ones(p.z & ket.v) * (p.x == 0) * coeff(p)
end



function expectation_value(p::Union{PauliBasis{N}, Pauli{N}}, d::Union{Dyad{N}, DyadBasis{N}}) where N
    sgn = count_ones(p.z & d.bra.v)  # sgn <j| = <j| z 
    val = d.ket.v ⊻ d.bra.v == p.x # <j|x|i>
    sgn1 = 1
    phs1 = 1
    if sgn % 2 != 0
        sgn1 = -1
    end
    if symplectic_phase(p) == 1
        phs1 = 1im
    elseif symplectic_phase(p) == 2
        phs1 = -1
    elseif symplectic_phase(p) == 3
        phs1 = -1im
    end

    return sgn1 * phs1 * val * coeff(p) * coeff(d)
    # return (-1)^sgn * val * coeff(p) * coeff(d) * 1im^symplectic_phase(p)
end

function expectation_value(p::PauliSum{N,T}, d::Union{Ket{N}, Dyad{N}, DyadBasis{N}}) where {N,T}
    eval = zero(T)
    for (pi,ci) in p
        eval += expectation_value(pi, d) * ci
    end
    return eval 
end

function expectation_value(p::Union{PauliBasis{N}, Pauli{N}}, d::DyadSum{N,T}) where {N,T}
    eval = zero(T)
    for (di,ci) in d
        eval += expectation_value(p, di) * ci
    end
    return eval 
end

function expectation_value(p::PauliSum{N,T}, d::DyadSum{N,T}) where {N,T}
    eval = zero(T)
    for (pi,ci) in p
        for (dj,cj) in d
            eval += expectation_value(pi, dj) * ci * cj
        end
    end
    return eval 
end
