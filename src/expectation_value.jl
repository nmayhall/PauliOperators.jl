
function expectation_value(p::Union{PauliBasis{N}, Pauli{N}}, ket::Ket{N}) where N
    sgn = count_ones(p.z & ket.v)
    return PHASE_TBL[2*sgn%4+1] * (p.x == 0) * coeff(p)
end



function expectation_value(p::Union{PauliBasis{N}, Pauli{N}}, d::Union{Dyad{N}, DyadBasis{N}}) where N
    sgn = count_ones(p.z & d.bra.v)  # sgn <j| = <j| z
    val = d.ket.v ⊻ d.bra.v == p.x # <j|x|i>
    return val * PHASE_TBL[(symplectic_phase(p) + 2*sgn)%4 + 1]*coeff(p)
end

function expectation_value(p::PauliSum{N,T}, d::Union{Ket{N}, Dyad{N}, DyadBasis{N}}) where {N,T}
    nt = reduction_nthreads(length(p))
    if nt == 1
        eval = zero(T)
        for (pi,ci) in p
            eval += expectation_value(pi, d) * ci
        end
        return eval
    end
    # threaded term reduction: per-thread partial sums, then combine
    prs = collect(p)
    partials = zeros(T, nt)
    ranges = chunk_ranges(length(prs), nt)
    @sync for c in 1:nt
        Threads.@spawn begin
            s = zero(T)
            @inbounds for idx in ranges[c]
                pr = prs[idx]
                s += expectation_value(pr.first, d) * pr.second
            end
            partials[c] = s
        end
    end
    return sum(partials)
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

function expectation_value(O::PauliSum, v::KetSum)
    ev = 0
    for (p,c) in O
        for (k1,c1) in v
            ev += expectation_value(p,k1)*c*c1'*c1
            for (k2,c2) in v
                k2 != k1 || continue
                ev += matrix_element(k2', p, k1)*c*c2'*c1
            end
        end
    end
    return ev
end


function matrix_element(b::Bra{N}, p::PauliBasis{N}, k::Ket{N}) where N
    sgn = count_ones(p.z & b.v)  # sgn <j| = <j| z
    val = k.v ⊻ b.v == p.x # <j|x|i>
    return val * PHASE_TBL[(symplectic_phase(p) + 2*sgn)%4 + 1]
end

function matrix_element(b::Bra{N}, p::PauliSum{N,T}, k::Ket{N}) where {N,T}
    nt = reduction_nthreads(length(p))
    if nt == 1
        eval = zero(T)
        for (pi,ci) in p
            eval += matrix_element(b, pi, k) * ci
        end
        return eval
    end
    prs = collect(p)
    partials = zeros(T, nt)
    ranges = chunk_ranges(length(prs), nt)
    @sync for c in 1:nt
        Threads.@spawn begin
            s = zero(T)
            @inbounds for idx in ranges[c]
                pr = prs[idx]
                s += matrix_element(b, pr.first, k) * pr.second
            end
            partials[c] = s
        end
    end
    return sum(partials)
end

function matrix_element(b::KetSum{N}, p::PauliBasis{N}, k::KetSum{N}) where {N}
    if length(k) < length(b)
        pk = p*k
        return inner_product(b, pk)
    else 
        pb = p*b
        return inner_product(pb, k)
    end
end

function matrix_element(b::KetSum{N}, p::PauliSum{N}, k::KetSum{N}) where {N}
    σ = p*k
    return inner_product(b,σ)
end