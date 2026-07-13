"""
    inner_product(O1::PauliSum{N,T}, O2::PauliSum{N,T}) where {N,T}

Evaluate the Liouville space inner product: tr(O1'*O2)
"""
function inner_product(O1::PauliSum{N,T}, O2::PauliSum{N,T}) where {N,T}
    out = T(0)
    if length(O1) < length(O2)
        for (p1,c1) in O1
            if haskey(O2,p1)
                out += c1'*O2[p1]
            end
        end
    else
        for (p2,c2) in O2
            if haskey(O1,p2)
                out += c2*O1[p2]'
            end
        end
    end
    return out
end

"""
    inner_product_threaded(O1::PauliSum{N,T}, O2::PauliSum{N,T}) where {N,T}

Threaded variant of [`inner_product`](@ref): `tr(O1'*O2)`. Iterates the smaller
sum's terms and probes the larger (read-only), split across threads with
per-thread partial sums — a race-free reduction (no shared writes; both dicts are
only read). Falls back to the serial path below the threading threshold
(`reduction_nthreads`). Same value as `inner_product` up to floating-point
summation order.

Prototype for the `dbf_groundstate` hot path: `opt_theta` evaluates six of these
per rotation, and they are otherwise single-threaded.
"""
function inner_product_threaded(O1::PauliSum{N,T}, O2::PauliSum{N,T}) where {N,T}
    # iterate the smaller dict, probe the larger
    small, big = length(O1) <= length(O2) ? (O1, O2) : (O2, O1)
    conj1 = length(O1) <= length(O2)          # match the serial conjugation convention
    nt = reduction_nthreads(length(small))
    if nt == 1
        out = T(0)
        for (p, c) in small
            v = get(big, p, nothing)
            v === nothing && continue
            out += conj1 ? c'*v : v'*c
        end
        return out
    end
    prs = collect(small)
    partials = zeros(T, nt)
    ranges = chunk_ranges(length(prs), nt)
    @sync for cc in 1:nt
        Threads.@spawn begin
            s = zero(T)
            @inbounds for idx in ranges[cc]
                p, c = prs[idx].first, prs[idx].second
                v = get(big, p, nothing)
                v === nothing && continue
                s += conj1 ? c'*v : v'*c
            end
            partials[cc] = s
        end
    end
    return sum(partials)
end

function inner_product(k1::KetSum{N,T}, k2::KetSum{N,T}) where {N,T}
    out = T(0)
    if length(k1) < length(k2)
        for (p1,c1) in k1
            if haskey(k2,p1)
                out += c1'*k2[p1]
            end
        end
    else
        for (p2,c2) in k2
            if haskey(k1,p2)
                out += c2*k1[p2]'
            end
        end
    end
    return out
end