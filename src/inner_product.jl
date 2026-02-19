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