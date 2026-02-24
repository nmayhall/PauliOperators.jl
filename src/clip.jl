"""
    weight(p::PauliBasis)

Number of non-identity single-qubit Pauli factors.
"""
function weight(p::PauliBasis)
    return count_ones(p.x | p.z)
end

"""
    coeff_clip!(ps::PauliSum{N}; thresh=1e-16)

Hard truncation: remove terms with |c| <= thresh.
"""
function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

"""
    weight_clip!(ps::PauliSum{N}, max_weight::Int)

Remove terms with Pauli weight above max_weight.
"""
function weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}
    return filter!(p->weight(p.first) <= max_weight, ps)
end
