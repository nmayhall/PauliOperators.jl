"""
    get_weight_counts(O::PauliSum{N}) where N

Return a vector of length N+1 where entry i contains the number of terms
with Pauli weight i-1.
"""
function get_weight_counts(O::AnyPauliSum{N}) where N
    counts = zeros(Int, N+1)
    for (p, _) in O
        counts[weight(p)+1] += 1
    end
    return counts
end

"""
    get_weight_probs(O::PauliSum{N}) where N

Return a vector of length N+1 where entry i contains the sum of |c|²
for terms with Pauli weight i-1.
"""
function get_weight_probs(O::AnyPauliSum{N}) where N
    probs = zeros(N+1)
    for (p, c) in O
        probs[weight(p)+1] += abs2(c)
    end
    return probs
end

"""
    get_majorana_weight_counts(O::PauliSum{N}) where N

Return a vector of length 2N+1 where entry i contains the number of terms
with Majorana weight i-1.
"""
function get_majorana_weight_counts(O::AnyPauliSum{N}) where N
    counts = zeros(Int, 2N+1)
    for (p, _) in O
        counts[majorana_weight(p)+1] += 1
    end
    return counts
end

"""
    get_majorana_weight_probs(O::PauliSum{N}) where N

Return a vector of length 2N+1 where entry i contains the sum of |c|²
for terms with Majorana weight i-1.
"""
function get_majorana_weight_probs(O::AnyPauliSum{N}) where N
    probs = zeros(2N+1)
    for (p, c) in O
        probs[majorana_weight(p)+1] += abs2(c)
    end
    return probs
end

"""
    find_top_k(O::PauliSum{N,T}, k::Int) where {N,T}

Return the `k` terms with largest absolute coefficients, sorted by decreasing |c|.
Returns a `Vector{Pair{PauliBasis{N}, T}}`. Efficient for k << length(O).
"""
function find_top_k(O::AnyPauliSum{N,T}, k::Int) where {N,T}
    k > 0 || throw(ArgumentError("k must be positive"))
    k = min(k, length(O))

    top_keys = Vector{PauliBasis{N}}(undef, k)
    top_vals = Vector{T}(undef, k)
    top_abs = Vector{Float64}(undef, k)

    n_found = 0
    min_val = 0.0
    min_idx = 1

    @inbounds for (key, val) in O
        abs_val = abs(val)

        if n_found < k
            n_found += 1
            top_keys[n_found] = key
            top_vals[n_found] = val
            top_abs[n_found] = abs_val

            if abs_val < min_val || n_found == 1
                min_val = abs_val
                min_idx = n_found
            end
        elseif abs_val > min_val
            top_keys[min_idx] = key
            top_vals[min_idx] = val
            top_abs[min_idx] = abs_val

            min_val = top_abs[1]
            min_idx = 1
            for i in 2:k
                if top_abs[i] < min_val
                    min_val = top_abs[i]
                    min_idx = i
                end
            end
        end
    end

    p = sortperm(view(top_abs, 1:n_found), rev=true)
    return [top_keys[p[i]] => top_vals[p[i]] for i in 1:n_found]
end

"""
    largest(ps::PauliSum{N,T}) where {N,T}

Return the term with the largest absolute coefficient as a single-term PauliSum.
"""
function largest(ps::PauliSum{N,T}) where {N,T}
    _, max_key = findmax(v -> abs(v), ps)
    return PauliSum{N,T}(max_key => ps[max_key])
end

"""
    largest_diag(ps::PauliSum{N,T}) where {N,T}

Return the `PauliBasis => coefficient` pair for the diagonal term (x == 0)
with the largest absolute coefficient.
"""
function largest_diag(ps::PauliSum{N,T}) where {N,T}
    return argmax(kv -> abs(last(kv)), filter(p -> p.first.x == 0, ps))
end

"""
    Base.Matrix(O::PauliSum{N,T}, S::Vector{<:Ket{N}}) where {N,T}

Construct the matrix representation of operator `O` in the subspace spanned by kets `S`.

Returns an `nS × nS` matrix where `M[i,j] = ⟨S[i]|O|S[j]⟩`.

Uses X-bitstring grouping for efficiency: for each ket pair `(i,j)`, only Pauli terms
whose X-bitstring matches `S[i].v ⊻ S[j].v` are visited, rather than all terms in `O`.
"""
function Base.Matrix(O::AnyPauliSum{N,T}, S::Vector{<:Ket{N}}) where {N,T}
    nS = length(S)

    # Group Pauli terms by X-bitstring for efficient subspace matrix construction.
    # Only terms with x == S[i].v ⊻ S[j].v contribute to M[i,j].
    W = uinttype(N)
    x_groups = Dict{W, Vector{Tuple{W, T}}}()
    for (P, c) in O
        group = get!(Vector{Tuple{W, T}}, x_groups, W(P.x))
        push!(group, (P.z, c))
    end

    empty_group = Vector{Tuple{W, T}}()

    M = zeros(promote_type(T, ComplexF64), nS, nS)
    for i in 1:nS
        M[i, i] = expectation_value(O, S[i])
        for j in i+1:nS
            x = W(S[i].v ⊻ S[j].v)
            for (z, c) in get(x_groups, x, empty_group)
                P = PauliBasis{N,W}(z, x)
                phase, _ = P * S[j]
                M[i, j] += phase * c

                phase, _ = P * S[i]
                M[j, i] += phase * c
            end
        end
    end
    return M
end

"""
    Base.Vector(K::KetSum{N,T}, S::Vector{<:Ket{N}}) where {N,T}

Project a KetSum onto the subspace defined by basis kets `S`.

Returns a vector of length `length(S)` with `v[i] = K[S[i]]`.
"""
function Base.Vector(K::KetSum{N,T}, S::Vector{<:Ket{N}}) where {N,T}
    nS = length(S)
    v = zeros(T, nS)
    for (i, ki) in enumerate(S)
        v[i] = get(K, ki, zero(T))
    end
    return v
end
