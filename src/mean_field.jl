"""
    _partial_alt_binom(n::Int, k_max::Int) -> Int

Compute `Σ_{m=0}^{min(k_max,n)} C(n,m) (-1)^m`.

Edge cases:
- `k_max < 0` returns `0`.
- `k_max >= n` returns `1` if `n == 0` else `0` (the full alternating row sum vanishes).
"""
function _partial_alt_binom(n::Int, k_max::Int)
    k_max < 0 && return 0
    if k_max >= n
        return n == 0 ? 1 : 0
    end
    result = 1
    binom  = 1  # C(n, 0)
    for m in 1:k_max
        binom  = binom * (n - m + 1) ÷ m   # C(n, m) — exact integer update
        result += iseven(m) ? binom : -binom
    end
    return result
end


# Emit sinks for the factorization kernel. `_mf_emit!(sink, z, x, c)` receives one
# output term (Z/X packed as Int128 bitstrings, coefficient `c`). Two sinks:
#   - PauliSum: accumulate onto the key (get/set +=).
#   - SparsePauliVector: stage the raw triple into the append region (defined in
#     spv_evolve.jl); the later sort-merge dedups, so no Dict is needed at all.
@inline function _mf_emit!(out::PauliSum{N,T}, z::Int128, x::Int128, c::T) where {N,T}
    key = PauliBasis{N}(z, x)
    out[key] = get(out, key, zero(T)) + c
    return nothing
end

# Emit every size-`need` subset of the set bits in `remaining` into `sink`, as a
# Z-string on `y_z_mask | T_mask` with off-diagonal support `x`. Allocation-free:
# the running subset mask `T_mask` and its ±1 mean product `sgn` are threaded as
# arguments (no index buffer, no closure), and `coeff_base = c · full_ε · f_t`
# already folds in the term coefficient, the full mean product, and the
# level multiplicity, so a leaf only multiplies by `sgn`. Standard bit-combination
# recursion with a `count_ones` feasibility prune.
@inline function _mf_emit_subsets!(sink, remaining::Int128, need::Int,
                                   y_z_mask::Int128, x::Int128, coeff_base::T,
                                   ψv::Int128, T_mask::Int128, sgn::Int) where {T}
    if need == 0
        _mf_emit!(sink, y_z_mask | T_mask, x, coeff_base * sgn)
        return
    end
    bits = remaining
    while bits != zero(Int128)
        lb = bits & (-bits)          # lowest set bit
        bits ⊻= lb                    # drop it from this level's future choices
        count_ones(bits) >= need - 1 || break   # not enough bits left to finish
        q  = trailing_zeros(lb)       # 0-based qubit position
        εq = 1 - 2 * Int((ψv >> q) & 1)
        _mf_emit_subsets!(sink, bits, need - 1, y_z_mask, x, coeff_base,
                          ψv, T_mask | lb, sgn * εq)
    end
    return
end


"""
    _mean_field_accumulate!(sink, pb::PauliBasis{N}, c, ψ::Ket{N}, k::Int)

Emit the order-`k` mean-field factorization of `c · pb` around `ψ` into `sink`
(a `PauliSum` or a `SparsePauliVector`; see `_mf_emit!`). The allocation-free core
of [`mean_field_factorize`](@ref) — see it for the math. `c`'s type must match the
sink's coefficient type.
"""
function _mean_field_accumulate!(sink, pb::PauliBasis{N},
                                 c::T, ψ::Ket{N}, k::Int) where {N,T}
    n_xy = count_ones(pb.x)
    n_xy > k && return sink                     # off-diagonal support alone exceeds budget

    z_only   = pb.z & ~pb.x                      # pure-Z qubits (the only fluctuating means)
    n_z      = count_ones(z_only)
    y_z_mask = pb.z & pb.x                       # Z-bits on Y qubits, carried unchanged
    budget   = k - n_xy                          # max |T|
    ψv       = ψ.v

    # full_ε = ∏_j ⟨ψ|Z_j|ψ⟩ over the pure-Z qubits = (-1)^popcount(z_only & ψ)
    full_ε = 1 - 2 * (count_ones(z_only & ψv) & 1)
    cε = c * full_ε

    # One level per retained-subset size t; skip whole levels whose multiplicity
    # vanishes (this collapses the k ≥ weight case to the single original term).
    for t in 0:min(budget, n_z)
        f = _partial_alt_binom(n_z - t, budget - t)
        f == 0 && continue
        _mf_emit_subsets!(sink, z_only, t, y_z_mask, pb.x, cε * f, ψv, Int128(0), 1)
    end
    return sink
end


"""
    mean_field_factorize(pb::PauliBasis{N}, c, ψ::Ket{N}, k::Int) -> PauliSum{N,T}

Order-`k` mean-field factorization of the single Pauli term `c · pb` around the
computational-basis reference `ψ`.

Replaces `pb` with a sum of Pauli strings of weight ≤ `k` that is exact when
`k ≥ weight(pb)` and preserves `⟨ψ|·|ψ⟩` for every `k`. Uses the multinomial
fluctuation decomposition with `δP_j = P_j − ⟨P_j⟩ I`; on a computational-basis
reference only pure-Z qubits contribute non-trivially, so enumeration is over
subsets of the pure-Z qubit positions in `pb`.
"""
function mean_field_factorize(pb::PauliBasis{N}, c::T, ψ::Ket{N}, k::Int) where {N,T}
    out = PauliSum(N, T)
    _mean_field_accumulate!(out, pb, c, ψ, k)
    return out
end


"""
    mean_field_factorize!(O::PauliSum{N,T}, ψ::Ket{N}, k::Int)

In-place replacement of every term in `O` with `weight(pb) > k` by its order-`k`
mean-field factorization around `ψ`. See [`mean_field_factorize`](@ref).
"""
function mean_field_factorize!(O::PauliSum{N,T}, ψ::Ket{N}, k::Int) where {N,T}
    high = [pb for (pb, _) in O if weight(pb) > k]
    for pb in high
        c = pop!(O, pb)                          # remove the high-weight term
        _mean_field_accumulate!(O, pb, c, ψ, k)  # fold its replacements straight back in
    end
    return O
end
