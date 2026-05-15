"""
    weight(p::PauliBasis)

Number of non-identity single-qubit Pauli factors.
"""
function weight(p::PauliBasis)
    return count_ones(p.x | p.z)
end

"""
    coeff_clip!(ps::PauliSum{N}, thresh::Real)

Remove Pauli terms with |coefficient| <= `thresh`.
"""
function coeff_clip!(ps::PauliSum{N}, thresh::Real) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

"""
    clip!(ps::PauliSum{N}; thresh=1e-16)

!!! warning "Deprecated"
    Use `coeff_clip!(ps, thresh)` instead.
"""
clip!(ps::PauliSum; thresh=1e-16) = coeff_clip!(ps, thresh)

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

# Algorithm
Uses a branchless O(1) bitwise algorithm. The original per-bit scan tracks a `control`
flag that starts `true` (at the MSB) and flips at each X/Y site. This is equivalent to
a suffix parity of the X-bitstring, computed via parallel prefix XOR (7 steps for Int128).
The control mask then selects which Z-only and I-only positions contribute to the weight.
"""
function majorana_weight(p::Union{PauliBasis{N}, Pauli{N}}) where N
    # Use unsigned to ensure logical (zero-filling) right shifts
    xbits = unsigned(p.x)       # X/Y positions (both X and Y have x-bit set)
    zbits = unsigned(p.z) & ~xbits   # Z-only positions

    # Suffix XOR: after this, bit j of S = XOR of xbits at positions j, j+1, ..., 127
    # This computes the running parity of X/Y sites from MSB downward
    S = xbits
    S ⊻= S >> 1
    S ⊻= S >> 2
    S ⊻= S >> 4
    S ⊻= S >> 8
    S ⊻= S >> 16
    S ⊻= S >> 32
    S ⊻= S >> 64

    # Control mask: bit j is set iff control=true at position j
    # control(j) = true ⊕ (parity of xbits at positions j+1..N-1)
    #            = NOT(suffix_xor(j) ⊕ xbits(j))
    # Above N bits: S=0, xbits=0, so ctrl=all 1s → ~ctrl=all 0s (auto-masks)
    ctrl = ~(S ⊻ xbits)

    # Weight contributions:
    #   X/Y sites: always +1
    #   Z-only sites where control=true: +2
    #   I-only sites where control=false: +2
    return count_ones(xbits) +
           2 * count_ones(zbits & ctrl) +
           2 * count_ones(~(unsigned(p.z) | xbits) & ~ctrl)
end

"""
    majorana_weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}

Remove terms with Majorana weight above `max_weight`.
"""
function majorana_weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}
    return filter!(p->majorana_weight(p.first) <= max_weight, ps)
end

"""
    coeff_clip!(ks::KetSum{N}, thresh::Real) where {N}

Remove Ket terms with |coefficient| <= `thresh`.
"""
function coeff_clip!(ks::KetSum{N}, thresh::Real) where {N}
    return filter!(p->abs(p.second) > thresh, ks)
end

"""
    clip!(ks::KetSum{N}; thresh=1e-16) where {N}

!!! warning "Deprecated"
    Use `coeff_clip!(ks, thresh)` instead.
"""
clip!(ks::KetSum; thresh=1e-16) = coeff_clip!(ks, thresh)

"""
    stochastic_clip!(ps::PauliSum{N,T}, ε::Real; rng=Random.default_rng())

Unbiased stochastic compression (Russian Roulette) of a PauliSum.

For each term (basis, c):
- If |c| >= ε: keep unchanged
- If |c| < ε: with probability |c|/ε, promote to ε·sign(c); otherwise delete

This is unbiased: E[c̃] = c for every term.
"""
function stochastic_clip!(ps::PauliSum{N,T}, ε::Real;
                          rng::AbstractRNG=Random.default_rng()) where {N,T}
    to_delete = PauliBasis{N}[]

    for (basis, c) in ps
        ac = abs(c)
        ac < ε || continue

        if rand(rng) < ac / ε
            ps[basis] = ε * (c / ac)   # promote (preserves phase for complex T)
        else
            push!(to_delete, basis)     # mark for deletion
        end
    end

    for basis in to_delete
        delete!(ps, basis)
    end
    return ps
end

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
