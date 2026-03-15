"""
    commutator(A::PauliSum{N,T}, B::PauliSum{N,T}) where {N,T}

Compute the commutator `[A, B] = AB - BA` using an optimized single-pass algorithm.

For Pauli basis elements, `P_i * P_j = phase * P_k` and `P_j * P_i = conj(phase) * P_k`,
so the commutator contribution is `(phase - conj(phase)) * c_i * c_j * P_k = 2i * Im(phase) * c_i * c_j * P_k`.
Commuting pairs (where the phase is real) are skipped entirely.
"""
function commutator(A::PauliSum{N,T}, B::PauliSum{N,T}) where {N,T}
    out = PauliSum(N, T)
    for (pa, ca) in A
        for (pb, cb) in B
            # Skip commuting pairs: [P_a, P_b] = 0
            commute(pa, pb) && continue

            # Compute P_a * P_b
            prod = Pauli(pa) * Pauli(pb)
            c = coeff(prod)
            basis = PauliBasis(prod)

            # phase = c (the phase from P_a * P_b)
            # P_b * P_a has phase conj(c) with the same basis
            # commutator contribution: (c - conj(c)) * ca * cb = 2i*Im(c) * ca * cb
            comm_coeff = 2im * imag(c) * ca * cb

            if haskey(out, basis)
                out[basis] += comm_coeff
            else
                out[basis] = comm_coeff
            end
        end
    end
    return out
end

"""
    anticommutator(A::PauliSum{N,T}, B::PauliSum{N,T}) where {N,T}

Compute the anticommutator `{A, B} = AB + BA` using an optimized single-pass algorithm.

For Pauli basis elements, the anticommutator contribution is
`(phase + conj(phase)) * c_i * c_j * P_k = 2 * Re(phase) * c_i * c_j * P_k`.
Anti-commuting pairs (where the phase is purely imaginary) are skipped entirely.
"""
function anticommutator(A::PauliSum{N,T}, B::PauliSum{N,T}) where {N,T}
    out = PauliSum(N, T)
    for (pa, ca) in A
        for (pb, cb) in B
            # Compute P_a * P_b
            prod = Pauli(pa) * Pauli(pb)
            c = coeff(prod)
            basis = PauliBasis(prod)

            # anticommutator contribution: (c + conj(c)) * ca * cb = 2*Re(c) * ca * cb
            anti_coeff = 2 * real(c) * ca * cb

            # Skip if contribution is zero (anti-commuting pairs with purely imaginary phase)
            iszero(anti_coeff) && continue

            if haskey(out, basis)
                out[basis] += anti_coeff
            else
                out[basis] = anti_coeff
            end
        end
    end
    return out
end
