"""
    commutator(A::PauliSum{N,W,T}, B::PauliSum{N,W,T}) where {N,W,T}

Compute the commutator `[A, B] = AB - BA` using an optimized single-pass algorithm.

For Pauli basis elements, `P_i * P_j = phase * P_k` and `P_j * P_i = conj(phase) * P_k`,
so the commutator contribution is `(phase - conj(phase)) * c_i * c_j * P_k = 2i * Im(phase) * c_i * c_j * P_k`.
Commuting pairs (where the phase is real) are skipped entirely.

# Optimization
Rather than constructing intermediate `Pauli` objects and doing complex arithmetic,
the phase exponent `k` (where phase = `i^k`) is computed directly from bitstring
operations. For non-commuting pairs, `k` is always odd (phase is `±i`), so the
imaginary part is `±1` and can be determined by a single integer comparison.
"""
function commutator(A::PauliSum{N,W,T}, B::PauliSum{N,W,T}) where {N,W,T}
    out = PauliSum(N, T)
    for (pa, ca) in A
        n_a = count_ones(pa.z & pa.x)
        for (pb, cb) in B
            # Fused commute check + phase computation:
            # commute(pa, pb) iff iseven(m1 - m2), and m1 is reused for the phase
            m1 = count_ones(pa.x & pb.z)
            m2 = count_ones(pa.z & pb.x)
            iseven(m1 - m2) && continue

            # Product basis (XOR of bitstrings)
            z_prod = pa.z ⊻ pb.z
            x_prod = pa.x ⊻ pb.x

            # Phase exponent k (mod 4) where P_a * P_b = i^k * P_{prod}
            # k ≡ n_ab - n_a - n_b + 2*m1 (mod 4)
            # where n = count_ones(z & x) counts Y factors in each operator
            n_b = count_ones(pb.z & pb.x)
            n_ab = count_ones(z_prod & x_prod)
            k = mod(n_ab - n_a - n_b + 2 * m1, 4)

            # For non-commuting pairs, k is odd: i^1 = i, i^3 = -i
            # imag(i^k): k=1 → +1, k=3 → -1, so sign = 2 - k
            sign = 2 - k

            basis = PauliBasis{N}(z_prod, x_prod)
            comm_coeff = (2 * sign) * im * ca * cb
            out[basis] = get(out, basis, zero(T)) + comm_coeff
        end
    end
    return out
end

"""
    anticommutator(A::PauliSum{N,W,T}, B::PauliSum{N,W,T}) where {N,W,T}

Compute the anticommutator `{A, B} = AB + BA` using an optimized single-pass algorithm.

For Pauli basis elements, the anticommutator contribution is
`(phase + conj(phase)) * c_i * c_j * P_k = 2 * Re(phase) * c_i * c_j * P_k`.
Anti-commuting pairs (where the phase is purely imaginary) are skipped entirely.

Uses the same bitstring-only phase computation as `commutator`.
"""
function anticommutator(A::PauliSum{N,W,T}, B::PauliSum{N,W,T}) where {N,W,T}
    out = PauliSum(N, T)
    for (pa, ca) in A
        n_a = count_ones(pa.z & pa.x)
        for (pb, cb) in B
            # Product basis
            z_prod = pa.z ⊻ pb.z
            x_prod = pa.x ⊻ pb.x

            # Phase exponent k (mod 4)
            n_b = count_ones(pb.z & pb.x)
            n_ab = count_ones(z_prod & x_prod)
            m = count_ones(pa.x & pb.z)
            k = mod(n_ab - n_a - n_b + 2 * m, 4)

            # For anticommutator, only even k contributes (real phase):
            # i^0 = 1, i^2 = -1, so real(i^k) = 1 - k for k ∈ {0, 2}
            isodd(k) && continue

            sign = 1 - k  # k=0 → +1, k=2 → -1
            anti_coeff = (2 * sign) * ca * cb

            basis = PauliBasis{N}(z_prod, x_prod)
            out[basis] = get(out, basis, zero(T)) + anti_coeff
        end
    end
    return out
end
