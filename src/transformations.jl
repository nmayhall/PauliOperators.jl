"""
    jordan_wigner(f::Integer, N::Integer)

Jordan–Wigner image of the fermionic creation operator ``\\hat{a}_f^\\dagger``
acting on `N` spin orbitals:

    a†_f = ½(X_f - i Y_f) ⊗ Z_{f-1} ⋯ Z_1

# Arguments
- `f`: orbital index in `1..N` for the creation operator.
- `N`: total number of spin orbitals.

Returns a `PauliSum{N, ComplexF64}`.
"""
function jordan_wigner(f::Integer, N::Integer)
    1 ≤ f ≤ N || throw(ArgumentError("f must satisfy 1 ≤ f ≤ N (got f=$f, N=$N)"))
    z_pre = Int128(2)^(f-1) - Int128(1)   # Z on sites 1..f-1
    x_f   = Int128(2)^(f-1)               # X (or Y) on site f
    out = PauliSum(N, ComplexF64)
    out[PauliBasis{N}(z_pre,        x_f)] =  0.5 + 0.0im   # ½ X_f Z_{<f}
    out[PauliBasis{N}(z_pre | x_f,  x_f)] = -0.5im         # -½i Y_f Z_{<f}
    return out
end


"""
    boson_to_paulis(nqubits::Integer; verbose=0)

Binary (truncated) encoding of a bosonic raising operator `b†` into `nqubits`
qubits, giving access to a Fock space of dimension `d = 2^nqubits`:

    b† = ∑_{n=0}^{d-2} √(n+1) |n+1⟩⟨n|

The bosonic occupation number `n` is encoded in the qubit register with qubit 1
as the most-significant bit (i.e. `n = q₁·2^{nq-1} + q₂·2^{nq-2} + … + q_{nq}·2^0`).
Adjoint the result to obtain the lowering operator `b = (b†)†`.

The encoding is the same one used in the original spin–boson workflow on the
`spinboson` branch (formerly `boson_binary_transformation`).

Returns a `PauliSum{nqubits, ComplexF64}`.
"""
function boson_to_paulis(nqubits::Integer; verbose=0)
    rep1 = []

    for ni in 0:nqubits-1
        stride = 2^(nqubits-ni)

        prodlist = ["I" for _ in 1:nqubits]
        prodlist[ni+1] = "σp"
        for nj in ni+2:nqubits
            prodlist[nj] = "σm"
        end

        start = 2^(nqubits-ni-1)
        dec_idx = Int(0)

        for _ in 1:2^ni
            str = bitstring(dec_idx)[64-ni+1:end]
            for (idx, ch) in enumerate(str)
                if ch == '1'
                    prodlist[idx] = "M"
                elseif ch == '0'
                    prodlist[idx] = "N"
                else
                    throw(ErrorException("unexpected bitstring character"))
                end
            end

            push!(rep1, (deepcopy(prodlist), start + dec_idx*stride))
            dec_idx += 1
        end
    end

    if verbose > 0
        println(" -------- Intermediate Representation -------- ")
        for entry in rep1
            for j in entry[1]
                @printf("%2s ", j)
            end
            @printf("√%i\n", entry[2])
        end
    end

    Nproj  = ((0.5,  "I"), (0.5,  "Z"))
    Mproj  = ((0.5,  "I"), (-0.5, "Z"))
    sigmap = ((0.5,  "X"), (-0.5im, "Y"))
    sigmam = ((0.5,  "X"), (0.5im,  "Y"))

    if verbose > 0
        println(" -------- Pauli Representation -------- ")
    end

    bdag = PauliSum(nqubits, ComplexF64)

    for entry in rep1
        to_prod = []
        for j in entry[1]
            if j == "σp"
                push!(to_prod, sigmap)
            elseif j == "σm"
                push!(to_prod, sigmam)
            elseif j == "N"
                push!(to_prod, Nproj)
            elseif j == "M"
                push!(to_prod, Mproj)
            end
        end

        for prod in Iterators.product(to_prod...)
            c = 1
            op = ""
            for term in prod
                c *= term[1]
                op *= term[2]
            end

            pbs = Pauli(op)
            key = PauliBasis(pbs)
            val = coeff(pbs) * c * sqrt(entry[2])
            bdag[key] = get(bdag, key, zero(ComplexF64)) + val
        end
    end

    return bdag
end
