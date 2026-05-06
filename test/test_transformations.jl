using PauliOperators
using LinearAlgebra
using Test

@testset "Transformations" begin

    @testset "jordan_wigner: anticommutation" begin
        for N in 2:4
            for i in 1:N, j in 1:N
                ai = jordan_wigner(i, N)'   # annihilation
                aj = jordan_wigner(j, N)'   # annihilation
                Mi = Matrix(ai)
                Mj = Matrix(aj)
                # {a_i, a_j†} = δ_ij I
                anticom = Mi * Mj' + Mj' * Mi
                expected = (i == j) ? Matrix{ComplexF64}(I, 2^N, 2^N) : zeros(ComplexF64, 2^N, 2^N)
                @test anticom ≈ expected atol=1e-12
                # {a_i, a_j} = 0
                @test norm(Mi * Mj + Mj * Mi) < 1e-12
            end
        end
    end

    @testset "jordan_wigner: nilpotency" begin
        N = 4
        for f in 1:N
            adag = jordan_wigner(f, N)
            a    = adag'
            @test norm(Matrix(adag) * Matrix(adag)) < 1e-12
            @test norm(Matrix(a)    * Matrix(a))    < 1e-12
        end
    end

    @testset "jordan_wigner: number operator" begin
        N = 3
        for f in 1:N
            a = jordan_wigner(f, N)'
            n_op = a' * a
            M = Matrix(n_op)
            # Diagonal occupation: bit (f-1) of basis index
            occ = [Float64((idx >> (f-1)) & 1) for idx in 0:(2^N - 1)]
            @test diag(M) ≈ occ atol=1e-12
            @test M ≈ Diagonal(occ) atol=1e-12
        end
    end

    # Fock-index ↔ matrix-index helper:
    # qubit 1 is the MSB of n, so matrix-index = bitreverse_in_nq_bits(n).
    fock_to_idx(n, nq) = sum(((n >> (nq - 1 - k)) & 1) << k for k in 0:(nq-1))

    @testset "boson_to_paulis: number operator spectrum" begin
        for nq in 1:3
            bdag = boson_to_paulis(nq)
            # bdag' * bdag = b * b† has eigenvalues {1,…,d-1, 0} == sorted {0,…,d-1}
            evals = sort(real.(eigvals(Matrix(bdag' * bdag))))
            @test evals ≈ collect(0.0:(2^nq - 1)) atol=1e-10
            # bdag * bdag' = b† * b = N has eigenvalues {0,1,…,d-1}
            evals2 = sort(real.(eigvals(Matrix(bdag * bdag'))))
            @test evals2 ≈ collect(0.0:(2^nq - 1)) atol=1e-10
        end
    end

    @testset "boson_to_paulis: raising action" begin
        for nq in 1:3
            bdag = boson_to_paulis(nq)
            Mb = Matrix(bdag)
            dim = 2^nq
            for n in 0:(dim - 2)
                v_n   = zeros(ComplexF64, dim); v_n[fock_to_idx(n,   nq) + 1] = 1
                v_np1 = zeros(ComplexF64, dim); v_np1[fock_to_idx(n+1, nq) + 1] = 1
                @test Mb * v_n ≈ sqrt(n+1) * v_np1 atol=1e-10
            end
            # b† annihilates the top Fock state due to truncation
            v_top = zeros(ComplexF64, dim); v_top[fock_to_idx(dim-1, nq) + 1] = 1
            @test norm(Mb * v_top) < 1e-12
        end
    end

    @testset "boson_to_paulis: explicit matrix nq=2" begin
        Mb = Matrix(boson_to_paulis(2))
        expected = zeros(ComplexF64, 4, 4)
        # Fock indices: n=0→idx 0, n=1→idx 2, n=2→idx 1, n=3→idx 3
        expected[fock_to_idx(1,2)+1, fock_to_idx(0,2)+1] = sqrt(1.0)
        expected[fock_to_idx(2,2)+1, fock_to_idx(1,2)+1] = sqrt(2.0)
        expected[fock_to_idx(3,2)+1, fock_to_idx(2,2)+1] = sqrt(3.0)
        @test Mb ≈ expected atol=1e-12
    end

    @testset "boson_to_paulis: adjoint round-trip" begin
        for nq in 1:3
            b = boson_to_paulis(nq)
            @test Matrix(b)' ≈ Matrix(b') atol=1e-12
        end
    end

    @testset "PauliSum osum (⊕)" begin
        # Single-qubit + single-qubit
        Hs = PauliSum(Pauli("Z"))   # PauliSum{1}
        Hb = PauliSum(Pauli("X"))   # PauliSum{1}
        H  = Hs ⊕ Hb                 # PauliSum{2}

        # Reference: same construction via the Pauli-level osum
        ref = osum(Pauli("Z"), Pauli("X"))
        @test Matrix(H) ≈ Matrix(ref) atol=1e-12

        # Mixed sizes
        Hs2 = PauliSum(Pauli("Z"))    # N=1
        Hb2 = PauliSum(Pauli("XY"))   # M=2
        H2 = Hs2 ⊕ Hb2                # PauliSum{3}
        ref2 = osum(Pauli("Z"), Pauli("XY"))
        @test Matrix(H2) ≈ Matrix(ref2) atol=1e-12

        # Multi-term sums
        A = PauliSum(Pauli("Z")) + PauliSum(Pauli("X"))   # 1-qubit sum
        B = PauliSum(Pauli("XY")) + PauliSum(Pauli("ZI")) # 2-qubit sum
        AB = A ⊕ B
        # Compare against dense direct sum on the H-side: A⊗I_2 + I_1⊗B
        I1 = PauliSum(1, ComplexF64); I1[PauliBasis{1}(Int128(0), Int128(0))] = 1.0+0im
        I2 = PauliSum(2, ComplexF64); I2[PauliBasis{2}(Int128(0), Int128(0))] = 1.0+0im
        ref3 = Matrix(A ⊗ I2) + Matrix(I1 ⊗ B)
        @test Matrix(AB) ≈ ref3 atol=1e-12
    end

end
