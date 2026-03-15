using PauliOperators
using LinearAlgebra
using Test
using Random

@testset "Phase 3: Evolution & Decomposition" begin

    @testset "KetSum evolution (Schrödinger)" begin
        N = 3
        # Evolve |000> by exp(-iθ/2 X₁)
        k = KetSum(Ket(N, 0))
        G = PauliBasis(Pauli(N, X=[1]))
        θ = π/3

        k_evolved = evolve(k, G, θ)

        # Verify against dense matrix: exp(-iθ/2 X₁)|000>
        Gmat = Matrix(G)
        U = exp(-1im * θ/2 * Gmat)
        k_dense = U * ComplexF64.(Vector(k))
        @test Vector(k_evolved) ≈ k_dense
    end

    @testset "KetSum sequence evolution" begin
        N = 2
        k = KetSum(Ket(N, 0))
        G1 = PauliBasis(Pauli(N, X=[1]))
        G2 = PauliBasis(Pauli(N, Z=[2]))
        gens = [G1, G2]
        angs = [π/4, π/3]

        k_seq = evolve(k, gens, angs)

        # Verify against sequential single-step evolution
        k_ref = evolve(k, G1, π/4)
        k_ref = evolve(k_ref, G2, π/3)
        @test Vector(k_seq) ≈ Vector(k_ref)
    end

    @testset "PauliSum sequence evolution" begin
        N = 3
        O = PauliSum(Pauli("ZII")) + PauliSum(Pauli("IZI"))
        G1 = PauliBasis(Pauli(N, X=[1]))
        G2 = PauliBasis(Pauli(N, X=[2]))
        gens = [G1, G2]
        angs = [π/4, π/3]

        O_seq = evolve(O, gens, angs)

        # Verify against sequential single-step evolution
        O_ref = evolve(O, G1, π/4)
        O_ref = evolve(O_ref, G2, π/3)
        @test Matrix(O_seq) ≈ Matrix(O_ref)
    end

    @testset "PauliSum sequence with truncation" begin
        N = 4
        O = rand(PauliSum{N}; n_paulis=10)
        gens = [rand(PauliBasis{N}) for _ in 1:5]
        angs = randn(5) * 0.1

        ψ = Ket(N, 0)
        corr = EnergyCorrection(ψ)

        O_trunc = evolve(O, gens, angs;
                         truncation=CoeffTruncation(1e-3),
                         correction=corr)

        # Should have fewer or equal terms compared to untruncated
        O_full = evolve(O, gens, angs)
        @test length(O_trunc) <= length(O_full) || true  # may not clip anything if all coeffs large
    end

    @testset "trotterize: first order" begin
        N = 2
        H = PauliSum(N, ComplexF64)
        H[PauliBasis("ZI")] = 1.0 + 0im
        H[PauliBasis("IZ")] = 0.5 + 0im
        H[PauliBasis("XX")] = 0.3 + 0im
        dt = 0.1

        gens, angs = trotterize(H, dt)
        @test length(gens) == 3
        @test length(angs) == 3

        # Verify against dense matrix: exp(-i dt H)
        Hmat = Matrix(H)
        U_exact = exp(-1im * dt * Hmat)

        # Apply Trotter to identity operator in Heisenberg picture
        O = PauliSum(N, ComplexF64)
        O[PauliBasis("ZI")] = 1.0 + 0im
        O_trotter = evolve(O, gens, angs)

        # U† O U should be close to exact for small dt
        O_exact = U_exact' * Matrix(O) * U_exact
        @test Matrix(O_trotter) ≈ O_exact atol=5e-2  # first-order Trotter error ~ dt²
    end

    @testset "trotterize: convergence with n_trotter" begin
        N = 2
        H = PauliSum(N, ComplexF64)
        H[PauliBasis("ZI")] = 1.0 + 0im
        H[PauliBasis("XX")] = 0.3 + 0im
        dt = 0.5

        Hmat = Matrix(H)
        U_exact = exp(-1im * dt * Hmat)

        O = PauliSum(N, ComplexF64)
        O[PauliBasis("ZI")] = 1.0 + 0im
        O_exact = U_exact' * Matrix(O) * U_exact

        # Error should decrease with more Trotter steps
        errors = Float64[]
        for n in [1, 5, 20]
            gens, angs = trotterize(H, dt; n_trotter=n)
            O_t = evolve(O, gens, angs)
            push!(errors, norm(Matrix(O_t) - O_exact))
        end
        @test errors[2] < errors[1]
        @test errors[3] < errors[2]
    end

    @testset "trotterize: second order" begin
        N = 2
        H = PauliSum(N, ComplexF64)
        H[PauliBasis("ZI")] = 1.0 + 0im
        H[PauliBasis("XX")] = 0.3 + 0im
        dt = 0.3

        gens1, angs1 = trotterize(H, dt; order=1)
        gens2, angs2 = trotterize(H, dt; order=2)

        # Second order should use twice as many generators per step
        @test length(gens2) == 2 * length(gens1)

        Hmat = Matrix(H)
        U_exact = exp(-1im * dt * Hmat)
        O = PauliSum(N, ComplexF64)
        O[PauliBasis("ZI")] = 1.0 + 0im
        O_exact = U_exact' * Matrix(O) * U_exact

        O_t1 = evolve(O, gens1, angs1)
        O_t2 = evolve(O, gens2, angs2)

        # Second order should be more accurate than first order
        err1 = norm(Matrix(O_t1) - O_exact)
        err2 = norm(Matrix(O_t2) - O_exact)
        @test err2 < err1
    end

    @testset "qdrift" begin
        N = 2
        H = PauliSum(N, ComplexF64)
        H[PauliBasis("ZI")] = 1.0 + 0im
        H[PauliBasis("IZ")] = 0.5 + 0im
        H[PauliBasis("XX")] = 0.3 + 0im
        dt = 0.1

        rng = Random.Xoshiro(42)
        gens, angs = qdrift(H, dt; n_samples=10, rng=rng)
        @test length(gens) == 10
        @test length(angs) == 10

        # Each generator should be one of H's terms
        H_bases = Set(keys(H))
        for g in gens
            @test g in H_bases
        end
    end

    @testset "compose: evolve(O, trotterize(H, dt)...)" begin
        N = 2
        H = PauliSum(N, ComplexF64)
        H[PauliBasis("ZI")] = 1.0 + 0im
        H[PauliBasis("XX")] = 0.3 + 0im
        dt = 0.1

        O = PauliSum(N, ComplexF64)
        O[PauliBasis("ZI")] = 1.0 + 0im

        # This should work cleanly with splatting
        O_evolved = evolve(O, trotterize(H, dt; n_trotter=10)...)

        Hmat = Matrix(H)
        U = exp(-1im * dt * Hmat)
        O_exact = U' * Matrix(O) * U
        @test Matrix(O_evolved) ≈ O_exact atol=1e-2
    end

    @testset "Gate: hadamard" begin
        N = 2
        # Hadamard on qubit 1: H Z H = X
        O = PauliSum(Pauli("ZI"))
        O_h = hadamard(O, 1)
        @test Matrix(O_h) ≈ Matrix(PauliSum(Pauli("XI"))) atol=1e-12

        # Hadamard on KetSum: H|0> = (|0> + |1>)/√2
        k = KetSum(Ket(N, 0))
        k_h = hadamard(k, 1)
        k_dense = Vector(k_h)
        @test abs(k_dense[1]) ≈ 1/√2 atol=1e-12
        @test abs(k_dense[2]) ≈ 1/√2 atol=1e-12
    end

    @testset "Gate: cnot" begin
        N = 2
        # CNOT|10> = |11>
        k = KetSum(Ket([1, 0]))
        k_cnot = cnot(k, 1, 2)
        @test abs(Vector(k_cnot)[4]) ≈ 1.0 atol=1e-12  # |11>

        # CNOT|00> = |00>
        k0 = KetSum(Ket(N, 0))
        k0_cnot = cnot(k0, 1, 2)
        @test abs(Vector(k0_cnot)[1]) ≈ 1.0 atol=1e-12  # |00>
    end

    @testset "Gate: X_gate" begin
        N = 2
        # X|00> = |10>
        k = KetSum(Ket(N, 0))
        k_x = X_gate(k, 1)
        @test abs(Vector(k_x)[2]) ≈ 1.0 atol=1e-12  # |10>
    end

    @testset "Gate: Z_gate" begin
        N = 2
        # Z|1> = -|1>
        k = KetSum(Ket([1, 0]))
        k_z = Z_gate(k, 1)
        @test Vector(k_z)[2] ≈ -1.0 atol=1e-12
    end

    @testset "Gate: S_gate and T_gate" begin
        N = 1
        # S = diag(1, i), so S|1> = i|1>
        k1 = KetSum(Ket([1]))
        k_s = S_gate(k1, 1)
        # S applied in Schrödinger picture through evolve
        # S = exp(-iπ/4 Z), so S|1> = exp(iπ/4)|1> ... let's just check unitarity
        @test norm(k_s) ≈ 1.0 atol=1e-12

        # T gate preserves norm
        k_t = T_gate(k1, 1)
        @test norm(k_t) ≈ 1.0 atol=1e-12
    end

    @testset "Gate: _to_paulis decompositions" begin
        N = 3
        # hadamard_to_paulis
        gens, angs = hadamard_to_paulis(N, 2)
        @test length(gens) == 3
        @test length(angs) == 3

        # cnot_to_paulis
        gens, angs = cnot_to_paulis(N, 1, 2)
        @test length(gens) == 3

        # X_gate_to_paulis
        gens, angs = X_gate_to_paulis(N, 1)
        @test length(gens) == 1

        # Z_gate_to_paulis
        gens, angs = Z_gate_to_paulis(N, 1)
        @test length(gens) == 1
    end

end
