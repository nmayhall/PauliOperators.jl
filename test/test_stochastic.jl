using PauliOperators
using Test
using Random
using LinearAlgebra

@testset "stochastic_clip!" begin

    @testset "large coefficients preserved" begin
        N = 4
        ps = PauliSum(N, Float64)
        p1 = PauliBasis(Pauli(N, Z=[1]))
        p2 = PauliBasis(Pauli(N, X=[2]))
        ps[p1] = 1.0
        ps[p2] = 0.5

        ε = 0.1
        ps_copy = deepcopy(ps)
        stochastic_clip!(ps_copy, ε)

        @test ps_copy[p1] == 1.0
        @test ps_copy[p2] == 0.5
        @test length(ps_copy) == 2
    end

    @testset "promoted terms have correct magnitude" begin
        N = 4
        ε = 0.1

        rng = Random.Xoshiro(42)
        for _ in 1:200
            ps = PauliSum(N, Float64)
            p1 = PauliBasis(Pauli(N, Z=[1]))
            ps[p1] = 0.01  # well below ε

            ps_copy = deepcopy(ps)
            stochastic_clip!(ps_copy, ε; rng=rng)

            if haskey(ps_copy, p1)
                # Promoted: magnitude should be ε
                @test abs(ps_copy[p1]) ≈ ε
            end
            # Otherwise it was killed, which is fine
        end
    end

    @testset "unbiasedness" begin
        N = 4
        ε = 0.05

        # Create a PauliSum with known small coefficients
        ps_orig = PauliSum(N, Float64)
        p1 = PauliBasis(Pauli(N, Z=[1]))
        p2 = PauliBasis(Pauli(N, X=[2]))
        p3 = PauliBasis(Pauli(N, Z=[1], X=[2]))
        ps_orig[p1] = 0.03   # below ε
        ps_orig[p2] = 0.01   # below ε
        ps_orig[p3] = 0.1    # above ε

        n_trials = 50000
        avg = Dict{PauliBasis{N}, Float64}()
        for basis in keys(ps_orig)
            avg[basis] = 0.0
        end

        rng = Random.Xoshiro(123)
        for _ in 1:n_trials
            ps_copy = deepcopy(ps_orig)
            stochastic_clip!(ps_copy, ε; rng=rng)
            for (basis, c) in ps_copy
                avg[basis] = get(avg, basis, 0.0) + c / n_trials
            end
        end

        # Each average should be close to the original
        for (basis, c_orig) in ps_orig
            c_avg = avg[basis]
            # Tolerance: ~5 sigma. For the coin flip, variance per trial is at most
            # ε * |c_orig|, so std of mean ≈ sqrt(ε * |c_orig| / n_trials)
            tol = 5 * sqrt(ε * abs(c_orig) / n_trials) + 1e-12
            @test abs(c_avg - c_orig) < tol
        end
    end

    @testset "unbiasedness with complex coefficients" begin
        N = 3
        ε = 0.05

        ps_orig = PauliSum(N, ComplexF64)
        p1 = PauliBasis(Pauli(N, Z=[1]))
        p2 = PauliBasis(Pauli(N, X=[2]))
        ps_orig[p1] = 0.02 + 0.01im
        ps_orig[p2] = 0.03 - 0.02im

        n_trials = 50000
        avg = Dict{PauliBasis{N}, ComplexF64}()
        for basis in keys(ps_orig)
            avg[basis] = 0.0 + 0.0im
        end

        rng = Random.Xoshiro(456)
        for _ in 1:n_trials
            ps_copy = deepcopy(ps_orig)
            stochastic_clip!(ps_copy, ε; rng=rng)
            for (basis, c) in ps_copy
                avg[basis] = get(avg, basis, ComplexF64(0)) + c / n_trials
            end
        end

        for (basis, c_orig) in ps_orig
            c_avg = avg[basis]
            tol = 5 * sqrt(ε * abs(c_orig) / n_trials) + 1e-12
            @test abs(c_avg - c_orig) < tol
        end
    end

    @testset "determinism with fixed RNG" begin
        N = 4
        ps = rand(PauliSum{N}; n_paulis=50)
        mul!(ps, 0.01)  # make coefficients small
        ε = 0.005

        rng1 = Random.Xoshiro(42)
        ps1 = deepcopy(ps)
        stochastic_clip!(ps1, ε; rng=rng1)

        rng2 = Random.Xoshiro(42)
        ps2 = deepcopy(ps)
        stochastic_clip!(ps2, ε; rng=rng2)

        @test ps1 == ps2
    end

    @testset "zero coefficients are deleted" begin
        N = 3
        ps = PauliSum(N, Float64)
        p1 = PauliBasis(Pauli(N, Z=[1]))
        ps[p1] = 0.0

        stochastic_clip!(ps, 0.1)
        @test !haskey(ps, p1)
    end
end

@testset "evolve" begin
    @testset "correctness against dense matrix" begin
        N = 3
        # Create a simple operator
        O = PauliSum(N, ComplexF64)
        O[PauliBasis(Pauli(N, Z=[1]))] = 1.0

        # Create a generator
        G = PauliBasis(Pauli(N, X=[1]))
        θ = 0.3

        # Evolve using our function
        O_evolved = evolve(O, G, θ)

        # Evolve using dense matrices
        U = exp(Matrix(-1im * θ/2 * Matrix(Pauli(N, X=[1]))))
        O_dense = U' * Matrix(O) * U

        @test norm(Matrix(O_evolved) - O_dense) < 1e-12
    end

    @testset "commuting terms unchanged" begin
        N = 3
        O = PauliSum(N, ComplexF64)
        O[PauliBasis(Pauli(N, Z=[1]))] = 1.0

        # Z1 commutes with Z2
        G = PauliBasis(Pauli(N, Z=[2]))
        O_evolved = evolve(O, G, 0.5)

        @test length(O_evolved) == 1
        @test abs(O_evolved[PauliBasis(Pauli(N, Z=[1]))] - 1.0) < 1e-14
    end

    @testset "evolve! matches evolve" begin
        N = 4
        O = rand(PauliSum{N}; n_paulis=10)
        G = PauliBasis(Pauli(N, X=[2], Z=[3]))
        θ = 0.7

        O1 = evolve(O, G, θ)

        O2 = deepcopy(O)
        evolve!(O2, G, θ)

        for (basis, c) in O1
            @test abs(c - get(O2, basis, 0.0)) < 1e-13
        end
        for (basis, c) in O2
            @test abs(c - get(O1, basis, 0.0)) < 1e-13
        end
    end
end

@testset "clip functions" begin
    @testset "coeff_clip!" begin
        N = 3
        ps = PauliSum(N, Float64)
        ps[PauliBasis(Pauli(N, Z=[1]))] = 1.0
        ps[PauliBasis(Pauli(N, X=[2]))] = 1e-18

        coeff_clip!(ps; thresh=1e-16)
        @test length(ps) == 1
        @test haskey(ps, PauliBasis(Pauli(N, Z=[1])))
    end

    @testset "weight_clip!" begin
        N = 4
        ps = PauliSum(N, Float64)
        ps[PauliBasis(Pauli(N, Z=[1]))] = 1.0                    # weight 1
        ps[PauliBasis(Pauli(N, Z=[1], X=[2]))] = 0.5             # weight 2
        ps[PauliBasis(Pauli(N, Z=[1], X=[2,3]))] = 0.3           # weight 3

        weight_clip!(ps, 2)
        @test length(ps) == 2
        @test !haskey(ps, PauliBasis(Pauli(N, Z=[1], X=[2,3])))
    end

    @testset "weight function" begin
        N = 4
        @test weight(PauliBasis(Pauli(N, Z=[1]))) == 1
        @test weight(PauliBasis(Pauli(N, Z=[1], X=[2]))) == 2
        @test weight(PauliBasis(Pauli(N, Y=[1]))) == 1   # Y on same qubit
        @test weight(PauliBasis{N}(Int128(0), Int128(0))) == 0   # identity
    end
end

@testset "stochastic_propagate" begin
    @testset "convergence against exact evolution" begin
        N = 3

        # Simple observable: Z1
        O = PauliSum(N, ComplexF64)
        O[PauliBasis(Pauli(N, Z=[1]))] = 1.0

        # A few rotation gates
        generators = [
            PauliBasis(Pauli(N, X=[1])),
            PauliBasis(Pauli(N, Z=[2], X=[1])),
            PauliBasis(Pauli(N, X=[2])),
        ]
        angles = [0.1, 0.2, 0.15]

        ψ = Ket{N}(0)

        # Exact evolution (no truncation)
        O_exact = deepcopy(O)
        for (gi, θi) in zip(generators, angles)
            O_exact = evolve(O_exact, gi, θi)
        end
        e_exact = real(expectation_value(O_exact, ψ))

        # Stochastic evolution with small threshold (should be very accurate)
        result = stochastic_propagate(O, generators, angles, ψ, 1e-6;
                                      n_samples=200, seed=42, verbose=0)

        # Mean should agree with exact within a few standard errors
        @test abs(result.mean - e_exact) < max(5 * result.stderr, 1e-10)
        @test length(result.samples) == 200
    end

    @testset "reduces to deterministic for large ε=Inf" begin
        N = 3
        O = PauliSum(N, ComplexF64)
        O[PauliBasis(Pauli(N, Z=[1]))] = 1.0

        generators = [PauliBasis(Pauli(N, X=[1]))]
        angles = [0.3]
        ψ = Ket{N}(0)

        # With ε=Inf, no term is ever below threshold => deterministic
        result = stochastic_propagate(O, generators, angles, ψ, Inf;
                                      n_samples=10, seed=1, verbose=0)

        # All samples should be identical
        @test all(s ≈ result.samples[1] for s in result.samples)
    end
end
