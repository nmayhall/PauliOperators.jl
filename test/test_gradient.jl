using PauliOperators
using LinearAlgebra
using Test
using Random

# Random Hermitian PauliSum: real coefficients on random basis elements.
function _rand_herm(N; n_paulis=8)
    H = PauliSum(N)
    while length(H) < n_paulis
        H[rand(PauliBasis{N})] = randn() + 0im
    end
    return H
end

@testset "Rotation-circuit gradients" begin

    @testset "Finite-difference match" begin
        Random.seed!(2)
        for N in 4:6
            M = 10
            H = _rand_herm(N)
            gens = [rand(PauliBasis{N}) for _ in 1:M]
            θ = 0.7 * randn(M)              # generic angles
            ψ = rand(Ket{N})

            c, g = expectation_value_gradient(H, gens, θ, ψ)
            @test g isa Vector{Float64}

            ε = 1e-5
            for i in 1:M
                θp = copy(θ); θp[i] += ε
                θm = copy(θ); θm[i] -= ε
                cp = real(expectation_value(H, gens, θp, ψ))
                cm = real(expectation_value(H, gens, θm, ψ))
                g_fd = (cp - cm) / (2ε)
                @test isapprox(g[i], g_fd; atol=1e-8, rtol=1e-6)
            end
        end
    end

    @testset "Sweep invariant" begin
        Random.seed!(3)
        N = 5
        M = 8
        H = _rand_herm(N)
        gens = [rand(PauliBasis{N}) for _ in 1:M]
        θ = 0.7 * randn(M)
        ψ = rand(Ket{N})

        # forward pass, build ρ̃, then check Σ a_P r_P is constant along the
        # reverse sweep (it equals C at every step, by unitarity)
        A = evolve(H, gens, θ)
        ρ = PauliSum(N, ComplexF64)
        for (p, _) in A
            p.x == 0 || continue
            ρ[p] = expectation_value(p, ψ)
        end
        C = real(PauliOperators._pair(A, ρ))
        @test C ≈ real(expectation_value(A, ψ))
        for i in M:-1:1
            evolve!(A, gens[i], -θ[i])
            evolve!(ρ, gens[i], -θ[i])
            @test real(PauliOperators._pair(A, ρ)) ≈ C
        end
    end

    @testset "Cost consistency" begin
        Random.seed!(4)
        N = 4
        M = 6
        H = _rand_herm(N)
        gens = [rand(PauliBasis{N}) for _ in 1:M]
        θ = 0.7 * randn(M)
        ψ = rand(Ket{N})

        c_ref = real(expectation_value(evolve(H, gens, θ), ψ))
        @test real(expectation_value(H, gens, θ, ψ)) ≈ c_ref
        c, _ = expectation_value_gradient(H, gens, θ, ψ)
        @test c ≈ c_ref
    end

    @testset "Commutator cross-check" begin
        Random.seed!(5)
        N = 4
        M = 6
        H = _rand_herm(N)
        gens = [rand(PauliBasis{N}) for _ in 1:M]
        θ = 0.7 * randn(M)
        ψ = rand(Ket{N})

        _, g = expectation_value_gradient(H, gens, θ, ψ)
        # independent route: ∂C/∂θ_i = (i/2)⟨ψ| V_i' [G_i, A_i] V_i |ψ⟩
        for i in (1, M ÷ 2, M)
            A_i = evolve(H, gens[1:i], θ[1:i])
            D = commutator(PauliSum(gens[i]), A_i)
            D_M = evolve(D, gens[i+1:end], θ[i+1:end])
            @test g[i] ≈ real((1im / 2) * expectation_value(D_M, ψ))
        end
    end

    @testset "ρ::PauliSum method" begin
        Random.seed!(6)
        N = 3
        M = 5
        H = _rand_herm(N; n_paulis=5)
        gens = [rand(PauliBasis{N}) for _ in 1:M]
        θ = 0.7 * randn(M)
        ψ = rand(Ket{N})

        # ρ for |ψ⟩⟨ψ| as tr(Pρ) values over all diagonal Paulis
        ρ = PauliSum(N, ComplexF64)
        for z in 0:(2^N - 1)
            p = PauliBasis{N}(Int128(z), Int128(0))
            ρ[p] = expectation_value(p, ψ)
        end
        c1, g1 = expectation_value_gradient(H, gens, θ, ψ)
        c2, g2 = expectation_value_gradient(H, gens, θ, ρ)
        @test c1 ≈ c2
        @test g1 ≈ g2
    end

    @testset "Exact at cancellation points ($method)" for method in (:dynamic, :static)
        # The projection tracks the reachable Pauli set, not the coefficient
        # support, so exact zeros (θ = 0, π/2, interference) must not break it.

        # π/2: cos θ = 0 deletes the parent from the support.
        # C(θ) = ⟨0|e^{iθ/2 Y} Z e^{-iθ/2 Y}|0⟩ = cos θ, dC/dθ = -sin θ
        N = 1
        H = PauliSum(N)
        H[PauliBasis(Pauli(N, Z=[1]))] = 1.0 + 0im
        gens = [PauliBasis(Pauli(N, Y=[1]))]
        ψ = Ket(N, 0)
        _, g = expectation_value_gradient(H, gens, [π / 2], ψ; method=method)
        @test isapprox(g[1], -1.0; atol=1e-12)

        # interference: images of X and Z cancel the X component at θ = π/4;
        # with ρ = |+⟩⟨+| (tr(Xρ)=1) the cancelled slot carries the gradient:
        # C(θ) = cos θ - sin θ, dC/dθ = -(sin θ + cos θ) = -√2 at π/4
        H2 = PauliSum(N)
        H2[PauliBasis(Pauli(N, X=[1]))] = 1.0 + 0im
        H2[PauliBasis(Pauli(N, Z=[1]))] = 1.0 + 0im
        ρ = PauliSum(N, ComplexF64)
        ρ[PauliBasis(Pauli(N, X=[1]))] = 1.0
        _, g = expectation_value_gradient(H2, gens, [π / 4], ρ; method=method)
        @test isapprox(g[1], -sqrt(2); atol=1e-12)

        # exact zeros and π/2 sprinkled through a multi-step circuit
        # (sin θ = 0 creates zero-coefficient partner keys; cos θ = 0 zeroes
        # parents — both must survive the reverse sweep's bookkeeping)
        Random.seed!(11)
        N2 = 4
        M = 8
        H3 = _rand_herm(N2)
        gens2 = [rand(PauliBasis{N2}) for _ in 1:M]
        θ = 0.7 * randn(M)
        θ[2] = 0.0
        θ[4] = π / 2
        θ[7] = 0.0
        ψ2 = rand(Ket{N2})
        _, g = expectation_value_gradient(H3, gens2, θ, ψ2; method=method)
        ε = 1e-5
        for i in 1:M
            θp = copy(θ); θp[i] += ε
            θm = copy(θ); θm[i] -= ε
            g_fd = (real(expectation_value(H3, gens2, θp, ψ2)) -
                    real(expectation_value(H3, gens2, θm, ψ2))) / (2ε)
            @test isapprox(g[i], g_fd; atol=1e-8, rtol=1e-6)
        end
    end

    @testset "Method equivalence and monotone truncation" begin
        # :dynamic and :static are identical without truncation
        Random.seed!(12)
        N = 5
        M = 10
        H = _rand_herm(N)
        gens = [rand(PauliBasis{N}) for _ in 1:M]
        θ = 0.7 * randn(M)
        ψ = rand(Ket{N})
        c1, g1 = expectation_value_gradient(H, gens, θ, ψ; method=:dynamic)
        c2, g2 = expectation_value_gradient(H, gens, θ, ψ; method=:static)
        @test c1 ≈ c2
        @test isapprox(g1, g2; atol=1e-12)

        # both engines truncate monotonically (only terms not already in the sum
        # are clipped), so at the π/4 interference point the cancelled X
        # slot survives — both terms already exist — and the gradient stays
        # exact even under truncation
        N1 = 1
        H2 = PauliSum(N1)
        H2[PauliBasis(Pauli(N1, X=[1]))] = 1.0 + 0im
        H2[PauliBasis(Pauli(N1, Z=[1]))] = 1.0 + 0im
        gens1 = [PauliBasis(Pauli(N1, Y=[1]))]
        ρ = PauliSum(N1, ComplexF64)
        ρ[PauliBasis(Pauli(N1, X=[1]))] = 1.0
        trunc = CoeffTruncation(1e-10)
        for method in (:dynamic, :static)
            _, gt = expectation_value_gradient(H2, gens1, [π / 4], ρ;
                                               truncation=trunc, method=method)
            @test isapprox(gt[1], -sqrt(2); atol=1e-12)
        end

        # with a mild threshold, the truncated gradient tracks the
        # untruncated one closely on a generic circuit, for both engines,
        # and the engines agree with each other to O(threshold)
        gm2 = Vector{Float64}[]
        for method in (:dynamic, :static)
            _, gt = expectation_value_gradient(H, gens, θ, ψ;
                                               truncation=CoeffTruncation(1e-10),
                                               method=method)
            @test isapprox(gt, g1; atol=1e-6)
            push!(gm2, gt)
        end
        @test isapprox(gm2[1], gm2[2]; atol=1e-6)

        @test_throws ArgumentError expectation_value_gradient(H, gens, θ, ψ; method=:bogus)
    end

    @testset "Truncated FD consistency (threshold → 0)" begin
        # The gradient differentiates the monotone-truncated cost (the
        # `monotone=true` variant of `expectation_value`); the FD mismatch
        # must be first order in the truncation threshold.
        Random.seed!(13)
        N = 5
        M = 10
        H = _rand_herm(N)
        gens = [rand(PauliBasis{N}) for _ in 1:M]
        θ = 0.7 * randn(M)
        ψ = rand(Ket{N})
        ε = 1e-5

        function fd_mismatch(thresh, method)
            trunc = CoeffTruncation(thresh)
            _, g = expectation_value_gradient(H, gens, θ, ψ;
                                              truncation=trunc, method=method)
            worst = 0.0
            for i in 1:M
                θp = copy(θ); θp[i] += ε
                θm = copy(θ); θm[i] -= ε
                cp = real(expectation_value(H, gens, θp, ψ;
                                            truncation=trunc, monotone=true))
                cm = real(expectation_value(H, gens, θm, ψ;
                                            truncation=trunc, monotone=true))
                worst = max(worst, abs(g[i] - (cp - cm) / (2ε)))
            end
            return worst
        end

        for method in (:dynamic, :static)
            # tight threshold: FD match to (near) stencil precision
            @test fd_mismatch(1e-12, method) < 1e-8
            # looser thresholds: error stays O(threshold)
            @test fd_mismatch(1e-8, method) < 1e-5
        end
    end

    @testset "Error paths" begin
        N = 3
        H = _rand_herm(N; n_paulis=3)
        gens = [rand(PauliBasis{N}) for _ in 1:4]
        ψ = Ket(N, 0)
        @test_throws DimensionMismatch expectation_value_gradient(H, gens, randn(3), ψ)
        @test_throws DimensionMismatch expectation_value(H, gens, randn(3), ψ)
    end
end
