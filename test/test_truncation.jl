using PauliOperators
using LinearAlgebra
using Test
using Random

@testset "Phase 2: Truncation Strategy System" begin

    @testset "NoTruncation" begin
        N = 4
        ps = rand(PauliSum{N}; n_paulis=10)
        ps_orig = deepcopy(ps)
        truncate!(ps, NoTruncation())
        @test ps == ps_orig
    end

    @testset "CoeffTruncation" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im
        ps[PauliBasis("XIII")] = 0.5 + 0im
        ps[PauliBasis("IXII")] = 1e-4 + 0im
        ps[PauliBasis("IIXI")] = 1e-8 + 0im

        truncate!(ps, CoeffTruncation(1e-3))
        @test length(ps) == 2
        @test haskey(ps, PauliBasis("IIII"))
        @test haskey(ps, PauliBasis("XIII"))
    end

    @testset "WeightTruncation" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im   # weight 0
        ps[PauliBasis("XIII")] = 0.5 + 0im    # weight 1
        ps[PauliBasis("XXII")] = 0.3 + 0im    # weight 2
        ps[PauliBasis("XXXI")] = 0.2 + 0im    # weight 3
        ps[PauliBasis("XXXX")] = 0.1 + 0im    # weight 4

        truncate!(ps, WeightTruncation(2))
        @test length(ps) == 3
        @test haskey(ps, PauliBasis("IIII"))
        @test haskey(ps, PauliBasis("XIII"))
        @test haskey(ps, PauliBasis("XXII"))
    end

    @testset "MajoranaWeightTruncation" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im  # majorana weight 0
        ps[PauliBasis("XIII")] = 0.5 + 0im
        ps[PauliBasis("XXXX")] = 0.1 + 0im

        mw_identity = majorana_weight(PauliBasis("IIII"))
        @test mw_identity == 0

        ps_clipped = deepcopy(ps)
        truncate!(ps_clipped, MajoranaWeightTruncation(0))
        @test length(ps_clipped) == 1
        @test haskey(ps_clipped, PauliBasis("IIII"))
    end

    @testset "StochasticCoeffTruncation" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        # Add many small terms, then set the large term last so it can't be overwritten
        for i in 1:20
            p = rand(PauliBasis{N})
            ps[p] = 0.001 * randn() + 0im
        end
        ps[PauliBasis("IIII")] = 1.0 + 0im

        n_before = length(ps)
        truncate!(ps, StochasticCoeffTruncation(0.01, MersenneTwister(123)))
        # Should have fewer terms (some small terms deleted stochastically)
        @test length(ps) <= n_before
        # Large term should survive
        @test haskey(ps, PauliBasis("IIII"))
    end

    @testset "StochasticSamplingTruncation" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        for i in 1:20
            p = rand(PauliBasis{N})
            ps[p] = randn() + 0im
        end

        truncate!(ps, StochasticSamplingTruncation(5, MersenneTwister(123)))
        @test length(ps) == 5
    end

    @testset "AdaptiveTruncation" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        for i in 1:50
            p = rand(PauliBasis{N})
            ps[p] = randn() + 0im
        end

        n_before = length(ps)
        truncate!(ps, AdaptiveTruncation(max_terms=10, min_thresh=1e-12))
        @test length(ps) <= 10
    end

    @testset "CompositeTruncation" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im   # weight 0, large coeff
        ps[PauliBasis("XIII")] = 0.5 + 0im    # weight 1, large coeff
        ps[PauliBasis("XXXI")] = 0.3 + 0im    # weight 3, large coeff
        ps[PauliBasis("IIXI")] = 1e-8 + 0im   # weight 1, tiny coeff

        # Composite: first clip coefficients, then clip weight
        strat = CompositeTruncation(CoeffTruncation(1e-3), WeightTruncation(2))
        truncate!(ps, strat)
        @test length(ps) == 2
        @test haskey(ps, PauliBasis("IIII"))
        @test haskey(ps, PauliBasis("XIII"))

        # Verify composite matches sequential application
        ps2 = PauliSum(N, ComplexF64)
        ps2[PauliBasis("IIII")] = 1.0 + 0im
        ps2[PauliBasis("XIII")] = 0.5 + 0im
        ps2[PauliBasis("XXXI")] = 0.3 + 0im
        ps2[PauliBasis("IIXI")] = 1e-8 + 0im

        coeff_clip!(ps2, 1e-3)
        weight_clip!(ps2, 2)
        @test ps == ps2
    end

    @testset "NoCorrection" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im
        ps[PauliBasis("XIII")] = 1e-8 + 0im

        # Should work without error and not track anything
        truncate!(ps, CoeffTruncation(1e-3), NoCorrection())
        @test length(ps) == 1
    end

    @testset "EnergyCorrection" begin
        N = 4
        ψ = Ket(N, 0)
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im
        ps[PauliBasis("ZIII")] = 0.5 + 0im
        ps[PauliBasis("XIII")] = 0.01 + 0im  # small off-diagonal term

        e_before = real(expectation_value(ps, ψ))
        corr = EnergyCorrection(ψ)
        truncate!(ps, CoeffTruncation(0.1), corr)
        e_after = real(expectation_value(ps, ψ))

        # The accumulated energy should track the difference
        @test corr.accumulated_energy ≈ e_after - e_before
    end

    @testset "EnergyVarianceCorrection" begin
        N = 4
        ψ = Ket(N, 0)
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im
        ps[PauliBasis("ZIII")] = 0.5 + 0im
        ps[PauliBasis("XIII")] = 0.3 + 0im
        ps[PauliBasis("IIXI")] = 0.01 + 0im  # small term to be clipped

        e_before = real(expectation_value(ps, ψ))
        v_before = variance(ps, ψ)
        corr = EnergyVarianceCorrection(ψ)
        truncate!(ps, CoeffTruncation(0.1), corr)
        e_after = real(expectation_value(ps, ψ))
        v_after = variance(ps, ψ)

        @test corr.accumulated_energy ≈ e_after - e_before
        @test corr.accumulated_variance ≈ v_after - v_before
    end

    @testset "Multiple truncations accumulate" begin
        N = 4
        ψ = Ket(N, 0)
        ps = PauliSum(N, ComplexF64)
        for i in 1:30
            p = rand(PauliBasis{N})
            ps[p] = randn() + 0im
        end

        e_initial = real(expectation_value(ps, ψ))
        corr = EnergyCorrection(ψ)

        # Apply multiple truncations
        truncate!(ps, CoeffTruncation(0.1), corr)
        truncate!(ps, WeightTruncation(2), corr)

        e_final = real(expectation_value(ps, ψ))

        # Total accumulated should equal total difference
        @test corr.accumulated_energy ≈ e_final - e_initial
    end

    @testset "CoeffTruncationMF" begin
        N = 4
        ψ = Ket(N, 0)  # |0000⟩, ⟨ψ|Z_i|ψ⟩ = +1
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im
        ps[PauliBasis("ZIII")] = 5e-4 + 0im   # diagonal, |c| ≤ thresh — redirects (+1·c)
        ps[PauliBasis("ZZII")] = 1e-4 + 0im   # diagonal, |c| ≤ thresh — redirects (+1·c)
        ps[PauliBasis("XIII")] = 1e-4 + 0im   # off-diagonal, redirects to 0 (just dropped)
        ps[PauliBasis("YYII")] = 0.5 + 0im    # large coeff — kept

        e_before = expectation_value(ps, ψ)
        truncate!(ps, CoeffTruncationMF(1e-3, ψ))
        e_after = expectation_value(ps, ψ)

        # ⟨ψ|O|ψ⟩ preserved exactly
        @test e_before ≈ e_after
        # The two diagonal small terms got dropped from non-identity slots
        @test !haskey(ps, PauliBasis("ZIII"))
        @test !haskey(ps, PauliBasis("ZZII"))
        @test !haskey(ps, PauliBasis("XIII"))
        # Large term still present
        @test haskey(ps, PauliBasis("YYII"))
        # Identity coeff has absorbed the diagonal contributions
        @test ps[PauliBasis("IIII")] ≈ 1.0 + 5e-4 + 1e-4
    end

    @testset "WeightTruncationMF" begin
        N = 4
        ψ = Ket(N, 0)  # |0000⟩
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im   # weight 0 — kept
        ps[PauliBasis("ZIII")] = 0.5 + 0im   # weight 1 — kept
        ps[PauliBasis("ZZZI")] = 0.3 + 0im   # weight 3, diagonal → redirects (+1·c)
        ps[PauliBasis("XXXI")] = 0.2 + 0im   # weight 3, off-diagonal → redirects to 0
        ps[PauliBasis("ZZZZ")] = 0.1 + 0im   # weight 4, diagonal → redirects (+1·c)

        e_before = expectation_value(ps, ψ)
        truncate!(ps, WeightTruncationMF(2, ψ))
        e_after = expectation_value(ps, ψ)

        # ⟨ψ|O|ψ⟩ preserved exactly
        @test e_before ≈ e_after
        # High-weight terms removed
        @test !haskey(ps, PauliBasis("ZZZI"))
        @test !haskey(ps, PauliBasis("XXXI"))
        @test !haskey(ps, PauliBasis("ZZZZ"))
        # Low-weight terms kept
        @test haskey(ps, PauliBasis("ZIII"))
        # Identity has absorbed diagonal high-weight contributions (+0.3 + 0.1)
        @test ps[PauliBasis("IIII")] ≈ 1.0 + 0.3 + 0.1
    end

    @testset "MF redirects with non-trivial reference" begin
        # Néel-style reference: |0101⟩ gives ⟨Z_i⟩ = +1,-1,+1,-1
        N = 4
        ψ = Ket(N, Int128(0b1010))  # bits 2, 4 set ⇒ sites 2,4 are |1⟩
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("ZZII")] = 0.4 + 0im   # ⟨Z₁Z₂⟩ = (+1)(-1) = -1, weight 2
        ps[PauliBasis("ZIZI")] = 0.2 + 0im   # ⟨Z₁Z₃⟩ = (+1)(+1) = +1, weight 2
        ps[PauliBasis("XIII")] = 0.5 + 0im   # off-diagonal, weight 1 — kept
        e_before = expectation_value(ps, ψ)
        truncate!(ps, WeightTruncationMF(1, ψ))
        e_after = expectation_value(ps, ψ)
        @test e_before ≈ e_after
        # Identity should absorb 0.4·(-1) + 0.2·(+1) = -0.2
        id = PauliBasis(repeat("I", N))
        @test ps[id] ≈ -0.2
    end

    @testset "Default convenience constructors" begin
        @test CoeffTruncation().thresh == 1e-6
        @test StochasticCoeffTruncation(0.1).epsilon == 0.1
        @test StochasticSamplingTruncation(5).n_keep == 5
        @test AdaptiveTruncation(max_terms=100, min_thresh=1e-6).max_terms == 100
        @test CompositeTruncation(NoTruncation(), CoeffTruncation(0.1)).strategies isa Vector{TruncationStrategy}
    end

end
