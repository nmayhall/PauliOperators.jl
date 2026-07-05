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

    @testset "XWeightTruncation" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im   # x-weight 0
        ps[PauliBasis("ZZZZ")] = 0.9 + 0im    # x-weight 0 (diagonal)
        ps[PauliBasis("XIII")] = 0.5 + 0im    # x-weight 1
        ps[PauliBasis("YZII")] = 0.4 + 0im    # x-weight 1 (Y counts, Z does not)
        ps[PauliBasis("XYII")] = 0.3 + 0im    # x-weight 2
        ps[PauliBasis("XYZX")] = 0.1 + 0im    # x-weight 3

        @test x_weight(PauliBasis("IIII")) == 0
        @test x_weight(PauliBasis("ZZZZ")) == 0
        @test x_weight(PauliBasis("YZII")) == 1
        @test x_weight(PauliBasis("XYZX")) == 3

        truncate!(ps, XWeightTruncation(1))
        @test length(ps) == 4
        @test haskey(ps, PauliBasis("IIII"))
        @test haskey(ps, PauliBasis("ZZZZ"))
        @test haskey(ps, PauliBasis("XIII"))
        @test haskey(ps, PauliBasis("YZII"))
    end

    @testset "XWeightDampedTruncation" begin
        N = 4

        # alpha = 0 reduces exactly to CoeffTruncation
        Random.seed!(2)
        ps = rand(PauliSum{N}; n_paulis=30)
        ps_damped = deepcopy(ps)
        ps_coeff = deepcopy(ps)
        truncate!(ps_damped, XWeightDampedTruncation(0.0, 0.3))
        truncate!(ps_coeff, CoeffTruncation(0.3))
        @test ps_damped == ps_coeff

        # Criterion check: remove iff |c|·exp(-alpha·x_weight) <= thresh
        alpha, thresh = 0.5, 0.05
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("ZZZZ")] = 1.0 + 0im    # xw=0: 1.0 > 0.05, keep
        ps[PauliBasis("XIII")] = 0.01 + 0im   # xw=1: 0.01·e^-0.5 ≈ 0.006, drop
        ps[PauliBasis("XYII")] = 0.5 + 0im    # xw=2: 0.5·e^-1 ≈ 0.184, keep
        ps[PauliBasis("XYXI")] = 0.1 + 0im    # xw=3: 0.1·e^-1.5 ≈ 0.022, drop
        ps[PauliBasis("XYXY")] = 1.0 + 0im    # xw=4: e^-2 ≈ 0.135, keep

        truncate!(ps, XWeightDampedTruncation(alpha, thresh))
        @test length(ps) == 3
        @test haskey(ps, PauliBasis("ZZZZ"))
        @test haskey(ps, PauliBasis("XYII"))
        @test haskey(ps, PauliBasis("XYXY"))
        @test ps[PauliBasis("XYXY")] == 1.0 + 0im

        # Monotonicity: larger alpha keeps a subset of the terms
        Random.seed!(3)
        ps = rand(PauliSum{N}; n_paulis=30)
        kept = map((0.0, 0.5, 2.0)) do a
            Set(keys(truncate!(deepcopy(ps), XWeightDampedTruncation(a, 0.1))))
        end
        @test issubset(kept[3], kept[2])
        @test issubset(kept[2], kept[1])

        # Composes with other strategies and matches sequential application
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("ZZZZ")] = 1.0 + 0im
        ps[PauliBasis("XYII")] = 0.5 + 0im
        ps[PauliBasis("XYXI")] = 0.4 + 0im
        ps[PauliBasis("XIII")] = 1e-8 + 0im
        ps2 = deepcopy(ps)

        truncate!(ps, CompositeTruncation(XWeightDampedTruncation(0.5, 0.05), XWeightTruncation(2)))
        x_weight_damped_clip!(ps2, 0.5, 0.05)
        x_weight_clip!(ps2, 2)
        @test ps == ps2

        # Default convenience constructor
        @test XWeightDampedTruncation(0.5).thresh == 1e-6
        @test XWeightDampedTruncation(0.5).alpha == 0.5
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

    @testset "WeightDampedTruncation" begin
        N = 4

        # alpha = 0 reduces exactly to CoeffTruncation
        Random.seed!(2)
        ps = rand(PauliSum{N}; n_paulis=30)
        ps_damped = deepcopy(ps)
        ps_coeff = deepcopy(ps)
        truncate!(ps_damped, WeightDampedTruncation(0.0, 0.3))
        truncate!(ps_coeff, CoeffTruncation(0.3))
        @test ps_damped == ps_coeff

        # Criterion check: remove iff |c|·exp(-alpha·w) <= thresh
        alpha, thresh = 0.5, 0.05
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im    # w=0: 1.0 > 0.05, keep
        ps[PauliBasis("XIII")] = 0.01 + 0im   # w=1: 0.01·e^-0.5 ≈ 0.006, drop
        ps[PauliBasis("XXII")] = 0.5 + 0im    # w=2: 0.5·e^-1 ≈ 0.184, keep
        ps[PauliBasis("XXXI")] = 0.1 + 0im    # w=3: 0.1·e^-1.5 ≈ 0.022, drop
        ps[PauliBasis("XXXX")] = 1.0 + 0im    # w=4: e^-2 ≈ 0.135, keep

        truncate!(ps, WeightDampedTruncation(alpha, thresh))
        @test length(ps) == 3
        @test haskey(ps, PauliBasis("IIII"))
        @test haskey(ps, PauliBasis("XXII"))
        @test haskey(ps, PauliBasis("XXXX"))
        # Kept coefficients are unmodified
        @test ps[PauliBasis("XXXX")] == 1.0 + 0im

        # Monotonicity: larger alpha keeps a subset of the terms
        Random.seed!(3)
        ps = rand(PauliSum{N}; n_paulis=30)
        kept = map((0.0, 0.5, 2.0)) do a
            Set(keys(truncate!(deepcopy(ps), WeightDampedTruncation(a, 0.1))))
        end
        @test issubset(kept[3], kept[2])
        @test issubset(kept[2], kept[1])

        # Composes with other strategies and matches sequential application
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im
        ps[PauliBasis("XXII")] = 0.5 + 0im
        ps[PauliBasis("XXXI")] = 0.4 + 0im
        ps[PauliBasis("XIII")] = 1e-8 + 0im
        ps2 = deepcopy(ps)

        truncate!(ps, CompositeTruncation(WeightDampedTruncation(0.5, 0.05), WeightTruncation(2)))
        weight_damped_clip!(ps2, 0.5, 0.05)
        weight_clip!(ps2, 2)
        @test ps == ps2

        # Works with correction accumulators
        ψ = Ket(N, 0)
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im
        ps[PauliBasis("ZZZZ")] = 0.1 + 0im
        e_before = real(expectation_value(ps, ψ))
        corr = EnergyCorrection(ψ)
        truncate!(ps, WeightDampedTruncation(1.0, 0.05), corr)
        e_after = real(expectation_value(ps, ψ))
        @test corr.accumulated_energy ≈ e_after - e_before

        # Default convenience constructor
        @test WeightDampedTruncation(0.5).thresh == 1e-6
        @test WeightDampedTruncation(0.5).alpha == 0.5
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

    @testset "Default convenience constructors" begin
        @test CoeffTruncation().thresh == 1e-6
        @test StochasticCoeffTruncation(0.1).epsilon == 0.1
        @test StochasticSamplingTruncation(5).n_keep == 5
        @test AdaptiveTruncation(max_terms=100, min_thresh=1e-6).max_terms == 100
        @test CompositeTruncation(NoTruncation(), CoeffTruncation(0.1)).strategies isa Tuple{NoTruncation, CoeffTruncation}
        # Vector convenience constructor still works (converts to tuple internally)
        @test CompositeTruncation([NoTruncation(), CoeffTruncation(0.1)]).strategies isa Tuple
    end

end
