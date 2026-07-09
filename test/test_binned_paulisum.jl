using PauliOperators
using LinearAlgebra
using Test
using Random

@testset "BinnedPauliSum" begin
    Random.seed!(3)

    @testset "round-trip and invariants" begin
        for N in (6, 70)
            O = rand(PauliSum{N}, n_paulis=50)
            for r in (1, 3, 5)
                A = rand(RankMap{N}, r)
                B = BinnedPauliSum(O, A)
                @test check_binning(B)
                @test length(B) == length(O)
                @test PauliSum(B) == O
                @test sum(bin_histogram(B)) == length(O)
            end
        end
    end

    @testset "reductions match flat equivalents" begin
        N = 8
        O1 = rand(PauliSum{N}, n_paulis=40)
        O2 = rand(PauliSum{N}, n_paulis=40)
        # ensure some overlap for a nontrivial inner product
        for (p, c) in Iterators.take(O1, 10)
            O2[p] = rand(ComplexF64)
        end
        A = rand(RankMap{N}, 4)
        B1 = BinnedPauliSum(O1, A)
        B2 = BinnedPauliSum(O2, A)
        @test norm(B1) ≈ norm(O1)
        @test norm(B1, 1) ≈ norm(O1, 1)
        @test norm(B1, Inf) ≈ norm(O1, Inf)
        @test inner_product(B1, B2) ≈ inner_product(O1, O2)
        @test tr(B1) ≈ tr(O1)
        ψ = rand(Ket{N})
        @test expectation_value(B1, ψ) ≈ expectation_value(O1, ψ)
    end

    @testset "version guard on inner_product" begin
        N = 4
        O = rand(PauliSum{N}, n_paulis=5)
        A = rand(RankMap{N}, 2)
        B1 = BinnedPauliSum(O, A)
        B2 = BinnedPauliSum(O, A)
        B2.version += 1
        @test_throws ErrorException inner_product(B1, B2)
    end

    @testset "truncate! matches flat truncation" begin
        N = 8
        A = rand(RankMap{N}, 3)
        for strategy in (CoeffTruncation(0.3),
                         WeightTruncation(3),
                         XWeightTruncation(2),
                         WeightDampedTruncation(0.5, 0.2),
                         CompositeTruncation(CoeffTruncation(0.2), WeightTruncation(4)),
                         AdaptiveTruncation(max_terms=10, min_thresh=1e-12))
            O = rand(PauliSum{N}, n_paulis=60)
            Oref = deepcopy(O)
            B = BinnedPauliSum(O, A)
            truncate!(B, strategy)
            truncate!(Oref, strategy)
            @test PauliSum(B) == Oref
            @test check_binning(B)
        end
        # global sampling truncation is explicitly unsupported
        O = rand(PauliSum{N}, n_paulis=20)
        B = BinnedPauliSum(O, A)
        @test_throws ErrorException truncate!(B, StochasticSamplingTruncation(5))
    end

    @testset "correction accumulators" begin
        N = 6
        O = rand(PauliSum{N}, n_paulis=40)
        A = rand(RankMap{N}, 3)
        ψ = rand(Ket{N})

        corr_flat = EnergyCorrection(ψ)
        corr_binned = EnergyCorrection(ψ)
        Oref = deepcopy(O)
        B = BinnedPauliSum(O, A)
        truncate!(Oref, CoeffTruncation(0.3), corr_flat)
        truncate!(B, CoeffTruncation(0.3), corr_binned)
        @test corr_binned.accumulated_energy ≈ corr_flat.accumulated_energy
    end

    @testset "rebin! after replacing the map" begin
        N = 8
        O = rand(PauliSum{N}, n_paulis=30)
        B = BinnedPauliSum(O, rand(RankMap{N}, 3))
        B.A = rand(RankMap{N}, 4)
        B.bin_owner = default_bin_owner(nbins(B.A), 1)
        B.version += 1
        rebin!(B)
        @test check_binning(B)
        @test PauliSum(B) == O
    end

    @testset "swap_row!" begin
        N = 8
        O = rand(PauliSum{N}, n_paulis=30)
        B = BinnedPauliSum(O, rand(RankMap{N}, 4))
        v0 = B.version
        newrow = RankRow(rand(Int128) & ((Int128(1) << N) - 1),
                         rand(Int128) & ((Int128(1) << N) - 1))
        swap_row!(B, 2, newrow)
        @test B.version == v0 + 1
        @test B.A.rows[2] == newrow
        @test check_binning(B)
        @test PauliSum(B) == O
        # stale circuit is rejected after the swap
        gens, angs = trotterize(rand(PauliSum{N}, n_paulis=4), 0.1)
        circ_old = compile(rand(RankMap{N}, 4), gens, angs, version=v0)
        @test_throws ErrorException evolve!(B, circ_old)
        @test_throws ErrorException swap_row!(B, 9, newrow)
    end

    @testset "bin ownership table" begin
        @test default_bin_owner(8, 2) == Int32[0, 0, 0, 0, 1, 1, 1, 1]
        @test default_bin_owner(4, 4) == Int32[0, 1, 2, 3]
        @test_throws ErrorException default_bin_owner(8, 3)
        N = 4
        B = BinnedPauliSum(rand(PauliSum{N}, n_paulis=5), rand(RankMap{N}, 3), nranks=2)
        @test collect(owned_bins(B, 0)) == [0, 1, 2, 3]
        @test collect(owned_bins(B, 1)) == [4, 5, 6, 7]
    end
end
