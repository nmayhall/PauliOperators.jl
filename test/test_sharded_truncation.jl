using PauliOperators
using LinearAlgebra
using Test
using Random

function _trunc_heisenberg(N; Jx=1.0, Jy=0.9, Jz=1.1)
    H = PauliSum(N)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, X=[i, i+1]))] = Jx
        H[PauliBasis(Pauli(N, Y=[i, i+1]))] = Jy
        H[PauliBasis(Pauli(N, Z=[i, i+1]))] = Jz
    end
    return H
end

@testset "Sharded truncation, corrections, adaptive thresholds" begin
    Random.seed!(17)
    maxt = min(4, Threads.nthreads())

    @testset "expectation_value and inner_product match Dict paths" begin
        N = 8
        O = rand(PauliSum{N}, n_paulis=20)
        A = rand(RankMap{N}, 3)
        S = ShardedPauliSum(O, A; T=ComplexF64)
        for _ in 1:5
            ψ = Ket(N, rand(0:2^N-1))
            @test expectation_value(S, ψ) ≈ expectation_value(O, ψ)
        end
        O2 = rand(PauliSum{N}, n_paulis=20)
        for k in collect(keys(O))[1:5]
            O2[k] = rand() - 0.5 + 0.0im   # force key overlap
        end
        S2 = ShardedPauliSum(O2, A; T=ComplexF64)
        @test inner_product(S, S2) ≈ inner_product(O, O2)
        @test_throws ErrorException inner_product(S,
            ShardedPauliSum(O2, rand(RankMap{N}, 3); T=ComplexF64))
    end

    @testset "truncate!(S, strategy) ≡ truncate!(PauliSum, strategy)" begin
        N = 8
        O = rand(PauliSum{N}, n_paulis=40)
        A = rand(RankMap{N}, 3)
        for strat in (CoeffTruncation(0.3), WeightTruncation(3),
                      WeightDampedTruncation(0.5, 0.2),
                      CompositeTruncation(CoeffTruncation(0.2), XWeightTruncation(2)))
            Oref = deepcopy(O)
            truncate!(Oref, strat)
            S = ShardedPauliSum(O, A; T=ComplexF64)
            truncate!(S, strat)
            @test check_sharding(S)
            Og = PauliSum(S)
            @test length(Og) == length(Oref)
            @test all(Og[k] == Oref[k] for k in keys(Oref))
        end
    end

    @testset "adaptive truncation keeps count near budget" begin
        N = 8
        O = rand(PauliSum{N}, n_paulis=30)
        A = rand(RankMap{N}, 3)
        gens, angs = trotterize(_trunc_heisenberg(N), 0.1, n_trotter=3, order=2)
        for nt in unique((1, maxt))
            S = ShardedPauliSum(O, A; T=ComplexF64, nthreads=nt)
            evolve!(S, compile(A, gens, angs; window=4);
                    truncation=AdaptiveTruncation(max_terms=500, min_thresh=1e-12))
            @test check_sharding(S)
            # bin quantization + one-window lag: allow the factor-of-2 band
            @test length(S) <= 2 * 500
            @test length(S) > 50            # did not collapse to nothing
        end
        # cold-path _apply! parity: threshold within one bin of serial top-k
        Sbig = ShardedPauliSum(rand(PauliSum{N}, n_paulis=200), A; T=ComplexF64)
        truncate!(Sbig, AdaptiveTruncation(max_terms=50, min_thresh=1e-12))
        @test length(Sbig) <= 50 * 2 && length(Sbig) >= 25
    end

    @testset "EnergyCorrection matches serial at window=1" begin
        N = 6
        O = rand(PauliSum{N}, n_paulis=10)
        gens, angs = trotterize(_trunc_heisenberg(N), 0.2, n_trotter=2, order=2)
        A = rand(RankMap{N}, 2)
        ψ = Ket(N, 5)
        strat = CoeffTruncation(2e-2)

        corr_ref = EnergyCorrection(ψ)
        Oref = evolve(O, gens, angs; truncation=strat, correction=corr_ref)

        for nt in unique((1, maxt))
            corr = EnergyCorrection(ψ)
            S = ShardedPauliSum(O, A; T=ComplexF64, nthreads=nt)
            evolve!(S, compile(A, gens, angs; window=1);
                    truncation=strat, correction=corr)
            # window=1 evolution is bit-exact, so the accumulated energy
            # correction must match to reduction-order precision
            @test corr.accumulated_energy ≈ corr_ref.accumulated_energy atol=1e-10
            # final states agree, so measured energies do too
            @test real(expectation_value(S, ψ)) ≈
                  real(expectation_value(Oref, ψ)) atol=1e-10
        end
    end

    @testset "EnergyCorrection matches the binned oracle at windowed cadence" begin
        N = 6
        O = rand(PauliSum{N}, n_paulis=10)
        gens, angs = trotterize(_trunc_heisenberg(N), 0.2, n_trotter=2, order=2)
        A = rand(RankMap{N}, 2)
        ψ = Ket(N, 3)
        strat = CoeffTruncation(2e-2)
        for window in (2, 8)
            corr_b = EnergyCorrection(ψ)
            B = BinnedPauliSum(O, A)
            evolve!(B, compile(B, gens, angs; window);
                    truncation=strat, correction=corr_b)
            for nt in unique((1, maxt))
                corr = EnergyCorrection(ψ)
                # generous capacity: an early merge would add truncation
                # events the binned oracle doesn't have
                S = ShardedPauliSum(O, A; T=ComplexF64, nthreads=nt,
                                    capacity_factor=200.0, append_factor=8.0)
                cnt = WindowCounters(cld(length(gens), window))
                evolve!(S, compile(A, gens, angs; window);
                        truncation=strat, correction=corr, counters=cnt)
                @test sum(cnt.early_merges) == 0
                # same map, same cadence, same truncation events — the Dict
                # engine is the oracle for the windowed correction total
                @test corr.accumulated_energy ≈ corr_b.accumulated_energy atol=1e-9
                @test norm(PauliSum(S) - PauliSum(B)) <= 1e-10 * norm(PauliSum(B))
            end
        end
    end

    @testset "cadence error vs window length (quantified, not assumed)" begin
        N = 8
        O = rand(PauliSum{N}, n_paulis=10)
        gens, angs = trotterize(_trunc_heisenberg(N), 0.1, n_trotter=3, order=2)
        A = rand(RankMap{N}, 3)
        strat = CoeffTruncation(1e-5)
        Oref = evolve(O, gens, angs; truncation=strat)
        errs = Float64[]
        for M in (1, 2, 4, 8, 16)
            S = ShardedPauliSum(O, A; T=ComplexF64)
            evolve!(S, compile(A, gens, angs; window=M); truncation=strat)
            push!(errs, norm(PauliSum(S) - Oref) / norm(Oref))
        end
        @test errs[1] == 0.0                    # eager cadence is exact
        @test all(errs .< 1e-3)                 # deferred merging stays bounded
    end

    @testset "unsupported corrections error loudly" begin
        N = 4
        O = rand(PauliSum{N}, n_paulis=4)
        A = rand(RankMap{N}, 1)
        S = ShardedPauliSum(O, A; T=ComplexF64)
        @test_throws ErrorException truncate!(S, CoeffTruncation(1e-3),
                                              EnergyVarianceCorrection(Ket(N, 0)))
        @test_throws ErrorException truncate!(S, StochasticCoeffTruncation(1e-3))
    end
end
