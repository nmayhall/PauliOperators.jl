using PauliOperators
using LinearAlgebra
using Test
using Random

# Heisenberg chain used across the sharded tests (distinct name from the
# binned test helper — both files load into the same test module).
function _sharded_heisenberg(N; Jx=1.0, Jy=0.9, Jz=1.1)
    H = PauliSum(N)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, X=[i, i+1]))] = Jx
        H[PauliBasis(Pauli(N, Y=[i, i+1]))] = Jy
        H[PauliBasis(Pauli(N, Z=[i, i+1]))] = Jz
    end
    return H
end

function _real_pauli_sum(N, n)
    O = PauliSum(N, Float64)
    while length(O) < n
        O[PauliBasis(rand(Pauli{N}))] = rand() - 0.5
    end
    return O
end

@testset "Sharded engine (serial)" begin
    Random.seed!(7)

    @testset "construction round trip, invariants 1+2" begin
        N = 8
        O = rand(PauliSum{N}, n_paulis=20)
        for r in 0:4
            A = rand(RankMap{N}, r)
            S = ShardedPauliSum(O, A; T=ComplexF64)
            @test check_sharding(S)
            @test length(S) == length(O)
            Og = PauliSum(S)
            @test length(Og) == length(O)
            @test all(Og[k] == O[k] for k in keys(O))
            @test norm(S) ≈ norm(O)
            @test norm(S, 1) ≈ norm(O, 1)
            @test norm(S, Inf) ≈ norm(O, Inf)
        end

        # identity term: lands in shard 0, tr preserved
        Oid = rand(PauliSum{N}, n_paulis=5)
        Oid[PauliBasis{N}(Int128(0), Int128(0))] = 2.5 + 0.0im
        S = ShardedPauliSum(Oid, rand(RankMap{N}, 3); T=ComplexF64)
        @test tr(S) ≈ tr(Oid)
        @test PauliSum(S)[PauliBasis{N}(Int128(0), Int128(0))] == 2.5 + 0.0im

        # real T rejects genuinely complex coefficients
        Oc = PauliSum(N)
        Oc[rand(PauliBasis{N})] = 1.0 + 0.5im
        @test_throws ErrorException ShardedPauliSum(Oc, rand(RankMap{N}, 2); T=Float64)

        # word type selection
        @test ShardedPauliSum(O, rand(RankMap{N}, 2); T=ComplexF64) isa
              ShardedPauliSum{N,UInt64,ComplexF64}
    end

    @testset "sin-branch sign formula: all 2-qubit (G, P) pairs bit-exact" begin
        θ = 0.3
        A = rand(RankMap{2}, 1)
        for gz in 0:3, gx in 0:3, pz in 0:3, px in 0:3
            (gz == 0 && gx == 0) && continue
            G = PauliBasis{2}(Int128(gz), Int128(gx))
            P = PauliBasis{2}(Int128(pz), Int128(px))
            O = PauliSum(2)
            O[P] = 0.7 - 0.4im
            Oref = evolve(O, [G], [θ])
            S = ShardedPauliSum(O, A; T=ComplexF64)
            circ = compile(A, [G], [θ]; version=S.version)
            evolve!(S, circ)
            Og = PauliSum(S)
            @test length(Og) == length(Oref)
            @test all(Og[k] == Oref[k] for k in keys(Oref))
        end
    end

    @testset "eager (window=1) bit-exact, windowed ≈, vs serial + binned" begin
        for N in (6, 8)
            O = rand(PauliSum{N}, n_paulis=10)
            gens, angs = trotterize(_sharded_heisenberg(N), 0.1, n_trotter=2, order=2)
            Oref = evolve(O, gens, angs)
            for r in (0, 2, 4), window in (1, 4, 1000)
                A = rand(RankMap{N}, r)
                S = ShardedPauliSum(O, A; T=ComplexF64)
                circ = compile(A, gens, angs; window)
                evolve!(S, circ)
                @test check_sharding(S)
                Og = PauliSum(S)
                if window == 1
                    @test length(Og) == length(Oref)
                    @test all(Og[k] == Oref[k] for k in keys(Oref))
                else
                    @test norm(Og - Oref) <= 1e-13 * norm(Oref)
                    # cross-oracle: Dict-based binned engine, same map and cadence
                    B = BinnedPauliSum(O, A)
                    evolve!(B, compile(B, gens, angs; window))
                    @test norm(Og - PauliSum(B)) <= 1e-13 * norm(Oref)
                end
            end
        end
    end

    @testset "T=Float64 real dynamics bit-exact vs serial" begin
        N = 6
        O = _real_pauli_sum(N, 10)
        gens, angs = trotterize(_sharded_heisenberg(N), 0.1, n_trotter=2, order=2)
        Oref = evolve(O, gens, angs)
        A = rand(RankMap{N}, 3)
        S = ShardedPauliSum(O, A)   # T defaults to Float64
        @test S isa ShardedPauliSum{N,UInt64,Float64}
        evolve!(S, compile(A, gens, angs; window=1))
        Og = PauliSum(S)
        @test length(Og) == length(Oref)
        @test all(Og[k] == Oref[k] for k in keys(Oref))
    end

    @testset "dense-matrix oracle (N=4)" begin
        N = 4
        O = rand(PauliSum{N}, n_paulis=6)
        gens, angs = trotterize(_sharded_heisenberg(N), 0.2, n_trotter=1, order=1)
        A = rand(RankMap{N}, 2)
        S = ShardedPauliSum(O, A; T=ComplexF64)
        evolve!(S, compile(A, gens, angs; window=3))
        U = Matrix(1.0I, 2^N, 2^N)
        for (G, θ) in zip(gens, angs)
            U = U * exp(-1im * θ / 2 * Matrix(G))
        end
        @test norm(Matrix(PauliSum(S)) - U' * Matrix(O) * U) < 1e-12
    end

    @testset "strict truncation at window=1 ≡ serial per-rotation truncation" begin
        N = 6
        O = rand(PauliSum{N}, n_paulis=8)
        gens, angs = trotterize(_sharded_heisenberg(N), 0.15, n_trotter=2, order=2)
        A = rand(RankMap{N}, 3)
        for strat in (CoeffTruncation(1e-3),
                      WeightTruncation(3),
                      XWeightTruncation(2),
                      MajoranaWeightTruncation(6),
                      WeightDampedTruncation(0.5, 1e-3),
                      XWeightDampedTruncation(0.4, 1e-3),
                      CompositeTruncation(CoeffTruncation(1e-3), WeightTruncation(4)))
            Oref = evolve(O, gens, angs; truncation=strat)
            S = ShardedPauliSum(O, A; T=ComplexF64)
            evolve!(S, compile(A, gens, angs; window=1); truncation=strat)
            Og = PauliSum(S)
            @test length(Og) == length(Oref)
            @test all(Og[k] == Oref[k] for k in keys(Oref))
        end
    end

    @testset "loose local truncation stays close to strict serial" begin
        N = 6
        O = rand(PauliSum{N}, n_paulis=8)
        gens, angs = trotterize(_sharded_heisenberg(N), 0.1, n_trotter=3, order=2)
        A = rand(RankMap{N}, 3)
        Oref = evolve(O, gens, angs; truncation=CoeffTruncation(1e-6))
        S = ShardedPauliSum(O, A; T=ComplexF64)
        evolve!(S, compile(A, gens, angs; window=8);
                truncation=CoeffTruncation(1e-6),
                local_truncation=CoeffTruncation(1e-8))
        @test norm(PauliSum(S) - Oref) <= 1e-4 * norm(Oref)
    end

    @testset "majorana weight on packed words matches PauliBasis version" begin
        for _ in 1:50
            p = rand(PauliBasis{8})
            @test PauliOperators._majorana_weight_bits(p.z % UInt64, p.x % UInt64) ==
                  majorana_weight(p)
            q = rand(PauliBasis{70})
            @test PauliOperators._majorana_weight_bits(q.z % UInt128, q.x % UInt128) ==
                  majorana_weight(q)
        end
    end

    @testset "N=70 (UInt128 words)" begin
        N = 70
        O = PauliSum(N)
        for _ in 1:6
            O[PauliBasis(rand(Pauli{N}))] = (rand() - 0.5) + 0.0im
        end
        G1 = PauliBasis(Pauli(N, X=[63, 64], Z=[65]))   # straddles the 64-bit line
        G2 = PauliBasis(Pauli(N, Y=[1, 70]))
        gens = [G1, G2, G1]
        angs = [0.3, 0.4, 0.2]
        Oref = evolve(O, gens, angs)
        A = rand(RankMap{N}, 2)
        S = ShardedPauliSum(O, A; T=ComplexF64)
        @test S isa ShardedPauliSum{N,UInt128,ComplexF64}
        evolve!(S, compile(A, gens, angs; window=1))
        @test check_sharding(S)
        Og = PauliSum(S)
        @test length(Og) == length(Oref)
        @test all(Og[k] == Oref[k] for k in keys(Oref))
    end

    @testset "capacity growth under pressure stays exact" begin
        N = 6
        O = rand(PauliSum{N}, n_paulis=10)
        gens, angs = trotterize(_sharded_heisenberg(N), 0.1, n_trotter=2, order=2)
        Oref = evolve(O, gens, angs)
        A = rand(RankMap{N}, 2)
        # deliberately tiny buffers: force early merges and boundary growth
        S = ShardedPauliSum(O, A; T=ComplexF64, capacity_factor=1.0, append_factor=0.1)
        cnt = WindowCounters(cld(length(gens), 16))
        evolve!(S, compile(A, gens, angs; window=16); counters=cnt)
        @test sum(cnt.early_merges) > 0
        Og = PauliSum(S)
        @test norm(Og - Oref) <= 1e-13 * norm(Oref)
    end

    @testset "unsupported options error loudly" begin
        N = 4
        O = rand(PauliSum{N}, n_paulis=4)
        A = rand(RankMap{N}, 1)
        gens, angs = trotterize(_sharded_heisenberg(N), 0.1, n_trotter=1)
        circ = compile(A, gens, angs)
        S() = ShardedPauliSum(O, A; T=ComplexF64)
        @test_throws ErrorException evolve!(S(), circ; truncation=StochasticCoeffTruncation(1e-3))
        @test_throws ErrorException evolve!(S(), circ; truncation=StochasticSamplingTruncation(10))
        # AdaptiveTruncation inside a composite cannot compile to one filter
        @test_throws ErrorException evolve!(S(), circ;
            truncation=CompositeTruncation(CoeffTruncation(1e-6), AdaptiveTruncation()))
        # version guard
        Sv = S()
        Sv.version += 1
        @test_throws ErrorException evolve!(Sv, circ)
    end
end
