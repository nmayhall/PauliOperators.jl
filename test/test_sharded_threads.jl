using PauliOperators
using LinearAlgebra
using Test
using Random

function _threads_heisenberg(N; Jx=1.0, Jy=0.9, Jz=1.1)
    H = PauliSum(N)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, X=[i, i+1]))] = Jx
        H[PauliBasis(Pauli(N, Y=[i, i+1]))] = Jy
        H[PauliBasis(Pauli(N, Z=[i, i+1]))] = Jz
    end
    return H
end

@testset "Sharded engine (threads)" begin
    Random.seed!(13)

    if Threads.nthreads() < 2
        @warn "Julia running with a single thread — start tests with --threads=4 " *
              "to exercise the multithreaded sharded engine"
    end
    maxt = min(4, Threads.nthreads())

    @testset "invariant 6: results independent of nthreads and r" begin
        N = 8
        O = rand(PauliSum{N}, n_paulis=12)
        gens, angs = trotterize(_threads_heisenberg(N), 0.1, n_trotter=2, order=2)
        Oref = evolve(O, gens, angs)
        for nt in unique((1, min(2, maxt), maxt)), r in (2, 4), window in (1, 5)
            A = rand(RankMap{N}, r)
            S = ShardedPauliSum(O, A; T=ComplexF64, nthreads=nt)
            evolve!(S, compile(A, gens, angs; window))
            @test check_sharding(S)
            Og = PauliSum(S)
            if window == 1
                # bit-exact at eager cadence for EVERY thread count
                @test length(Og) == length(Oref)
                @test all(Og[k] == Oref[k] for k in keys(Oref))
            else
                @test norm(Og - Oref) <= 1e-12 * norm(Oref)
            end
        end
    end

    @testset "invariant 3: protected generators never cross shards" begin
        N = 8
        # protect the ZZ family; drive a ZZ-only circuit
        zz = [PauliBasis(Pauli(N, Z=[i, i+1])) for i in 1:N-1]
        A = RankMap{N}(3; protected=zz)
        @test all(bin_shift(A, G) == 0 for G in zz)
        O = rand(PauliSum{N}, n_paulis=12)
        angs = fill(0.2, length(zz))
        Oref = evolve(O, zz, angs)
        nt = maxt
        S = ShardedPauliSum(O, A; T=ComplexF64, nthreads=nt)
        circ = @test_logs (:warn, r"never leave") compile(A, zz, angs; window=3)
        cnt = WindowCounters(cld(length(zz), 3))
        evolve!(S, circ; counters=cnt)
        @test sum(cnt.cross_appends) == 0
        @test sum(cnt.terms_created) > 0
        Og = PauliSum(S)
        @test norm(Og - Oref) <= 1e-13 * norm(Oref)
    end

    @testset "capacity pressure: early merges + growth stay exact" begin
        N = 8
        O = rand(PauliSum{N}, n_paulis=12)
        gens, angs = trotterize(_threads_heisenberg(N), 0.1, n_trotter=2, order=2)
        Oref = evolve(O, gens, angs)
        A = rand(RankMap{N}, 3)
        S = ShardedPauliSum(O, A; T=ComplexF64, nthreads=maxt,
                            capacity_factor=1.0, append_factor=0.1)
        cnt = WindowCounters(cld(length(gens), 16))
        evolve!(S, compile(A, gens, angs; window=16); counters=cnt)
        @test sum(cnt.early_merges) > 0
        @test norm(PauliSum(S) - Oref) <= 1e-12 * norm(Oref)
    end

    @testset "rebalancing keeps results invariant" begin
        N = 8
        O = rand(PauliSum{N}, n_paulis=12)
        gens, angs = trotterize(_threads_heisenberg(N), 0.1, n_trotter=2, order=2)
        Oref = evolve(O, gens, angs)
        A = rand(RankMap{N}, 4)
        S = ShardedPauliSum(O, A; T=ComplexF64, nthreads=maxt)
        evolve!(S, compile(A, gens, angs; window=4); rebalance_threshold=1.01)
        @test check_sharding(S)
        @test norm(PauliSum(S) - Oref) <= 1e-12 * norm(Oref)
        if maxt > 1
            @test length(unique(S.owner)) > 1   # ownership actually spread
        end
    end

    @testset "strict truncation at window=1 under threads ≡ serial" begin
        N = 8
        O = rand(PauliSum{N}, n_paulis=10)
        gens, angs = trotterize(_threads_heisenberg(N), 0.15, n_trotter=2, order=2)
        A = rand(RankMap{N}, 3)
        strat = CompositeTruncation(CoeffTruncation(1e-3), WeightTruncation(4))
        Oref = evolve(O, gens, angs; truncation=strat)
        S = ShardedPauliSum(O, A; T=ComplexF64, nthreads=maxt)
        evolve!(S, compile(A, gens, angs; window=1); truncation=strat)
        Og = PauliSum(S)
        @test length(Og) == length(Oref)
        @test all(Og[k] == Oref[k] for k in keys(Oref))
    end

    @testset "steady-state zero allocation under threads" begin
        N = 8
        O = PauliSum(N, Float64)
        while length(O) < 200
            O[PauliBasis(rand(Pauli{N}))] = rand() - 0.5
        end
        gens, angs = trotterize(_threads_heisenberg(N), 0.02, n_trotter=3, order=2)
        A = rand(RankMap{N}, 3)
        circ = compile(A, gens, angs; window=6)
        nw = length(circ.window_subgroups)
        strat = CompositeTruncation(WeightTruncation(2), CoeffTruncation(1e-9))
        S = ShardedPauliSum(O, A; T=Float64, nthreads=maxt,
                            capacity_factor=50.0, append_factor=2.0)
        for _ in 1:2
            evolve!(S, circ; truncation=strat, counters=WindowCounters(nw))
        end
        cnt = WindowCounters(nw)
        evolve!(S, circ; truncation=strat, counters=cnt)
        # window 1 includes the worker-pool spawn; steady-state windows must
        # be allocation-free even with all threads spinning on the barrier
        @test all(==(0), cnt.allocd[2:end])
        @test sum(cnt.early_merges) == 0
    end

    @testset "guards" begin
        N = 4
        O = rand(PauliSum{N}, n_paulis=4)
        A = rand(RankMap{N}, 1)
        gens, angs = trotterize(_threads_heisenberg(N), 0.1, n_trotter=1)
        S = ShardedPauliSum(O, A; T=ComplexF64, nthreads=Threads.nthreads() + 1)
        @test_throws ErrorException evolve!(S, compile(A, gens, angs))
        # pin_engine! without ThreadPinning loaded: informational no-op
        S1 = ShardedPauliSum(O, A; T=ComplexF64)
        @test_logs (:info, r"ThreadPinning") pin_engine!(S1)
    end
end
