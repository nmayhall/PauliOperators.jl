using PauliOperators
using LinearAlgebra
using Test
using Random
using BenchmarkTools
using PauliOperators: _rotate_shard!, _merge_shard!, _sort_ws!, _gather_append!,
                      NOFILTER, _pack, _compile_filter

const _HAS_JET = try
    import JET
    true
catch
    false
end

# Zero allocation in the steady-state hot path is a correctness property of
# the sharded engine (GC pauses serialize all threads), not an optimization.
# These tests FAIL on any regression.
@testset "Sharded engine allocations" begin
    Random.seed!(11)
    N = 8
    O = PauliSum(N, Float64)
    while length(O) < 200
        O[PauliBasis(rand(Pauli{N}))] = rand() - 0.5
    end
    H = PauliSum(N)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, X=[i, i+1]))] = 1.0
        H[PauliBasis(Pauli(N, Z=[i, i+1]))] = 1.1
    end
    gens, angs = trotterize(H, 0.02, n_trotter=3, order=2)
    A = rand(RankMap{N}, 3)
    S = ShardedPauliSum(O, A; T=Float64, capacity_factor=50.0, append_factor=2.0)

    reset_cursors!(S) = for t in 1:S.nthreads, j in 1:length(S.shards)
        S.cur[t][j] = S.shards[j].seg_lo[t]
        S.mark[t][j] = S.shards[j].seg_lo[t]
    end

    gz, gx = _pack(UInt64, gens[1])
    ng = count_ones(gz & gx)

    @testset "rotation kernel" begin
        a = @ballocated _rotate_shard!($S, 1, 1, 1, $gz, $gx, $ng, 0.99, 0.1, $NOFILTER) setup =
            ($reset_cursors!($S)) evals = 1
        @test a == 0
        reset_cursors!(S)
        # with an active (weight + damped coeff) filter
        f = _compile_filter(CompositeTruncation(WeightTruncation(6),
                                                WeightDampedTruncation(0.3, 1e-9)))
        a = @ballocated _rotate_shard!($S, 1, 1, 1, $gz, $gx, $ng, 0.99, 0.1, $f) setup =
            ($reset_cursors!($S)) evals = 1
        @test a == 0
        reset_cursors!(S)
    end

    @testset "sort + gather + merge kernels" begin
        sh = S.shards[1]
        m = min(200, length(sh.ws))
        for i in 1:m
            sh.ws[i] = (rand(UInt64), rand(UInt64), rand() - 0.5)
        end
        @test (@ballocated _sort_ws!($(sh.ws), 1, $m)) == 0
        @test (@ballocated _gather_append!($sh, $(S.cur), 1, 1)) == 0
        @test (@ballocated _merge_shard!($sh, 0, $NOFILTER)) == 0
    end

    @testset "windowed driver steady state (gc_num per window)" begin
        circ = compile(A, gens, angs; window=6)
        nw = length(circ.window_subgroups)
        # weight cutoff bounds the population (≤ 3^2·binomial(8,2)+3·8+1 terms),
        # so capacities provably plateau: steady state must show zero growth
        # and no capacity-forced early merges
        strat = CompositeTruncation(WeightTruncation(2), CoeffTruncation(1e-9))
        Sd = ShardedPauliSum(O, A; T=Float64, capacity_factor=50.0, append_factor=2.0)
        for _ in 1:2                                               # warm-up to plateau
            evolve!(Sd, circ; truncation=strat, counters=WindowCounters(nw))
        end
        cnt = WindowCounters(nw)
        evolve!(Sd, circ; truncation=strat, counters=cnt)
        @test all(==(0), cnt.allocd)
        @test sum(cnt.early_merges) == 0
    end

    @testset "JET type stability of kernels" begin
        if _HAS_JET
            reset_cursors!(S)
            JET.@test_opt _rotate_shard!(S, 1, 1, 1, gz, gx, ng, 0.99, 0.1, NOFILTER)
            reset_cursors!(S)
            JET.@test_opt _merge_shard!(S.shards[1], 0, NOFILTER)
            JET.@test_opt _gather_append!(S.shards[1], S.cur, 1, 1)
        else
            @warn "JET not available; skipping type-stability checks"
            @test_skip false
        end
    end
end
