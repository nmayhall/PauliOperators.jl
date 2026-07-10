using PauliOperators
using LinearAlgebra
using Test
using Random
using BenchmarkTools
using PauliOperators: _rotate_spv!, _merge_spv!, _sort_ws!, _gather_append!,
                      _compact_spv!, _pack, _compile_filter, NOFILTER,
                      merge_pending!, WindowCounters, check_spv

const _HAS_JET = try
    import JET
    true
catch
    false
end

# Zero allocation in the steady-state hot path is a correctness property of
# the SparsePauliVector engine (and the contract the future threaded engine
# inherits), not an optimization. These tests FAIL on any regression.

function _rand_spv(N, T; nterms=200, W=PauliOperators._word_type(N))
    O = PauliSum(N, T)
    while length(O) < nterms
        O[PauliBasis(rand(Pauli{N}))] = T <: Complex ? T(rand() - 0.5, rand() - 0.5) :
                                                       T(rand() - 0.5)
    end
    return SparsePauliVector(O; T=T, capacity_factor=50.0, append_factor=2.0)
end

reset_appends!(v) = (v.an = 0; v)

@testset "SparsePauliVector allocations" begin
    Random.seed!(11)
    N = 8

    @testset "kernels, T=$T" for T in (Float64, ComplexF64)
        v = _rand_spv(N, T)
        G = PauliBasis(rand(Pauli{N}))
        gz, gx = _pack(UInt64, G)
        ng = count_ones(gz & gx)

        # rotation kernel, no filter and with an active composite filter
        a = @ballocated _rotate_spv!($v, $gz, $gx, $ng, 0.99, 0.1, $NOFILTER) setup =
            (reset_appends!($v)) evals = 1
        @test a == 0
        f = _compile_filter(CompositeTruncation(WeightTruncation(6),
                                                WeightDampedTruncation(0.3, 1e-9)))
        a = @ballocated _rotate_spv!($v, $gz, $gx, $ng, 0.99, 0.1, $f) setup =
            (reset_appends!($v)) evals = 1
        @test a == 0
        reset_appends!(v)

        # sort + gather + merge + compact
        m = min(200, length(v.ws))
        for i in 1:m
            v.ws[i] = (rand(UInt64), rand(UInt64), T <: Complex ? T(rand(), rand()) : T(rand()))
        end
        @test (@ballocated _sort_ws!($(v.ws), 1, $m)) == 0
        @test (@ballocated _gather_append!($v)) == 0
        @test (@ballocated _merge_spv!($v, 0, $NOFILTER)) == 0
        @test (@ballocated _compact_spv!($v, $NOFILTER)) == 0
        @test (@ballocated merge_pending!($v)) == 0

        # single-rotation evolve! (rotate + sort + merge), post-warmup
        evolve!(v, G, 0.1)
        @test (@ballocated evolve!($v, $G, 0.1) evals = 1) == 0
    end

    @testset "reductions and in-place ops" begin
        v = _rand_spv(N, Float64)
        v2 = _rand_spv(N, Float64)
        ψ = rand(Ket{N})
        p = PauliBasis(rand(Pauli{N}))

        @test (@ballocated expectation_value($v, $ψ)) == 0
        @test (@ballocated inner_product($v, $v2)) == 0
        @test (@ballocated norm($v)) == 0
        @test (@ballocated norm($v, 1)) == 0
        @test (@ballocated norm($v, Inf)) == 0
        @test (@ballocated tr($v)) == 0
        @test (@ballocated mul!($v, 1.001)) == 0
        @test (@ballocated get($v, $p, 0.0)) == 0
        @test (@ballocated haskey($v, $p)) == 0
        @test (@ballocated ishermitian($v)) == 0

        # sum! is a straight merge once workspaces are warm
        sum!(v, v2)
        @test (@ballocated sum!($v, $v2) evals = 1) == 0

        # clips (compiled filters)
        @test (@ballocated coeff_clip!($v, 1e-30) evals = 1) == 0
        @test (@ballocated weight_clip!($v, 8) evals = 1) == 0
        @test (@ballocated x_weight_clip!($v, 8) evals = 1) == 0
        @test (@ballocated majorana_weight_clip!($v, 16) evals = 1) == 0
        @test (@ballocated weight_damped_clip!($v, 0.1, 1e-30) evals = 1) == 0
        rng = Xoshiro(4)
        @test (@ballocated stochastic_clip!($v, 1e-30; rng=$rng) evals = 1) == 0

        # truncate! with a compiled strategy and no correction
        strat = CompositeTruncation(WeightTruncation(8), CoeffTruncation(1e-30))
        @test (@ballocated truncate!($v, $strat) evals = 1) == 0

        # channels
        @test (@ballocated depolarizing_channel!($v, 0.001) evals = 1) == 0
        @test (@ballocated dephasing_channel!($v, 0.001) evals = 1) == 0
        @test (@ballocated pauli_channel!($v, 0.001, 0.001, 0.001) evals = 1) == 0
    end

    @testset "commutator! into a presized output" begin
        A = _rand_spv(6, ComplexF64; nterms=40)
        B = _rand_spv(6, ComplexF64; nterms=40)
        out = SparsePauliVector(6, ComplexF64; capacity=40 * 40)
        commutator!(out, A, B)   # warm-up
        @test (@ballocated commutator!($out, $A, $B) evals = 1) == 0
        anticommutator!(out, A, B)
        @test (@ballocated anticommutator!($out, $A, $B) evals = 1) == 0
    end

    @testset "wide words (N=70, UInt128)" begin
        v = _rand_spv(70, Float64; nterms=100)
        G = PauliBasis(rand(Pauli{70}))
        gz, gx = _pack(UInt128, G)
        ng = count_ones(gz & gx)
        a = @ballocated _rotate_spv!($v, $gz, $gx, $ng, 0.99, 0.1, $NOFILTER) setup =
            (reset_appends!($v)) evals = 1
        @test a == 0
        reset_appends!(v)
        evolve!(v, G, 0.1)
        @test (@ballocated evolve!($v, $G, 0.1) evals = 1) == 0
        ψ = rand(Ket{70})
        @test (@ballocated expectation_value($v, $ψ)) == 0
    end

    @testset "windowed driver steady state (gc_num per window)" begin
        H = PauliSum(N)
        for i in 1:N-1
            H[PauliBasis(Pauli(N, X=[i, i+1]))] = 1.0
            H[PauliBasis(Pauli(N, Z=[i, i+1]))] = 1.1
        end
        gens, angs = trotterize(H, 0.02, n_trotter=3, order=2)
        window = 6
        nw = cld(length(gens), window)
        # weight cutoff bounds the population, so capacities provably
        # plateau: steady state must show zero growth and no early merges
        strat = CompositeTruncation(WeightTruncation(2), CoeffTruncation(1e-9))
        O = PauliSum(N, Float64)
        while length(O) < 200
            O[PauliBasis(rand(Pauli{N}))] = rand() - 0.5
        end
        Sd = SparsePauliVector(O; T=Float64, capacity_factor=50.0, append_factor=2.0)
        for _ in 1:2                                             # warm-up to plateau
            evolve!(Sd, gens, angs; window=window, truncation=strat,
                    counters=WindowCounters(nw))
        end
        cnt = WindowCounters(nw)
        evolve!(Sd, gens, angs; window=window, truncation=strat, counters=cnt)
        @test all(==(0), cnt.allocd)
        @test sum(cnt.early_merges) == 0
        @test check_spv(Sd)

        # same contract at window=1 (the PauliSum-parity default)
        nw1 = length(gens)
        for _ in 1:2
            evolve!(Sd, gens, angs; window=1, truncation=strat,
                    counters=WindowCounters(nw1))
        end
        cnt = WindowCounters(nw1)
        evolve!(Sd, gens, angs; window=1, truncation=strat, counters=cnt)
        @test all(==(0), cnt.allocd)
    end

    @testset "JET type stability of kernels" begin
        if _HAS_JET
            # @eval defers JET.@test_opt macro expansion to runtime, so this
            # file loads even when JET is absent.
            @eval begin
                v = _rand_spv($N, Float64)
                G = PauliBasis(rand(Pauli{$N}))
                gz, gx = _pack(UInt64, G)
                ng = count_ones(gz & gx)
                reset_appends!(v)
                JET.@test_opt _rotate_spv!(v, gz, gx, ng, 0.99, 0.1, NOFILTER)
                reset_appends!(v)
                JET.@test_opt _merge_spv!(v, 0, NOFILTER)
                JET.@test_opt _gather_append!(v)
                JET.@test_opt _compact_spv!(v, NOFILTER)
                JET.@test_opt expectation_value(v, rand(Ket{$N}))
                JET.@test_opt inner_product(v, v)
                JET.@test_opt evolve!(v, G, 0.1)
            end
        else
            @warn "JET not available; skipping type-stability checks"
            @test_skip false
        end
    end
end
