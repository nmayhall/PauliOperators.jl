# Distributed propagation tests. Run under MPI with exactly 2 ranks:
#
#     mpiexecjl -n 2 julia --project test/mpi/runtests.jl
#
# (launched automatically by the main test suite unless
# PAULIOPERATORS_TEST_MPI=false). Every rank seeds the same RNG, so the
# replicated inputs (operator, rank maps, circuits) are identical everywhere.

using MPI
using PauliOperators
using LinearAlgebra
using Test
using Random

MPI.Init()
const comm = MPI.COMM_WORLD
const me = MPI.Comm_rank(comm)

MPI.Comm_size(comm) == 2 || error("this test script requires exactly 2 MPI ranks")

function _heisenberg_chain(N; Jx=1.0, Jy=0.9, Jz=1.1)
    H = PauliSum(N)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, X=[i, i+1]))] = Jx
        H[PauliBasis(Pauli(N, Y=[i, i+1]))] = Jy
        H[PauliBasis(Pauli(N, Z=[i, i+1]))] = Jz
    end
    return H
end

@testset "MPI 2-rank distributed propagation" begin

    N = 8
    H = _heisenberg_chain(N)
    gens, angs = trotterize(H, 0.1, n_trotter=2, order=2)
    L = length(gens)

    @testset "exactness vs serial (no truncation)" begin
        Random.seed!(42)
        O = rand(PauliSum{N}, n_paulis=10)
        Oref = evolve(O, gens, angs)
        for r in (1, 4), window in (1, 5, 1000)
            A = rand(RankMap{N}, r)
            D = DistributedPauliSum(O, A, comm)
            circ = compile(D, gens, angs; window)
            counters = PropagationCounters()
            evolve!(D, circ; counters)

            @test check_ownership(D)
            @test check_binning(D.localpart)
            @test counters.rotations == L
            @test counters.merges == cld(L, window)

            # allreduce reductions against serial values
            @test norm(D) ≈ norm(Oref)
            @test norm(D, 1) ≈ norm(Oref, 1)
            @test length(D) == length(Oref)
            @test tr(D) ≈ tr(Oref)
            ψ = Ket(N, 0)
            @test expectation_value(D, ψ) ≈ expectation_value(Oref, ψ)
            @test sum(bin_histogram(D)) == length(Oref)

            # full-state comparison on rank 0 (duplicates co-located: the
            # gathered sum has as many terms as the ranks hold combined)
            G = gather(D)
            if me == 0
                @test length(G) == length(Oref)
                @test isapprox(G, Oref, atol=1e-10)
            else
                @test G === nothing
            end
        end
    end

    @testset "inner product across ranks" begin
        Random.seed!(7)
        O1 = rand(PauliSum{N}, n_paulis=30)
        O2 = rand(PauliSum{N}, n_paulis=30)
        for (p, c) in Iterators.take(O1, 10)
            O2[p] = rand(ComplexF64)
        end
        A = rand(RankMap{N}, 3)
        D1 = DistributedPauliSum(O1, A, comm)
        D2 = DistributedPauliSum(O2, A, comm)
        @test inner_product(D1, D2) ≈ inner_product(O1, O2)
    end

    @testset "windowed truncation cadence" begin
        Random.seed!(11)
        O = PauliSum(N)
        O[PauliBasis(Pauli(N, Z=[1]))] = 1.0 + 0im
        gens2, angs2 = trotterize(H, 0.05, n_trotter=5, order=2)
        strict = CoeffTruncation(1e-4)
        loose = CoeffTruncation(1e-5)
        Oref = evolve(O, gens2, angs2, truncation=strict)
        A = rand(RankMap{N}, 3)

        # eager: identical to serial per-rotation truncation
        D1 = DistributedPauliSum(O, A, comm)
        evolve!(D1, compile(D1, gens2, angs2, window=1),
                truncation=strict, local_truncation=loose)
        G1 = gather(D1)
        me == 0 && @test isapprox(G1, Oref, atol=1e-10)

        # windowed: bounded cadence error
        D8 = DistributedPauliSum(O, A, comm)
        evolve!(D8, compile(D8, gens2, angs2, window=8),
                truncation=strict, local_truncation=loose)
        @test check_ownership(D8)
        G8 = gather(D8)
        me == 0 && @test norm(G8 - Oref) / norm(Oref) < 1e-2
    end

    @testset "correction accumulators ride the merge collectives" begin
        Random.seed!(13)
        O = rand(PauliSum{N}, n_paulis=20)
        ψ = rand(Ket{N})
        strict = CoeffTruncation(1e-2)

        corr_serial = EnergyCorrection(ψ)
        Oref = evolve(O, gens, angs, truncation=strict, correction=corr_serial)

        A = rand(RankMap{N}, 3)
        D = DistributedPauliSum(O, A, comm)
        corr_dist = EnergyCorrection(ψ)
        evolve!(D, compile(D, gens, angs, window=1),
                truncation=strict, correction=corr_dist)
        @test corr_dist.accumulated_energy ≈ corr_serial.accumulated_energy

        # unsupported-on-distributed paths error loudly instead of being wrong
        @test_throws ErrorException truncate!(D, NoTruncation(), EnergyVarianceCorrection(ψ))
        @test_throws ErrorException truncate!(D, StochasticSamplingTruncation(5))
    end

    @testset "distributed AdaptiveTruncation (histogram threshold)" begin
        Random.seed!(17)
        O = rand(PauliSum{N}, n_paulis=200)
        A = rand(RankMap{N}, 3)
        D = DistributedPauliSum(O, A, comm)
        max_terms = 50
        truncate!(D, AdaptiveTruncation(max_terms=max_terms, min_thresh=1e-12))
        @test length(D) <= max_terms
        G = gather(D)
        if me == 0
            # approximate order statistic: every kept |c| is within one
            # half-decade bucket of every dropped |c|
            dropped = [abs(c) for (p, c) in O if !haskey(G, p)]
            kept = [abs(c) for (p, c) in G]
            @test minimum(kept) >= maximum(dropped) / 10^0.5
        end
    end

    @testset "protected generators are communication free (zero bytes)" begin
        Random.seed!(23)
        protected = [PauliBasis(Pauli(N, Z=[i, i+1])) for i in 1:N-1]   # all ZZ layers
        A = RankMap{N}(4, protected=protected)
        for G in protected
            @test bin_shift(A, G) == 0
        end

        O = rand(PauliSum{N}, n_paulis=20)
        D = DistributedPauliSum(O, A, comm)
        circ = compile(D, gens, angs, window=1)    # eager: merge i covers rotation i
        @test any(!=(0), circ.shifts)              # not in the degenerate limit
        counters = PropagationCounters()
        evolve!(D, circ; counters)

        for i in 1:L
            if circ.shifts[i] == 0
                @test counters.moved_per_rotation[i] == 0
                @test counters.shipped_per_merge[i] == 0
                @test counters.bytes_per_merge[i] == 0
            end
        end
        # ...and the unprotected layers really do communicate
        total_shipped = MPI.Allreduce(sum(counters.shipped_per_merge), +, comm)
        @test total_shipped > 0
        # eager evolution with a constrained map is still exact
        G = gather(D)
        me == 0 && @test isapprox(G, evolve(O, gens, angs), atol=1e-10)
    end

    @testset "distributed greedy bisection" begin
        Random.seed!(29)
        # clustered population: diagonal sector plus a few off-diagonal terms
        O = PauliSum(N)
        for i in 1:N, j in i+1:N
            O[PauliBasis(Pauli(N, Z=[i, j]))] = 1.0 + 0im
        end
        O[PauliBasis(Pauli(N, X=[1]))] = 1.0 + 0im
        O[PauliBasis(Pauli(N, X=[2]))] = 1.0 + 0im

        protected = [PauliBasis(Pauli(N, Z=[i, i+1])) for i in 1:3]
        Aseed = RankMap{N}(3, protected=protected)
        D = DistributedPauliSum(O, Aseed, comm)
        Agreedy = greedy_bisection_rankmap(D, 3, protected=protected, ncandidates=64)

        # replicated determinism: every rank computed the same map
        rows0 = MPI.bcast(Agreedy.rows, comm; root=0)
        @test rows0 == Agreedy.rows
        for G in protected
            @test bin_shift(Agreedy, G) == 0
        end
        # the greedy map balances the population at least as well as the seed
        maxbin(A) = maximum(bin_histogram(BinnedPauliSum(O, A)))
        @test maxbin(Agreedy) <= maxbin(Aseed)
    end

    @testset "bin reassignment rebalances and preserves the state" begin
        Random.seed!(31)
        # skewed population: everything Z-heavy so a z-blind map concentrates it
        O = PauliSum(N)
        for i in 1:N, j in i+1:N
            O[PauliBasis(Pauli(N, Z=[i, j]))] = rand(ComplexF64)
        end
        for _ in 1:10
            O[rand(PauliBasis{N})] = rand(ComplexF64)
        end
        A = rand(RankMap{N}, 4)
        D = DistributedPauliSum(O, A, comm)
        Oflat = gather(D)

        loads(DD) = begin
            h = bin_histogram(DD)
            l = zeros(Int, MPI.Comm_size(comm))
            for b in eachindex(h)
                l[DD.localpart.bin_owner[b]+1] += h[b]
            end
            l
        end
        before = maximum(loads(D))
        tv0 = D.localpart.table_version
        changed = rebalance_bins!(D)
        after = maximum(loads(D))
        @test check_ownership(D)
        @test after <= before
        changed && @test D.localpart.table_version == tv0 + 1
        # state untouched by the table edit
        G = gather(D)
        me == 0 && @test G == Oflat
        # evolution on the NON-CONTIGUOUS table is still exact vs serial
        ref = me == 0 ? evolve(Oflat, gens, angs) : nothing
        evolve!(D, compile(D, gens, angs, window=5))
        @test check_ownership(D)
        Gev = gather(D)
        me == 0 && @test isapprox(Gev, ref, atol=1e-10)
    end

    @testset "driver-integrated rebalancing" begin
        Random.seed!(37)
        O = PauliSum(N)
        O[PauliBasis(Pauli(N, Z=[1]))] = 1.0 + 0im
        gens2, angs2 = trotterize(H, 0.05, n_trotter=5, order=2)
        A = rand(RankMap{N}, 4)

        # without truncation, rebalancing must not change the result at all
        D = DistributedPauliSum(O, A, comm)
        evolve!(D, compile(D, gens2, angs2, window=8), rebalance_threshold=1.3)
        @test check_ownership(D)
        G = gather(D)
        me == 0 && @test isapprox(G, evolve(O, gens2, angs2), atol=1e-10)

        # with truncation, deviation stays at the windowed-cadence scale
        strict = CoeffTruncation(1e-5)
        Oref = evolve(O, gens2, angs2, truncation=strict)
        Dt = DistributedPauliSum(O, A, comm)
        evolve!(Dt, compile(Dt, gens2, angs2, window=8), truncation=strict,
                rebalance_threshold=1.3)
        @test check_ownership(Dt)
        Gt = gather(Dt)
        me == 0 && @test norm(Gt - Oref) / norm(Oref) < 1e-2
    end

    @testset "row swap across ranks" begin
        Random.seed!(41)
        O = rand(PauliSum{N}, n_paulis=25)
        A = rand(RankMap{N}, 4)
        D = DistributedPauliSum(O, A, comm)
        Oflat = gather(D)
        v0 = D.localpart.version
        newrow = RankRow(rand(Int128) & ((Int128(1) << N) - 1),
                         rand(Int128) & ((Int128(1) << N) - 1))
        swap_row!(D, 3, newrow)
        @test D.localpart.version == v0 + 1
        @test check_ownership(D)
        @test check_binning(D.localpart)
        G = gather(D)
        me == 0 && @test G == Oflat
        # recompiled circuit works after the swap and stays exact
        evolve!(D, compile(D, gens, angs, window=3))
        Gev = gather(D)
        if me == 0
            @test isapprox(Gev, evolve(Oflat, gens, angs), atol=1e-10)
        end
    end

    @testset "version guard" begin
        Random.seed!(19)
        O = rand(PauliSum{N}, n_paulis=5)
        A = rand(RankMap{N}, 2)
        D = DistributedPauliSum(O, A, comm)
        circ = compile(D, gens, angs)
        D.localpart.version += 1     # both ranks bump symmetrically: no hang
        @test_throws ErrorException evolve!(D, circ)
        @test_throws ErrorException merge_bins!(D, circ, 1)
    end
end

MPI.Finalize()
