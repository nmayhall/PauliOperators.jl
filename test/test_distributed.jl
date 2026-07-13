using Distributed

_dist_added = Int[]
if nprocs() == 1
    _dist_added = addprocs(2; exeflags="--threads=2")   # multithreaded workers
end

using PauliOperators
using Test

function _spv_distributed_fixture(N)
    O = PauliSum(N)
    O[PauliBasis(Pauli(N; Z=[1, 2]))] = 1.0 + 0.0im
    O[PauliBasis(Pauli(N; X=[2, 3]))] = 0.35 - 0.1im
    O[PauliBasis(Pauli(N; Y=[3]))] = -0.22 + 0.05im
    O[PauliBasis(Pauli(N; Z=[4, 5]))] = 0.17 + 0.0im
    O[PauliBasis(Pauli(N; X=[1], Z=[6]))] = -0.11 + 0.2im
    O[PauliBasis(Pauli(N; Y=[2, 5]))] = 0.09 - 0.03im

    gens = PauliBasis{N,PauliOperators.uinttype(N)}[
        PauliBasis(Pauli(N; X=[1, 2])),
        PauliBasis(Pauli(N; Y=[3])),
        PauliBasis(Pauli(N; Z=[2, 4])),
        PauliBasis(Pauli(N; X=[5], Z=[6])),
    ]
    angs = [0.21, -0.13, 0.17, 0.09]
    return O, gens, angs
end

function _max_coeffdiff(a::PauliSum{N}, b::PauliSum{N}) where {N}
    allk = union(keys(a), keys(b))
    isempty(allk) && return 0.0
    return maximum(abs(get(a, k, 0.0 + 0im) - get(b, k, 0.0 + 0im)) for k in allk)
end

@testset "distributed Pauli evolution" begin

    # workers are multithreaded, so the distributed==serial checks below also
    # exercise the on-node threaded rotation path (chunked + merged).
    @testset "workers are multithreaded" begin
        @test all(remotecall_fetch(Threads.nthreads, p) >= 1 for p in workers())
    end

    @testset "runtime storage selection" begin
        pid = first(workers())
        old = get(ENV, "PAULI_STORAGE", nothing)
        dS = nothing
        dD = nothing
        try
            O, _, _ = _spv_distributed_fixture(8)

            ENV["PAULI_STORAGE"] = "spv"
            @test pauli_storage() == :spv
            dS = distribute(O; workers=[pid])
            @test dS.storage == :spv
            @test remotecall_fetch(PauliOperators._dps_local_copy, pid, dS.id) isa SparsePauliVector

            ENV["PAULI_STORAGE"] = "dict"
            @test pauli_storage() == :dict
            dD = distribute(SparsePauliVector(O); workers=[pid])
            @test dD.storage == :dict
            @test remotecall_fetch(PauliOperators._dps_local_copy, pid, dD.id) isa PauliSum

            ENV["PAULI_STORAGE"] = "bad"
            @test_throws ArgumentError pauli_storage()
            @test pauli_storage(; default=:spv, env=nothing) == :spv
            @test pauli_storage(:paulisum) == :dict
            @test pauli_storage("SparsePauliVector") == :spv
        finally
            dS !== nothing && destroy!(dS)
            dD !== nothing && destroy!(dD)
            if old === nothing
                delete!(ENV, "PAULI_STORAGE")
            else
                ENV["PAULI_STORAGE"] = old
            end
        end
    end

    @testset "multithread SPV single-worker shard" begin
        pid = first(workers())
        @test remotecall_fetch(Threads.nthreads, pid) >= 2

        O, gens, angs = _spv_distributed_fixture(8)
        d_threaded = distribute(SparsePauliVector(O); workers=[pid], storage=:spv)
        d_serial = distribute(SparsePauliVector(O); workers=[pid], storage=:spv)

        evolve!(d_threaded, gens, angs; threaded=true)
        evolve!(d_serial, gens, angs; threaded=false)

        Othreaded = collect_paulisum(d_threaded)
        Oserial = collect_paulisum(d_serial)
        @test _max_coeffdiff(Othreaded, Oserial) < 1e-12
        @test remotecall_fetch(PauliOperators._dps_local_copy, pid, d_threaded.id) isa SparsePauliVector
        @test PauliOperators.check_spv(collect_sparsepaulivector(d_threaded))

        destroy!(d_threaded)
        destroy!(d_serial)
    end

    @testset "multinode SPV shards" begin
        pids = workers()[1:min(2, length(workers()))]
        @test length(pids) >= 2

        O, gens, angs = _spv_distributed_fixture(8)
        # Add terms until the hash partition has at least one initial term on
        # each selected worker, so this test really covers multinode storage.
        covered = Set(PauliOperators._pauli_owner(pb, pids) for pb in keys(O))
        for q in 1:8
            length(covered) == length(pids) && break
            pb = PauliBasis(Pauli(8; Z=[q]))
            O[pb] = 0.01q + 0.0im
            push!(covered, PauliOperators._pauli_owner(pb, pids))
        end
        @test length(covered) == length(pids)

        ref = deepcopy(O)
        for (G, θ) in zip(gens, angs)
            evolve!(ref, G, θ)
        end

        dO = distribute(SparsePauliVector(O); workers=pids, storage=:spv)
        @test dO.storage == :spv
        @test all(last(s) > 0 for s in sharded_summary(dO))
        @test all(remotecall_fetch(PauliOperators._dps_local_copy, pid, dO.id) isa SparsePauliVector
                  for pid in pids)

        evolve!(dO, gens, angs)
        got = collect_paulisum(dO)
        @test _max_coeffdiff(ref, got) < 1e-12
        @test PauliSum(collect_sparsepaulivector(dO)) == got

        destroy!(dO)
    end

    @testset "multinode Dict windowed pending merge" begin
        pids = workers()[1:min(2, length(workers()))]
        @test length(pids) >= 2

        O, gens, angs = _spv_distributed_fixture(8)
        ref = deepcopy(O)
        for (G, θ) in zip(gens, angs)
            evolve!(ref, G, θ)
        end

        dO = distribute(O; workers=pids, storage=:dict)
        evolve!(dO, gens, angs; window=3)
        got = collect_paulisum(dO)
        @test _max_coeffdiff(ref, got) < 1e-12
        destroy!(dO)
    end

    # Distributed evolution must reproduce serial evolution exactly (same
    # arithmetic, just hash-partitioned across workers).
    @testset "distributed == serial (N=$N, storage=$storage)" for N in (10, 14), storage in (:dict, :spv)
        O = PauliSum(N)
        O[PauliBasis(Pauli(N; Z=[1, 2]))] = 1.0 + 0.0im
        O[PauliBasis(Pauli(N; Z=[3, 4]))] = 0.5 + 0.0im
        O[PauliBasis(Pauli(N; X=[2]))]    = 0.25 + 0.0im
        # generators chosen to anticommute with some terms -> real branching/growth
        Tw = PauliOperators.uinttype(N)
        gens = PauliBasis{N,Tw}[]
        angs = Float64[]
        for k in 1:min(N - 1, 6)
            push!(gens, PauliBasis(Pauli(N; X=[k, k + 1]))); push!(angs, 0.17k)
            push!(gens, PauliBasis(Pauli(N; Y=[k])));        push!(angs, 0.11k)
        end

        Oser = deepcopy(O)
        for (G, θ) in zip(gens, angs)
            evolve!(Oser, G, θ)
        end

        input = storage == :spv ? SparsePauliVector(O) : O
        dO = distribute(input; workers=workers(), storage=storage)
        @test dO.storage == storage
        evolve!(dO, gens, angs)
        Odist = collect_paulisum(dO)
        if storage == :spv
            @test PauliSum(collect_sparsepaulivector(dO)) == Odist
        end

        @test length(Odist) == length(Oser)
        @test length(Oser) > length(O)     # evolution actually grew the sum
        maxdiff = 0.0
        for k in union(keys(Oser), keys(Odist))
            maxdiff = max(maxdiff, abs(get(Oser, k, 0.0 + 0im) - get(Odist, k, 0.0 + 0im)))
        end
        @test maxdiff < 1e-12

        # shard counts sum to the total
        @test sum(last, sharded_summary(dO)) == length(Odist)
        # 2-norm matches serial
        @test isapprox(opnorm2(dO), sqrt(sum(abs2, values(Oser))); atol=1e-10)
        destroy!(dO)
    end

    @testset "distributed clip == serial clip" begin
        N = 12
        O = PauliSum(N)
        for k in 1:6
            O[PauliBasis(Pauli(N; Z=[k]))] = ComplexF64(10.0^(-k))   # 0.1 .. 1e-6
        end
        Oser = deepcopy(O); coeff_clip!(Oser, 1e-3)
        dO = distribute(O; workers=workers()); coeff_clip!(dO, 1e-3)
        @test length(collect_paulisum(dO)) == length(Oser)
        destroy!(dO)

        dS = distribute(SparsePauliVector(O); workers=workers(), storage=:spv)
        coeff_clip!(dS, 1e-3)
        @test length(collect_paulisum(dS)) == length(Oser)
        destroy!(dS)
    end

    @testset "runs at N=1000 (storage=$storage)" for storage in (:dict, :spv)
        N = 1000
        O = PauliSum(N)
        O[PauliBasis(Pauli(N; Z=[1, 2]))] = 1.0 + 0.0im
        input = storage == :spv ? SparsePauliVector(O) : O
        dO = distribute(input; workers=workers(), storage=storage)
        gens = [PauliBasis(Pauli(N; X=[2, 3])), PauliBasis(Pauli(N; Y=[2]))]
        angs = [0.3, 0.2]
        evolve!(dO, gens, angs; truncation_thresh=1e-8)
        @test length(dO) >= 1
        @test isfinite(opnorm2(dO))
        destroy!(dO)
    end
end

if !isempty(_dist_added)
    rmprocs(_dist_added)
end
