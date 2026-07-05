using Distributed

_dist_added = Int[]
if nprocs() == 1
    _dist_added = addprocs(2)
end

using PauliOperators
using Test

@testset "distributed Pauli evolution" begin

    # Distributed evolution must reproduce serial evolution exactly (same
    # arithmetic, just hash-partitioned across workers).
    @testset "distributed == serial (N=$N)" for N in (10, 14)
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

        dO = distribute(O; workers=workers())
        evolve!(dO, gens, angs)
        Odist = collect_paulisum(dO)

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
    end

    @testset "runs at N=1000" begin
        N = 1000
        O = PauliSum(N)
        O[PauliBasis(Pauli(N; Z=[1, 2]))] = 1.0 + 0.0im
        dO = distribute(O; workers=workers())
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
