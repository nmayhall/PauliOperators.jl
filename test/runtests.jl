using PauliOperators
using MPI
using Test

@testset "Paulis.jl" begin
    include("test_operator_methods.jl")
    include("test_Pauli.jl")
    include("test_Ket.jl")
    include("test_multiplication.jl")
    include("test_addition.jl")
    include("test_allocations.jl")
    include("test_stochastic.jl")
    include("test_phase1.jl")
    include("test_truncation.jl")
    include("test_evolution.jl")
    include("test_rankmap.jl")
    include("test_binned_paulisum.jl")
    include("test_binned_evolve.jl")
    include("test_geometric_rankmap.jl")
    include("test_analysis.jl")
    include("test_channels.jl")
    include("test_transformations.jl")

    # Distributed tests run in subprocesses under mpiexec (set
    # PAULIOPERATORS_TEST_MPI=false to skip, e.g. on constrained CI)
    @testset "MPI (2 ranks)" begin
        if get(ENV, "PAULIOPERATORS_TEST_MPI", "true") == "true"
            script = joinpath(@__DIR__, "mpi", "runtests.jl")
            exe = MPI.mpiexec()
            p = run(ignorestatus(`$exe -n 2 $(Base.julia_cmd()) --project=$(Base.active_project()) $script`))
            @test success(p)
        end
    end
end
