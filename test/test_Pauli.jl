using PauliOperators
using Test
using LinearAlgebra

@testset "convert" begin
    N = 5
    for i in 1:100
        a = rand(Pauli{N})
        err = Matrix(a) - coeff(a)*Matrix(PauliBasis(a))
        @test norm(err) < 1e-14
    end
end

@testset "iterate" begin
    for N in 1:4
        basis = PauliBasis{N}[]
        for p in PauliBasis{N}
            push!(basis, p)
        end
        @test length(basis) == 4^N
        @test length(Set(basis)) == 4^N

        ps = Pauli{N}[]
        for p in Pauli{N}
            push!(ps, p)
        end
        @test length(ps) == 4^N
        @test length(Set(ps)) == 4^N
    end
end
