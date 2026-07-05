using PauliOperators
using Test

# Wide (>128 qubit) Pauli representation via BitIntegers, plus flexible coefficient
# float types. Target application: a 10x10x10 = 1000-qubit Heisenberg lattice.

@testset "wide Pauli representation" begin

    @testset "uinttype width selection" begin
        bits(N) = sizeof(PauliOperators.uinttype(N)) * 8
        @test PauliOperators.uinttype(4)   === UInt8
        @test PauliOperators.uinttype(8)   === UInt8
        @test PauliOperators.uinttype(9)   === UInt16
        @test PauliOperators.uinttype(64)  === UInt64
        @test PauliOperators.uinttype(128) === UInt128
        @test bits(129)  == 256          # BitIntegers.UInt256
        @test bits(256)  == 256
        @test bits(1000) == 1024         # BitIntegers.UInt1024
        @test PauliOperators.uinttype(1000) <: Unsigned
    end

    @testset "construction and type parameter (N=$N)" for N in (200, 256, 1000)
        T = PauliOperators.uinttype(N)
        zz = PauliBasis(Pauli(N; Z=[1, 2]))
        xx = PauliBasis(Pauli(N; X=[2, 3]))
        @test zz isa PauliBasis{N,T}
        @test weight(zz) == 2
        # Z1Z2 and X2X3 share exactly one anticommuting site (site 2) -> anticommute
        @test commute(zz, xx) == false
        # a disjoint pair commutes
        @test commute(PauliBasis(Pauli(N; Z=[1])), PauliBasis(Pauli(N; X=[5]))) == true
    end

    @testset "multiplication weight (N=$N)" for N in (200, 1000)
        # X2X3 * Z1Z2 = Z1 (X2 Z2) X3  ~ Z1 Y2 X3  (weight 3)
        prod = Pauli(N; X=[2, 3]) * Pauli(N; Z=[1, 2])
        @test weight(PauliBasis(prod)) == 3
    end

    @testset "Heisenberg-picture evolution (N=$N)" for N in (256, 1000)
        zz = PauliBasis(Pauli(N; Z=[1, 2]))
        G  = PauliBasis(Pauli(N; X=[2, 3]))
        O = PauliSum(N)
        O[zz] = 1.0 + 0.0im
        # one rotation with an anticommuting generator branches into 2 terms
        O2 = evolve(O, G, 0.3)
        @test length(O2) == 2
        # cos^2 + sin^2 norm preservation of the coefficient vector
        @test isapprox(sum(abs2, values(O2)), 1.0; atol=1e-12)
        # a short sequence then clip stays finite and shrinks
        Os = deepcopy(O)
        for k in 1:5
            Os = evolve(Os, PauliBasis(Pauli(N; X=[k, k + 1])), 0.2)
            Os = evolve(Os, PauliBasis(Pauli(N; Z=[k, k + 2])), 0.2)
        end
        n_before = length(Os)
        coeff_clip!(Os, 1e-3)
        @test length(Os) <= n_before
        @test all(isfinite, real.(values(Os)))
    end

    @testset "wide Ket expectation value (N=1000)" begin
        N = 1000
        k = Ket(N, 0)
        @test k isa Ket{N}
        @test sizeof(k.v) * 8 == 1024
        O = PauliSum(N)
        O[PauliBasis(Pauli(N; Z=[1, 2]))] = 1.0 + 0.0im   # <0|Z1Z2|0> = 1
        O[PauliBasis(Pauli(N; Z=[500]))]  = 2.0 + 0.0im   # <0|Z500|0>  = 2
        @test isapprox(expectation_value(O, k), 3.0 + 0.0im; atol=1e-12)
    end

    @testset "flexible coefficient float type" begin
        N = 1000
        G = PauliBasis(Pauli(N; X=[2, 3]))
        O32 = PauliSum(N, ComplexF32)
        O32[PauliBasis(Pauli(N; Z=[1, 2]))] = ComplexF32(1)
        O32b = evolve(O32, G, 0.3)
        @test valtype(O32b) === ComplexF32          # eltype preserved (half memory)
        O64 = PauliSum(N, ComplexF64)
        O64[PauliBasis(Pauli(N; Z=[1, 2]))] = 1.0 + 0.0im
        O64b = evolve(O64, G, 0.3)
        @test isapprox(sum(abs2, values(O32b)), sum(abs2, values(O64b)); atol=1e-5)
    end
end
