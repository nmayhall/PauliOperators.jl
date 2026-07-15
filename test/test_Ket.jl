using PauliOperators
using Test
using LinearAlgebra

@testset "Ket" begin
    Random.seed!(1)
    N = 5
    for i in 1:100
        a = rand(Ket{7})
        b = rand(Ket{7})
        err = Vector(a) + Vector(b) - Vector(a+b) 
        @test norm(err) < 1e-14 
       

        err = Vector(a') + Vector(b') - Vector(a'+b')
        @test norm(err) < 1e-14
    end

    # Ket(N, v) always uses the canonical storage word, regardless of the
    # type of the integer literal — Ket(4, 0b0011) must match PauliBasis{4,UInt64}
    @test Ket(4, 0b0011) isa Ket{4, UInt64}
    @test Bra(4, 0b0011) isa Bra{4, UInt64}
    @test Ket(4, 0b0011) == Ket{4}(3)
    @test Ket(4, Int128(3)) isa Ket{4, UInt64}
    @test Ket(100, 3) isa Ket{100, UInt128}
end
