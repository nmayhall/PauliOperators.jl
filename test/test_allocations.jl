using PauliOperators
using Test
using Printf
using LinearAlgebra
using Random
using BenchmarkTools

@testset "Allocations" begin
    Random.seed!(1)
    
    function test()
        a = rand(PauliBasis{9})
        b = rand(Pauli{9})
        c = rand(PauliSum{9})
        d = rand(DyadBasis{9})
        e = rand(Dyad{9})
        f = rand(DyadSum{9})
        g = rand(Ket{9})
        h = rand(KetSum{9})
    end
    @test (@ballocated $test) == 0
end
   

@testset "Allocations *" begin
    function test()
        for i in 1:5
            a = rand(PauliBasis{i})
            b = rand(Pauli{i})
            c = rand(PauliSum{i})
            d = rand(DyadBasis{i})
            e = rand(Dyad{i})
            f = rand(DyadSum{i})
            g = rand(Ket{i})
            h = rand(KetSum{i})
            aa = a * a
            ab = a * b
            ac = a * c
            ad = a * d
            ae = a * e
            af = a * f
            ag = a * g
            
            ba = b * a
            bb = b * b
            bc = b * c
            bd = b * d
            be = b * e
            bf = b * f
            bg = b * g
            
            ca = c * a
            cb = c * b
            cc = c * c
            cd = c * d
            ce = c * e
            cf = c * f
            # cg = c * g

            # Needs to flesh this out more
        end
        return 1
    end
    test()
    @test (@ballocated $test) == 0
end