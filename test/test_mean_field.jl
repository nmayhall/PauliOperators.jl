using PauliOperators
using LinearAlgebra
using Test
using Random

@testset "Mean-field factorization" begin

    @testset "k >= weight recovers original" begin
        N = 6
        ψ = Ket(N, Int128(0b010110))
        for pb_str in ("ZIIZZI", "XYZIZI", "ZZZZZZ", "XIYIZI")
            pb = PauliBasis(pb_str)
            w  = weight(pb)
            mf = mean_field_factorize(pb, 1.0 + 0im, ψ, w)
            @test length(mf) == 1
            @test haskey(mf, pb)
            @test mf[pb] ≈ 1.0 + 0im
        end
    end

    @testset "k = 0 on Z-only is scalar × I" begin
        N = 5
        ψ = Ket(N, Int128(0b10110))
        pb = PauliBasis("ZZIZI")
        c  = 2.0 + 0im
        mf = mean_field_factorize(pb, c, ψ, 0)
        I_N = PauliBasis{N}(Int128(0), Int128(0))
        @test length(mf) == 1
        @test haskey(mf, I_N)
        ev = expectation_value(PauliSum{N,ComplexF64}(pb => c), ψ)
        @test mf[I_N] ≈ ev
    end

    @testset "k = 0 on XY-containing string is empty" begin
        N = 4
        ψ = Ket(N, Int128(0))
        for s in ("XIII", "IYII", "XYZI", "YYYY")
            mf = mean_field_factorize(PauliBasis(s), 1.0 + 0im, ψ, 0)
            @test length(mf) == 0
        end
    end

    @testset "expectation value is preserved for every k" begin
        N = 6
        ψ = Ket(N, Int128(0b101011))
        for s in ("IIIIII", "ZIIIII", "ZZIIZI", "XYZIZI", "XIYIZI", "ZZZZZZ")
            pb  = PauliBasis(s)
            w   = weight(pb)
            ev0 = expectation_value(PauliSum{N,ComplexF64}(pb => 1.0 + 0im), ψ)
            for k in 0:w
                mf = mean_field_factorize(pb, 1.0 + 0im, ψ, k)
                @test expectation_value(mf, ψ) ≈ ev0
            end
        end
    end

    @testset "output weights are bounded by k" begin
        N = 8
        ψ = Ket(N, Int128(0b10101010))
        Random.seed!(0xCAFE)
        for _ in 1:10
            pb = rand(PauliBasis{N})
            w  = weight(pb)
            for k in 0:max(w, 1)
                mf = mean_field_factorize(pb, 1.0 + 0im, ψ, k)
                for p in keys(mf)
                    @test weight(p) <= k
                end
            end
        end
    end

    @testset "_partial_alt_binom" begin
        @test PauliOperators._partial_alt_binom(0, 0) == 1
        @test PauliOperators._partial_alt_binom(0, 5) == 1
        @test PauliOperators._partial_alt_binom(5, -1) == 0
        @test PauliOperators._partial_alt_binom(5, 10) == 0
        # Σ_{m=0}^{2} C(4,m)(-1)^m = 1 - 4 + 6 = 3
        @test PauliOperators._partial_alt_binom(4, 2) == 3
        # Σ_{m=0}^{1} C(5,m)(-1)^m = 1 - 5 = -4
        @test PauliOperators._partial_alt_binom(5, 1) == -4
    end

    @testset "MeanFieldTruncation strategy" begin
        N = 8
        ψ = Ket(N, Int128(0b10101010))
        O = PauliSum(N, ComplexF64)
        O[PauliBasis("ZZZZZZZZ")] = 1.0 + 0im
        O[PauliBasis("XYZIZIZI")] = 0.5 + 0im
        O[PauliBasis("ZIIIIIIZ")] = 0.25 + 0im   # weight 2, untouched at max_weight=3

        ev_before = expectation_value(O, ψ)
        truncate!(O, MeanFieldTruncation(3, ψ))

        for p in keys(O)
            @test weight(p) <= 3
        end
        @test expectation_value(O, ψ) ≈ ev_before
    end

    @testset "mean_field_factorize! only expands high-weight terms" begin
        N = 4
        ψ = Ket(N, Int128(0))
        O = PauliSum(N, ComplexF64)
        O[PauliBasis("IIII")] = 1.0 + 0im          # weight 0
        O[PauliBasis("ZIII")] = 0.5 + 0im          # weight 1
        O[PauliBasis("ZZZZ")] = 0.25 + 0im         # weight 4 → expanded
        ev_before = expectation_value(O, ψ)

        mean_field_factorize!(O, ψ, 2)

        @test !haskey(O, PauliBasis("ZZZZ"))
        for p in keys(O)
            @test weight(p) <= 2
        end
        @test expectation_value(O, ψ) ≈ ev_before
    end

end
