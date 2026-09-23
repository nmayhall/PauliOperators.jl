using PauliOperators
using LinearAlgebra
using Test
using Random

@testset "Majorana mean-field factorization" begin

    @testset "k >= majorana_weight recovers c·pb" begin
        N = 6
        Random.seed!(2024)
        ψ = Ket(N, rand(Int128) & ((Int128(1) << N) - 1))
        for _ in 1:50
            pb = rand(PauliBasis{N})
            c  = randn(ComplexF64)
            mw = majorana_weight(pb)
            mf = majorana_mean_field_factorize(pb, c, ψ, mw)
            ref = PauliSum{N,ComplexF64}(pb => c)
            for key in union(keys(mf), keys(ref))
                @test get(mf, key, 0.0im) ≈ get(ref, key, 0.0im) atol = 1e-9
            end
        end
    end

    @testset "expectation preserved for every k; outputs Majorana-weight ≤ k" begin
        N = 6
        Random.seed!(7)
        ψ = Ket(N, rand(Int128) & ((Int128(1) << N) - 1))
        for _ in 1:50
            pb = rand(PauliBasis{N})
            c  = randn(ComplexF64)
            mw = majorana_weight(pb)
            ev0 = expectation_value(PauliSum{N,ComplexF64}(pb => c), ψ)
            for k in 0:mw
                mf = majorana_mean_field_factorize(pb, c, ψ, k)
                @test expectation_value(mf, ψ) ≈ ev0 atol = 1e-9
                for (p2, _) in mf
                    @test majorana_weight(p2) <= k
                end
            end
        end
    end

    @testset "off-diagonal support exceeding budget folds to nothing" begin
        N = 5
        ψ = Ket{N}(0)
        # X_1 has a single Majorana (γ_1), Majorana weight 1
        mf = majorana_mean_field_factorize(PauliBasis(Pauli(N, X=[1])), 1.0 + 0im, ψ, 0)
        @test length(mf) == 0
    end

    @testset "MajoranaMeanFieldTruncation strategy (PauliSum)" begin
        N = 8
        Random.seed!(99)
        ψ = Ket(N, rand(Int128) & ((Int128(1) << N) - 1))
        O = PauliSum(N, ComplexF64)
        for _ in 1:40
            O[rand(PauliBasis{N})] = randn(ComplexF64)
        end
        k = 3
        ev_before = expectation_value(O, ψ)
        truncate!(O, MajoranaMeanFieldTruncation(k, ψ))
        for p in keys(O)
            @test majorana_weight(p) <= k
        end
        @test expectation_value(O, ψ) ≈ ev_before atol = 1e-8
    end

    @testset "SparsePauliVector matches PauliSum" begin
        N = 8
        Random.seed!(0xBEEF)
        ψ = Ket(N, rand(Int128) & ((Int128(1) << N) - 1))
        for _ in 1:4
            O = PauliSum(N, ComplexF64)
            for _ in 1:30
                O[rand(PauliBasis{N})] = randn(ComplexF64)
            end
            k = 3
            Odict = deepcopy(O)
            truncate!(Odict, MajoranaMeanFieldTruncation(k, ψ))
            v = SparsePauliVector(O)
            truncate!(v, MajoranaMeanFieldTruncation(k, ψ))
            Ospv = PauliSum(v)
            for p in union(keys(Odict), keys(Ospv))
                @test get(Odict, p, 0.0im) ≈ get(Ospv, p, 0.0im) atol = 1e-10
            end
        end
    end

    @testset "correction path: energy delta ≈ 0" begin
        N = 6
        Random.seed!(5)
        ψ = Ket(N, rand(Int128) & ((Int128(1) << N) - 1))
        O = PauliSum(N, ComplexF64)
        for _ in 1:30
            O[rand(PauliBasis{N})] = randn(ComplexF64)
        end
        corr = EnergyVarianceCorrection(ψ)
        truncate!(O, MajoranaMeanFieldTruncation(2, ψ), corr)
        @test corr.accumulated_energy ≈ 0.0 atol = 1e-9
    end

end
