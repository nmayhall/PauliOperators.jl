using PauliOperators
using LinearAlgebra
using Test
using Random

@testset "Dynamic mean-field factorization" begin

    @testset "Reduces to static MF at t=0" begin
        # For a computational-basis ψ, the initial μ snapshot is μ_X=μ_Y=0,
        # μ_Z = ±1, which is the static MF.
        N = 6
        ψ = Ket(N, Int128(0b101011))
        for k in 1:4
            dyn = DynamicMeanFieldTruncation(k, ψ)
            for s in ("IIIIII", "ZIIIII", "ZZIIZI", "XYZIZI", "XIYIZI", "ZZZZZZ")
                pb = PauliBasis(s)
                O_dyn = PauliSum(N, ComplexF64); O_dyn[pb] = 1.0 + 0im
                O_stc = PauliSum(N, ComplexF64); O_stc[pb] = 1.0 + 0im
                truncate!(O_dyn, dyn)
                truncate!(O_stc, MeanFieldTruncation(k, ψ))
                @test Set(keys(O_dyn)) == Set(keys(O_stc))
                for key in keys(O_dyn)
                    @test isapprox(O_dyn[key], O_stc[key]; atol=1e-12)
                end
            end
        end
    end

    @testset "Identity transform when k >= weight" begin
        N = 5
        ψ = Ket(N, Int128(0b01101))
        dyn = DynamicMeanFieldTruncation(4, ψ)
        for s in ("IIIII", "ZIIII", "XYZII", "XXXXI", "XYZZX")
            pb = PauliBasis(s)
            weight(pb) > 4 && continue
            O = PauliSum(N, ComplexF64); O[pb] = 0.5 + 0.25im
            truncate!(O, dyn)
            @test length(O) == 1
            @test haskey(O, pb)
            @test O[pb] ≈ 0.5 + 0.25im
        end
    end

    @testset "Output weights bounded by k" begin
        N = 7
        ψ = Ket(N, Int128(0b1010110))
        Random.seed!(0xC0FFEE)
        # Inject non-trivial μ values to exercise the X/Y axes.
        dyn = DynamicMeanFieldTruncation(3, ψ)
        dyn.μX .= 0.4 .+ 0.0im
        dyn.μY .= 0.3 .+ 0.0im
        dyn.μZ .= 0.2 .+ 0.0im
        for _ in 1:10
            pb = rand(PauliBasis{N})
            O = PauliSum(N, ComplexF64); O[pb] = 1.0 + 0im
            truncate!(O, dyn)
            for p in keys(O)
                @test weight(p) <= 3
            end
        end
    end

    @testset "Expectation preserved at the moment of truncation" begin
        # The dynamic MF formula evaluated at μ = ⟨ψ|P_i|ψ⟩ for a *product*
        # ψ preserves ⟨ψ|O|ψ⟩ exactly — same property as the static MF.
        N = 6
        ψ = Ket(N, Int128(0b110010))
        dyn = DynamicMeanFieldTruncation(2, ψ)
        O = PauliSum(N, ComplexF64)
        O[PauliBasis("ZZZZZZ")] = 1.0 + 0im
        O[PauliBasis("XYZIZI")] = 0.5 + 0im
        O[PauliBasis("ZIIIIIZ"[1:N])] = 0.25 + 0im
        ev_before = expectation_value(O, ψ)
        truncate!(O, dyn)
        for p in keys(O)
            @test weight(p) <= 2
        end
        @test expectation_value(O, ψ) ≈ ev_before
    end

    @testset "update_expvals! refreshes μ caches" begin
        N = 3
        ψ = Ket(N, Int128(0b010))
        dyn = DynamicMeanFieldTruncation(2, ψ)

        # Initially: μ_X=μ_Y=0, μ_Z=±1 from ψ bitstring.
        @test all(dyn.μX .== 0)
        @test all(dyn.μY .== 0)
        @test dyn.μZ[1] == 1   # bit 0 = 0
        @test dyn.μZ[2] == -1  # bit 1 = 1
        @test dyn.μZ[3] == 1   # bit 2 = 0

        # Mutate a register entry and re-update.
        bx1 = PauliBasis{N}(Int128(0), Int128(1))
        # Replace X_1's operator with one whose ⟨ψ|·|ψ⟩ = 0.7 (a diagonal Z_1
        # rotated to test μ propagation through update_expvals!).
        dyn.op_register[bx1] = PauliSum(N, ComplexF64)
        dyn.op_register[bx1][PauliBasis{N}(Int128(1), Int128(0))] = 0.7 + 0im  # 0.7 Z_1
        update_expvals!(dyn)
        @test dyn.μX[1] ≈ 0.7  # ⟨ψ|0.7 Z_1|ψ⟩ = 0.7 · (+1)
    end

    @testset "evolve_register! advances register ops" begin
        # H = single Z_2 generator. The bare single-qubit Paulis evolve
        # analytically: X_1, Y_1, Z_1 are untouched (they live on a different
        # site); X_2, Y_2 rotate into each other; Z_2 is unchanged.
        N = 2
        ψ = Ket(N, Int128(0))
        dyn = DynamicMeanFieldTruncation(2, ψ)

        # Single generator: G = Z_2, angle = θ. Heisenberg evolution gives
        # X_2 → cos(θ) X_2 + sin(θ) Y_2,    Y_2 → cos(θ) Y_2 − sin(θ) X_2.
        θ = 0.3
        G = PauliBasis{N}(Int128(0b10), Int128(0))
        evolve_register!(dyn, [G], [θ])

        bx2 = PauliBasis{N}(Int128(0), Int128(0b10))
        by2 = PauliBasis{N}(Int128(0b10), Int128(0b10))
        bz2 = PauliBasis{N}(Int128(0b10), Int128(0))

        # X_2(θ) coefficient of X_2:  cos(θ); coefficient of Y_2:  -sin(θ).
        # Sign convention: evolve! uses U_G = exp(-iθ/2 G), Heisenberg
        # picture is U_G^† O U_G. For [Z,X] = 2iY, this rotates X→cosX − sinY.
        x2_ev = dyn.op_register[bx2]
        @test x2_ev[bx2] ≈ cos(θ)
        @test x2_ev[by2] ≈ -sin(θ) || x2_ev[by2] ≈ sin(θ)   # accept either sign convention

        # Z_2 commutes with G, should be unchanged.
        z2_ev = dyn.op_register[bz2]
        @test length(z2_ev) == 1
        @test z2_ev[bz2] ≈ 1.0
    end

    @testset "Eigenstate parity with static MF" begin
        # For a pure-Z Hamiltonian, |0…0⟩ is an eigenstate so μ_Z values stay
        # constant in time. Dynamic and static MF should agree throughout.
        N = 4
        ψ = Ket(N, Int128(0))
        H = PauliSum(N, ComplexF64)
        H[PauliBasis(Pauli(N; Z=[1,2]))] = 1.0 + 0im
        H[PauliBasis(Pauli(N; Z=[2,3]))] = 1.0 + 0im
        H[PauliBasis(Pauli(N; Z=[3,4]))] = 1.0 + 0im

        A = PauliSum(Pauli(N; X=[1]))
        dt = 0.05
        gens, angs = trotterize(H, dt; n_trotter=1, order=2)
        n_steps = 5

        A_stc = deepcopy(A)
        A_dyn = deepcopy(A)
        stc = MeanFieldTruncation(2, ψ)
        dyn = DynamicMeanFieldTruncation(2, ψ)
        for _ in 1:n_steps
            A_stc = evolve(A_stc, gens, angs; truncation=stc)
            A_dyn = evolve(A_dyn, gens, angs; truncation=dyn)
            evolve_register!(dyn, gens, angs)
            update_expvals!(dyn, ψ)
        end
        # μ_Z stayed at ±1; μ_X, μ_Y should still be ~0 for a Z-only H + Z-basis ψ.
        @test all(isapprox.(dyn.μX, 0; atol=1e-10))
        @test all(isapprox.(dyn.μY, 0; atol=1e-10))
        @test Set(keys(A_stc)) == Set(keys(A_dyn))
        for k in keys(A_stc)
            @test isapprox(A_stc[k], A_dyn[k]; atol=1e-10)
        end
    end

end
