using PauliOperators
using LinearAlgebra
using Test
using Random

function _heisenberg_chain(N; Jx=1.0, Jy=0.9, Jz=1.1)
    H = PauliSum(N)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, X=[i, i+1]))] = Jx
        H[PauliBasis(Pauli(N, Y=[i, i+1]))] = Jy
        H[PauliBasis(Pauli(N, Z=[i, i+1]))] = Jz
    end
    return H
end

@testset "Binned evolution" begin
    Random.seed!(4)

    @testset "gf2_span" begin
        @test gf2_span(Int[]) == [0]
        @test gf2_span([0, 0]) == [0]
        @test gf2_span([1, 2]) == [0, 1, 2, 3]
        @test gf2_span([3, 5]) == [0, 3, 5, 6]
        @test gf2_span([1, 2, 3]) == [0, 1, 2, 3]      # 3 is dependent
        @test gf2_span([1, 2]; cap=1) === nothing
    end

    @testset "compile" begin
        N = 4
        H = _heisenberg_chain(N)
        gens, angs = trotterize(H, 0.1, n_trotter=2)
        A = rand(RankMap{N}, 3)
        circ = compile(A, gens, angs, window=4)
        @test length(circ) == length(gens)
        @test circ.shifts == [bin_shift(A, G) for G in gens]
        @test length(circ.window_subgroups) == cld(length(gens), 4)
        for (w, sg) in enumerate(circ.window_subgroups)
            lo, hi = (w-1)*4 + 1, min(w*4, length(gens))
            @test sg == gf2_span(circ.shifts[lo:hi])
        end
        # degenerate all-shifts-zero limit warns
        Hzz = PauliSum(N)
        for i in 1:N-1
            Hzz[PauliBasis(Pauli(N, Z=[i, i+1]))] = 1.0
        end
        gz, az = trotterize(Hzz, 0.1)
        Ax = RankMap{N}([RankRow(Int128(0), Int128(0b0011))])  # x-only row: blind to Z strings
        @test_logs (:warn, r"never leave") compile(Ax, gz, az)
    end

    @testset "eager and windowed exactness (no truncation)" begin
        for N in (6, 8)
            O = rand(PauliSum{N}, n_paulis=10)
            H = _heisenberg_chain(N)
            gens, angs = trotterize(H, 0.1, n_trotter=2, order=2)
            Oref = evolve(O, gens, angs)
            for r in (1, 2, 4), window in (1, 4, 1000)
                A = rand(RankMap{N}, r)
                B = BinnedPauliSum(O, A)
                circ = compile(B, gens, angs; window)
                evolve!(B, circ)
                @test check_binning(B)
                @test isapprox(PauliSum(B), Oref, atol=1e-10)
                @test norm(B) ≈ norm(O)   # unitary conjugation preserves Frobenius norm
            end
        end
    end

    @testset "dense-matrix oracle (N=4)" begin
        N = 4
        O = rand(PauliSum{N}, n_paulis=6)
        H = _heisenberg_chain(N)
        gens, angs = trotterize(H, 0.2)
        A = rand(RankMap{N}, 2)
        B = BinnedPauliSum(O, A)
        evolve!(B, compile(B, gens, angs))
        U = Matrix{ComplexF64}(I, 2^N, 2^N)
        for (G, θ) in zip(gens, angs)
            U = U * exp(-1im * θ/2 * Matrix(G))
        end
        @test Matrix(PauliSum(B)) ≈ U' * Matrix(O) * U atol = 1e-10
    end

    @testset "sin-branch destination and phase" begin
        # A watches the z slot of qubit 1: G = Z₁ has shift 1
        A = RankMap{2}([RankRow(Int128(0b01), Int128(0))])
        p = PauliBasis("XI")
        G = PauliBasis("ZI")
        @test bin_shift(A, G) == 1
        c = 0.7 + 0.2im
        θ = 0.3
        O = PauliSum(2)
        O[p] = c
        B = BinnedPauliSum(O, A)
        @test collect(nonempty_bins(B)) == [0]
        evolve!(B, G, θ)
        # cos branch stays in bin 0; sin branch (G*p ∝ Y₁) lands in bin 0 ⊻ 1
        @test B.bins[1][p] ≈ c * cos(θ)
        y = PauliBasis("YI")
        @test haskey(B.bins[2], y)
        @test B.bins[2][y] ≈ coeff(c * (1im*sin(θ)) * G * Pauli(p))
        @test check_binning(B)
        # and the whole thing matches the serial kernel
        @test isapprox(PauliSum(B), evolve(O, G, θ), atol=1e-14)
    end

    @testset "two-phase regression: bins mapped onto each other" begin
        # XI in bin 0 and YI in bin 1; G = Z₁ has shift 1, so each bin's sin
        # branch lands in the other bin mid-rotation. A single-phase kernel
        # would rotate the delivered branches a second time.
        A = RankMap{2}([RankRow(Int128(0b01), Int128(0))])
        G = PauliBasis("ZI")
        O = PauliSum(2)
        O[PauliBasis("XI")] = 1.0 + 0im
        O[PauliBasis("YI")] = 0.5 + 0im
        B = BinnedPauliSum(O, A)
        θ = 0.4
        evolve!(B, G, θ)
        @test isapprox(PauliSum(B), evolve(O, G, θ), atol=1e-14)
    end

    @testset "windowed cadence with truncation" begin
        N = 8
        O = PauliSum(N)
        O[PauliBasis(Pauli(N, Z=[1]))] = 1.0 + 0im
        H = _heisenberg_chain(N)
        gens, angs = trotterize(H, 0.05, n_trotter=5, order=2)
        strict = CoeffTruncation(1e-4)
        loose = CoeffTruncation(1e-5)
        Oref = evolve(O, gens, angs, truncation=strict)
        A = rand(RankMap{N}, 3)

        # M=1: loose-then-strict every rotation ≡ serial per-rotation strict
        B1 = BinnedPauliSum(O, A)
        evolve!(B1, compile(B1, gens, angs, window=1),
                truncation=strict, local_truncation=loose)
        @test isapprox(PauliSum(B1), Oref, atol=1e-10)

        # larger windows: loose local truncation between merges introduces a
        # small, bounded cadence error
        for M in (4, 16)
            B = BinnedPauliSum(O, A)
            evolve!(B, compile(B, gens, angs, window=M),
                    truncation=strict, local_truncation=loose)
            @test check_binning(B)
            rel_err = norm(PauliSum(B) - Oref) / norm(Oref)
            @test rel_err < 1e-2
        end
    end

    @testset "counters" begin
        N = 6
        O = rand(PauliSum{N}, n_paulis=8)
        H = _heisenberg_chain(N)
        gens, angs = trotterize(H, 0.1)
        A = rand(RankMap{N}, 3)
        B = BinnedPauliSum(O, A)
        circ = compile(B, gens, angs, window=4)
        counters = PropagationCounters()
        evolve!(B, circ; counters)
        L = length(gens)
        @test counters.rotations == L
        @test counters.merges == cld(L, 4)
        @test length(counters.moved_per_rotation) == L
        # protected rotations (shift 0) never move terms between bins
        for i in 1:L
            circ.shifts[i] == 0 && @test counters.moved_per_rotation[i] == 0
        end
        @test all(==(0), counters.shipped_per_merge)   # single node ships nothing
    end

    @testset "degenerate limit: all shifts zero" begin
        N = 4
        Hzz = PauliSum(N)
        for i in 1:N-1
            Hzz[PauliBasis(Pauli(N, Z=[i, i+1]))] = 1.0
        end
        gens, angs = trotterize(Hzz, 0.3, n_trotter=3)
        Ax = RankMap{N}([RankRow(Int128(0), Int128(0b0011)),
                         RankRow(Int128(0), Int128(0b1100))])
        O = rand(PauliSum{N}, n_paulis=10)
        B = BinnedPauliSum(O, Ax)
        initial_nonempty = Set(collect(nonempty_bins(B)))
        circ = @test_logs (:warn, r"never leave") compile(B, gens, angs)
        evolve!(B, circ)
        @test Set(collect(nonempty_bins(B))) ⊆ initial_nonempty
        @test check_binning(B)
    end

    @testset "version guard" begin
        N = 4
        O = rand(PauliSum{N}, n_paulis=5)
        B = BinnedPauliSum(O, rand(RankMap{N}, 2))
        gens, angs = trotterize(_heisenberg_chain(N), 0.1)
        circ = compile(B, gens, angs)
        B.version += 1
        @test_throws ErrorException evolve!(B, circ)
    end
end
