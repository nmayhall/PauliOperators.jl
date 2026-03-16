using PauliOperators
using LinearAlgebra
using Test

@testset "Phase 4: Analysis & Utilities" begin

    @testset "get_weight_counts" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im   # weight 0
        ps[PauliBasis("XIII")] = 0.5 + 0im    # weight 1
        ps[PauliBasis("XXII")] = 0.3 + 0im    # weight 2
        ps[PauliBasis("XXXI")] = 0.2 + 0im    # weight 3
        ps[PauliBasis("ZIII")] = 0.1 + 0im    # weight 1

        counts = get_weight_counts(ps)
        @test length(counts) == N + 1
        @test counts[1] == 1   # weight 0
        @test counts[2] == 2   # weight 1
        @test counts[3] == 1   # weight 2
        @test counts[4] == 1   # weight 3
        @test counts[5] == 0   # weight 4
        @test sum(counts) == length(ps)
    end

    @testset "get_weight_probs" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im   # weight 0
        ps[PauliBasis("XIII")] = 0.5 + 0im    # weight 1

        probs = get_weight_probs(ps)
        @test probs[1] ≈ 1.0    # |1.0|² for weight 0
        @test probs[2] ≈ 0.25   # |0.5|² for weight 1
        @test sum(probs) ≈ sum(abs2(c) for (_, c) in ps)
    end

    @testset "get_majorana_weight_counts" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im  # majorana weight 0
        ps[PauliBasis("XIII")] = 0.5 + 0im
        ps[PauliBasis("XXXX")] = 0.1 + 0im

        counts = get_majorana_weight_counts(ps)
        @test length(counts) == 2N + 1
        @test counts[1] == 1  # majorana weight 0
        @test sum(counts) == length(ps)
    end

    @testset "get_majorana_weight_probs" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 2.0 + 0im
        ps[PauliBasis("ZIII")] = 1.0 + 0im

        probs = get_majorana_weight_probs(ps)
        @test probs[1] ≈ 4.0  # |2.0|² for majorana weight 0
        @test sum(probs) ≈ sum(abs2(c) for (_, c) in ps)
    end

    @testset "find_top_k" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 10.0 + 0im
        ps[PauliBasis("XIII")] = 5.0 + 0im
        ps[PauliBasis("XXII")] = 3.0 + 0im
        ps[PauliBasis("XXXI")] = 1.0 + 0im
        ps[PauliBasis("XXXX")] = 0.1 + 0im

        top = find_top_k(ps, 3)
        @test length(top) == 3
        @test abs(top[1].second) >= abs(top[2].second) >= abs(top[3].second)
        @test abs(top[1].second) ≈ 10.0
        @test abs(top[2].second) ≈ 5.0
        @test abs(top[3].second) ≈ 3.0

        # k larger than length returns all terms sorted
        top_all = find_top_k(ps, 100)
        @test length(top_all) == 5
    end

    @testset "largest" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 0.1 + 0im
        ps[PauliBasis("XIII")] = 5.0 + 0im
        ps[PauliBasis("XXII")] = 3.0 + 0im

        big = largest(ps)
        @test length(big) == 1
        @test haskey(big, PauliBasis("XIII"))
        @test big[PauliBasis("XIII")] ≈ 5.0 + 0im
    end

    @testset "largest_diag" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("IIII")] = 1.0 + 0im    # diagonal (x==0)
        ps[PauliBasis("ZIII")] = 3.0 + 0im    # diagonal (x==0)
        ps[PauliBasis("XIII")] = 10.0 + 0im   # off-diagonal (x!=0)

        pair = largest_diag(ps)
        @test pair.first == PauliBasis("ZIII")
        @test pair.second ≈ 3.0 + 0im
    end

    @testset "Matrix subspace" begin
        N = 3
        H = PauliSum(N, ComplexF64)
        H[PauliBasis("ZII")] = 1.0 + 0im
        H[PauliBasis("IZI")] = 0.5 + 0im
        H[PauliBasis("XXI")] = 0.3 + 0im

        # Full basis should match dense Matrix
        full_basis = [Ket(N, i) for i in 0:2^N-1]
        M_sub = Matrix(H, full_basis)
        M_full = Matrix(H)
        @test M_sub ≈ M_full

        # 2-ket subspace
        S = [Ket(N, 0), Ket(N, 1)]
        M2 = Matrix(H, S)
        @test size(M2) == (2, 2)
        # Diagonal: ⟨000|H|000⟩ and ⟨001|H|001⟩ should be expectation values
        @test M2[1,1] ≈ expectation_value(H, S[1])
        @test M2[2,2] ≈ expectation_value(H, S[2])
    end

    @testset "Vector subspace" begin
        N = 3
        k = KetSum(N, T=ComplexF64)
        k[Ket(N, 0)] = 0.8 + 0im
        k[Ket(N, 3)] = 0.6 + 0im

        S = [Ket(N, 0), Ket(N, 1), Ket(N, 3)]
        v = Vector(k, S)
        @test length(v) == 3
        @test v[1] ≈ 0.8 + 0im
        @test v[2] ≈ 0.0 + 0im
        @test v[3] ≈ 0.6 + 0im
    end

    @testset "show methods" begin
        N = 2
        # PauliBasis show
        p = PauliBasis("XI")
        buf = IOBuffer()
        show(buf, p)
        @test length(String(take!(buf))) > 0

        # Pauli show
        p2 = Pauli("XI")
        show(buf, p2)
        @test length(String(take!(buf))) > 0

        # PauliSum show (MIME)
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("XI")] = 1.0 + 0im
        show(buf, MIME("text/plain"), ps)
        s = String(take!(buf))
        @test contains(s, "XI")

        # DyadBasis show
        db = DyadBasis(Ket(N, 0), Bra(N, 1))
        show(buf, db)
        @test length(String(take!(buf))) > 0

        # DyadSum show (MIME)
        ds = DyadSum(DyadBasis(Ket(N, 0), Bra(N, 1)))
        show(buf, MIME("text/plain"), ds)
        @test length(String(take!(buf))) > 0
    end

end
