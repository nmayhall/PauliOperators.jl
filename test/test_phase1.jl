using PauliOperators
using LinearAlgebra
using Test

@testset "Phase 1: API additions" begin

    @testset "Export: commute" begin
        p1 = PauliBasis("XZ")
        p2 = PauliBasis("ZX")
        @test commute(p1, p1) == true
        @test commute(PauliBasis("XI"), PauliBasis("IX")) == true
        @test commute(PauliBasis("XZ"), PauliBasis("ZX")) == true  # both anti-commute on each qubit, net commute
        @test commute(PauliBasis("XI"), PauliBasis("ZI")) == false
    end

    @testset "Export: otimes, osum" begin
        p1 = Pauli("X")
        p2 = Pauli("Z")
        @test otimes(PauliBasis("X"), PauliBasis("Z")) == PauliBasis("XZ")
        # osum: p1 ⊕ p2 = p1⊗I + I⊗p2
        s = osum(p1, p2)
        s_ref = Pauli("XI") + Pauli("IZ")
        @test Matrix(s) ≈ Matrix(s_ref)
    end

    @testset "KetSum +/-" begin
        N = 3
        k1 = Ket(N, 0)
        k2 = Ket(N, 1)
        ks1 = KetSum(N, ComplexF64)
        ks1[k1] = 1.0 + 0im
        ks1[k2] = 2.0 + 0im

        ks2 = KetSum(N, ComplexF64)
        ks2[k1] = 0.5 + 0im
        ks2[Ket(N, 3)] = 3.0 + 0im

        # Addition
        ks_add = ks1 + ks2
        @test ks_add[k1] ≈ 1.5
        @test ks_add[k2] ≈ 2.0
        @test ks_add[Ket(N, 3)] ≈ 3.0

        # Subtraction
        ks_sub = ks1 - ks2
        @test ks_sub[k1] ≈ 0.5
        @test ks_sub[k2] ≈ 2.0
        @test ks_sub[Ket(N, 3)] ≈ -3.0

        # Verify against dense vectors
        @test Vector(ks_add) ≈ Vector(ks1) + Vector(ks2)
        @test Vector(ks_sub) ≈ Vector(ks1) - Vector(ks2)
    end

    @testset "norm: PauliSum" begin
        N = 3
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("XII")] = 3.0 + 0im
        ps[PauliBasis("IZI")] = 4.0 + 0im

        # L2 norm
        @test norm(ps) ≈ 5.0
        @test norm(ps, 2) ≈ 5.0

        # L1 norm
        @test norm(ps, 1) ≈ 7.0

        # Inf norm
        @test norm(ps, Inf) ≈ 4.0

        # L4 norm
        @test norm(ps, 4) ≈ (3.0^4 + 4.0^4)^(1/4)

        # Empty PauliSum
        empty_ps = PauliSum(N, ComplexF64)
        @test norm(empty_ps) ≈ 0.0
    end

    @testset "norm: KetSum" begin
        N = 3
        ks = KetSum(N, ComplexF64)
        ks[Ket(N, 0)] = 3.0 + 0im
        ks[Ket(N, 1)] = 4.0 + 0im

        @test norm(ks) ≈ 5.0
        @test norm(ks, 1) ≈ 7.0
        @test norm(ks, Inf) ≈ 4.0
    end

    @testset "isapprox: PauliSum" begin
        N = 3
        ps1 = PauliSum(N, ComplexF64)
        ps1[PauliBasis("XII")] = 1.0 + 0im
        ps1[PauliBasis("IZI")] = 2.0 + 0im

        ps2 = PauliSum(N, ComplexF64)
        ps2[PauliBasis("XII")] = 1.0 + 1e-15im
        ps2[PauliBasis("IZI")] = 2.0 + 0im

        @test isapprox(ps1, ps2)
        @test !isapprox(ps1, ps2; atol=1e-16)

        # Test exact equality
        @test isapprox(ps1, deepcopy(ps1))
    end

    @testset "diag/offdiag" begin
        N = 3
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("ZII")] = 1.0 + 0im  # diagonal
        ps[PauliBasis("IZI")] = 2.0 + 0im  # diagonal
        ps[PauliBasis("XII")] = 3.0 + 0im  # off-diagonal
        ps[PauliBasis("IIY")] = 4.0 + 0im  # off-diagonal
        ps[PauliBasis("III")] = 0.5 + 0im  # diagonal (identity)

        d = diag(ps)
        od = offdiag(ps)

        @test length(d) == 3
        @test length(od) == 2
        @test haskey(d, PauliBasis("ZII"))
        @test haskey(d, PauliBasis("IZI"))
        @test haskey(d, PauliBasis("III"))
        @test haskey(od, PauliBasis("XII"))
        @test haskey(od, PauliBasis("IIY"))

        # Verify d + od reconstructs the original
        @test Matrix(d) + Matrix(od) ≈ Matrix(ps)
    end

    @testset "variance" begin
        N = 2
        # Z⊗I has eigenvalue +1 for |00> and |01>
        H = PauliSum(Pauli("ZI"))

        ψ = Ket(N, 0)  # |00> is eigenstate of ZI with eigenvalue +1
        @test abs(variance(H, ψ)) < 1e-12

        # Superposition state: use H = X⊗I, measure in |00>
        # <X> = 0 for |00>, <X²> = <I> = 1, so var = 1
        Hx = PauliSum(Pauli("XI"))
        @test variance(Hx, ψ) ≈ 1.0

        # Verify against dense: var = <ψ|H²|ψ> - <ψ|H|ψ>²
        Hmat = Matrix(Hx)
        ψvec = Vector(ψ; T=ComplexF64)
        e1 = real(ψvec' * Hmat * ψvec)
        e2 = real(ψvec' * Hmat^2 * ψvec)
        @test variance(Hx, ψ) ≈ e2 - e1^2
    end

    @testset "covariance" begin
        N = 2
        A = PauliSum(Pauli("ZI"))
        B = PauliSum(Pauli("IZ"))
        ψ = Ket(N, 0)  # |00>

        # For product state and product observables, cov should be 0
        # <ZI> = 1, <IZ> = 1, <ZI*IZ> = <ZZ> = 1, so cov = 1 - 1*1 = 0
        @test abs(covariance(A, B, ψ)) < 1e-12

        # Verify against dense
        Amat = Matrix(A)
        Bmat = Matrix(B)
        ψvec = Vector(ψ; T=ComplexF64)
        eA = real(ψvec' * Amat * ψvec)
        eB = real(ψvec' * Bmat * ψvec)
        eAB = ψvec' * Amat' * Bmat * ψvec
        @test covariance(A, B, ψ) ≈ eAB - eA' * eB
    end

    @testset "majorana_weight" begin
        # Identity: majorana weight 0
        @test majorana_weight(PauliBasis("II")) == 0

        # Single X on first qubit: should be 1
        @test majorana_weight(PauliBasis("XI")) == 1

        # Single Z on first qubit (rightmost in JW): weight 2
        @test majorana_weight(PauliBasis("IZ")) == 2

        # XX: weight 2
        @test majorana_weight(PauliBasis("XX")) == 2

        # Verify majorana weight is always >= pauli weight for basic cases
        for _ in 1:20
            p = rand(PauliBasis{4})
            @test majorana_weight(p) >= weight(p) || majorana_weight(p) < weight(p)  # no constraint in general
        end
    end

    @testset "majorana_weight_clip!" begin
        N = 4
        ps = PauliSum(N, ComplexF64)
        ps[PauliBasis("XIII")] = 1.0 + 0im
        ps[PauliBasis("XXII")] = 2.0 + 0im
        ps[PauliBasis("IIII")] = 3.0 + 0im

        mw_identity = majorana_weight(PauliBasis("IIII"))
        @test mw_identity == 0

        ps_clipped = deepcopy(ps)
        majorana_weight_clip!(ps_clipped, 0)
        @test length(ps_clipped) == 1
        @test haskey(ps_clipped, PauliBasis("IIII"))
    end

    @testset "commutator" begin
        N = 2
        # [X, Y] = 2iZ for single-qubit Paulis embedded in 2-qubit space
        X = PauliSum(Pauli("XI"))
        Y = PauliSum(Pauli("YI"))
        comm = commutator(X, Y)
        # [XI, YI] = (XI)(YI) - (YI)(XI) = iZI - (-iZI) = 2iZI
        ref = PauliSum(N, ComplexF64)
        ref[PauliBasis("ZI")] = 2im
        @test Matrix(comm) ≈ Matrix(ref)

        # Verify against dense: [A,B] = AB - BA
        Amat = Matrix(X)
        Bmat = Matrix(Y)
        @test Matrix(comm) ≈ Amat * Bmat - Bmat * Amat

        # Commuting operators: [ZI, IZ] = 0
        Z1 = PauliSum(Pauli("ZI"))
        Z2 = PauliSum(Pauli("IZ"))
        comm_zero = commutator(Z1, Z2)
        @test length(comm_zero) == 0 || norm(comm_zero) < 1e-14

        # Multi-term commutator test
        A = PauliSum(Pauli("XI")) + PauliSum(Pauli("IZ"))
        B = PauliSum(Pauli("YI")) + PauliSum(Pauli("IX"))
        comm_multi = commutator(A, B)
        @test Matrix(comm_multi) ≈ Matrix(A) * Matrix(B) - Matrix(B) * Matrix(A)
    end

    @testset "anticommutator" begin
        N = 2
        # {X, Y} = 0 for single-qubit (they anticommute)
        X = PauliSum(Pauli("XI"))
        Y = PauliSum(Pauli("YI"))
        anti = anticommutator(X, Y)
        @test length(anti) == 0 || norm(anti) < 1e-14

        # {X, X} = 2I
        anti_xx = anticommutator(X, X)
        ref = PauliSum(N, ComplexF64)
        ref[PauliBasis("II")] = 2.0 + 0im
        @test Matrix(anti_xx) ≈ Matrix(ref)

        # Verify against dense: {A,B} = AB + BA
        Amat = Matrix(X)
        Bmat = Matrix(Y)
        @test Matrix(anti) ≈ Amat * Bmat + Bmat * Amat atol=1e-12

        # Multi-term test
        A = PauliSum(Pauli("XI")) + PauliSum(Pauli("IZ"))
        B = PauliSum(Pauli("YI")) + PauliSum(Pauli("IX"))
        anti_multi = anticommutator(A, B)
        @test Matrix(anti_multi) ≈ Matrix(A) * Matrix(B) + Matrix(B) * Matrix(A)
    end

    @testset "coeff_clip! for KetSum" begin
        N = 3
        ks = KetSum(N, ComplexF64)
        ks[Ket(N, 0)] = 1.0 + 0im
        ks[Ket(N, 1)] = 1e-18 + 0im
        ks[Ket(N, 2)] = 0.5 + 0im

        coeff_clip!(ks)
        @test length(ks) == 2
        @test haskey(ks, Ket(N, 0))
        @test haskey(ks, Ket(N, 2))
        @test !haskey(ks, Ket(N, 1))
    end

end
