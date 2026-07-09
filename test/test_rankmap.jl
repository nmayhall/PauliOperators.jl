using PauliOperators
using Test
using Random

@testset "RankMap: GF(2) bin index and shifts" begin

    @testset "hand-computed 2-qubit cases" begin
        # Row watching only the z slot of qubit 1
        A = RankMap{2}([RankRow(Int128(0b01), Int128(0))])
        @test nbits(A) == 1
        @test nbins(A) == 2
        @test bin_index(A, PauliBasis("ZI")) == 1
        @test bin_index(A, PauliBasis("IZ")) == 0
        @test bin_index(A, PauliBasis("XI")) == 0   # x bits invisible to a z-only row
        @test bin_index(A, PauliBasis("YI")) == 1   # Y sets the z bit too
        @test bin_index(A, PauliBasis("II")) == 0

        # Two rows: (z of qubit 1), (x of qubit 2)
        A2 = RankMap{2}([RankRow(Int128(0b01), Int128(0)),
                         RankRow(Int128(0), Int128(0b10))])
        @test nbins(A2) == 4
        @test bin_index(A2, PauliBasis("ZX")) == 0b11
        @test bin_index(A2, PauliBasis("IX")) == 0b10
        @test bin_index(A2, PauliBasis("ZI")) == 0b01
    end

    @testset "identity Pauli is always bin 0" begin
        for N in (4, 70)
            for _ in 1:10
                A = rand(RankMap{N}, 6)
                @test bin_index(A, PauliBasis{N}(0, 0)) == 0
            end
        end
    end

    @testset "linearity: bin(G*p) == bin(p) ⊻ shift(G)" begin
        Random.seed!(2)
        # N=70 exercises the high Int128 bits
        for N in (4, 70)
            for _ in 1:100
                A = rand(RankMap{N}, rand(1:8))
                p = rand(PauliBasis{N})
                G = rand(PauliBasis{N})
                @test bin_index(A, PauliBasis(G * p)) ==
                      bin_index(A, p) ⊻ bin_shift(A, G)
            end
        end
    end

    @testset "duplicates co-locate" begin
        N = 8
        A = rand(RankMap{N}, 4)
        p = rand(PauliBasis{N})
        q = PauliBasis{N}(p.z, p.x)
        @test bin_index(A, p) == bin_index(A, q)
    end

    @testset "validation" begin
        # Row watching bits outside the register
        @test_throws ErrorException RankMap{2}([RankRow(Int128(0b100), Int128(0))])
        @test_throws ErrorException RankMap{2}([RankRow(Int128(0), Int128(1) << 50)])
        # Too many rows
        @test_throws ErrorException RankMap{4}([RankRow(Int128(1), Int128(0)) for _ in 1:21])
    end
end

@testset "Constrained RankMap construction (protected generators)" begin
    Random.seed!(21)

    @testset "nullspace dimension" begin
        N = 6
        zs = [PauliBasis(Pauli(N, Z=[i])) for i in 1:3]
        @test length(protected_row_basis(zs)) == 2N - 3
        # duplicate and dependent masks cost nothing
        @test length(protected_row_basis([zs; zs])) == 2N - 3
        y1 = PauliBasis(Pauli(N, Y=[1]))   # mask = Z₁ mask ⊻ X₁ mask
        x1 = PauliBasis(Pauli(N, X=[1]))
        z1 = PauliBasis(Pauli(N, Z=[1]))
        @test length(protected_row_basis([x1, z1, y1])) == 2N - 2
        # empty constraint set: full space
        @test length(protected_row_basis(PauliBasis{N}[])) == 2N
    end

    @testset "basis vectors satisfy the evenness constraints" begin
        N = 8
        protected = [PauliBasis(Pauli(N, Z=[i, i+1])) for i in 1:3]
        append!(protected, [PauliBasis(Pauli(N, X=[i, i+1])) for i in 1:3])
        basis = protected_row_basis(protected)
        @test length(basis) == 2N - 6
        for b in basis, G in protected
            @test bin_shift(RankMap{N}([b]), G) == 0
        end
    end

    @testset "constrained random maps protect exactly the protected set" begin
        N = 8
        protected = [PauliBasis(Pauli(N, Z=[i, i+1])) for i in 1:N-1]   # all ZZ bonds
        unprotected = [PauliBasis(Pauli(N, X=[i, i+1])) for i in 1:N-1]
        push!(unprotected, PauliBasis(Pauli(N, X=[1])))
        for _ in 1:20
            A = RankMap{N}(4, protected=protected)
            for G in protected
                @test bin_shift(A, G) == 0
            end
            # linearity identity holds for constrained maps too
            p, G = rand(PauliBasis{N}), rand(PauliBasis{N})
            @test bin_index(A, PauliBasis(G * p)) == bin_index(A, p) ⊻ bin_shift(A, G)
        end
        # random constrained maps generically see unprotected generators
        A = RankMap{N}(6, protected=protected)
        @test any(bin_shift(A, G) != 0 for G in unprotected)
    end

    @testset "over-constraining errors" begin
        N = 2
        all_single = [PauliBasis(Pauli(N, X=[1])), PauliBasis(Pauli(N, Z=[1])),
                      PauliBasis(Pauli(N, X=[2])), PauliBasis(Pauli(N, Z=[2]))]
        @test length(protected_row_basis(all_single)) == 0
        @test_throws ErrorException rand_valid_row(protected_row_basis(all_single))
        @test_throws ErrorException RankMap{N}(1, protected=all_single)
        # asking for more independent rows than the nullspace holds
        N2 = 4
        prot = [PauliBasis(Pauli(N2, Z=[i])) for i in 1:3]
        @test_throws ErrorException RankMap{N2}(6, protected=prot)
    end

    @testset "greedy bisection beats random on a clustered population" begin
        N = 8
        # heavily clustered: the diagonal (Z-string) sector plus a few X terms
        terms = PauliBasis{N}[]
        for i in 1:N, j in i+1:N
            push!(terms, PauliBasis(Pauli(N, Z=[i, j])))
        end
        for i in 1:N
            push!(terms, PauliBasis(Pauli(N, Z=[i])))
        end
        push!(terms, PauliBasis(Pauli(N, X=[1])), PauliBasis(Pauli(N, X=[2])))

        protected = [PauliBasis(Pauli(N, Z=[i, i+1])) for i in 1:3]
        r = 3
        Agreedy = greedy_bisection_rankmap(terms, r, protected=protected, ncandidates=64)
        for G in protected
            @test bin_shift(Agreedy, G) == 0
        end
        O = PauliSum(N)
        for p in terms
            O[p] = 1.0 + 0im
        end
        maxbin(A) = maximum(bin_histogram(BinnedPauliSum(O, A)))
        Arand = RankMap{N}(r, protected=protected)
        @test maxbin(Agreedy) <= maxbin(Arand)

        # PauliSum / BinnedPauliSum entry points agree with the term-vector one
        # (candidates and split counts are independent of term order)
        A2 = greedy_bisection_rankmap(O, r, protected=protected,
                                      rng=Random.MersenneTwister(1))
        A3 = greedy_bisection_rankmap(BinnedPauliSum(O, Agreedy), r, protected=protected,
                                      rng=Random.MersenneTwister(1))
        @test A2.rows == A3.rows
    end
end
