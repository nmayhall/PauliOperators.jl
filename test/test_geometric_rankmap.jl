using PauliOperators
using LinearAlgebra
using Test
using Random

# 2D Hubbard on Lx×Ly, Jordan-Wigner mapped. Qubit for (site s, spin σ∈1:2)
# is 2(s-1)+σ; site (ix,iy) has index ix + (iy-1)Lx.
function hubbard_2d(Lx, Ly; t=1.0, U=4.0)
    Nsites = Lx * Ly
    N = 2Nsites
    q(s, σ) = 2(s - 1) + σ
    site(ix, iy) = ix + (iy - 1) * Lx
    bonds = Tuple{Int,Int}[]
    for iy in 1:Ly, ix in 1:Lx
        ix < Lx && push!(bonds, (site(ix, iy), site(ix + 1, iy)))
        iy < Ly && push!(bonds, (site(ix, iy), site(ix, iy + 1)))
    end
    H = PauliSum(N)
    acc!(H, ps, w) = for (p, c) in ps
        H[p] = get(H, p, zero(ComplexF64)) + w * c
    end
    for (i, j) in bonds, σ in 1:2
        ai = jordan_wigner(q(i, σ), N)
        aj = jordan_wigner(q(j, σ), N)
        acc!(H, ai * aj', -t)
        acc!(H, aj * ai', -t)
    end
    for s in 1:Nsites
        nup = jordan_wigner(q(s, 1), N) * jordan_wigner(q(s, 1), N)'
        ndn = jordan_wigner(q(s, 2), N) * jordan_wigner(q(s, 2), N)'
        acc!(H, nup * ndn, U)
    end
    coeff_clip!(H, 1e-12)
    return H, bonds
end

# qubits (both spins) of a set of sites
_site_qubits(sites) = sort!(vcat([[2(s - 1) + 1, 2(s - 1) + 2] for s in sites]...))

@testset "Geometric rank maps on 2D Hubbard (design doc §5)" begin
    Random.seed!(51)
    Lx, Ly = 2, 2
    H, bonds = hubbard_2d(Lx, Ly)
    N = 2 * Lx * Ly
    q(s, σ) = 2(s - 1) + σ

    @testset "model sanity" begin
        @test ishermitian(H)
        # 4 bonds × 2 spins × 2 strings + 4×(I, Z↑, Z↓, ZZ) merged identity
        @test length(H) == 16 + 1 + 8 + 4
    end

    gens, angs = trotterize(H, 0.05, n_trotter=1)
    hop_gens = [G for G in gens if G.x != 0]
    diag_gens = [G for G in gens if G.x == 0]

    # left-column cut: x slots of both spin-orbitals of sites {1, 3}
    region = [1, 3]
    cut = RankRow(N, x=_site_qubits(region))
    Acut = RankMap{N}([cut])

    @testset "x-cut row = domain decomposition" begin
        for G in hop_gens
            qs = PauliOperators.get_on_bits(G.x)
            sites = unique((qq + 1) ÷ 2 for qq in qs)
            @test length(sites) == 2
            crossing = (sites[1] in region) ⊻ (sites[2] in region)
            @test bin_shift(Acut, G) == (crossing ? 1 : 0)
        end
        # both Pauli strings of one hop share the same x bits → same shift
        for xbits in unique(G.x for G in hop_gens)
            pair = [G for G in hop_gens if G.x == xbits]
            @test length(pair) == 2
            @test length(unique(bin_shift(Acut, G) for G in pair)) == 1
        end
        # every interaction term is Z-only → automatically free
        for G in diag_gens
            @test bin_shift(Acut, G) == 0
        end
    end

    @testset "z-pair rows split the diagonal sector but keep ZZ free" begin
        zrow = RankRow(N, z=[q(2, 1), q(2, 2)])   # watch z pair on site 2
        Az = RankMap{N}([zrow])
        @test bin_shift(Az, PauliBasis(Pauli(N, Z=[q(2, 1), q(2, 2)]))) == 0  # on-site ZZ free
        @test bin_shift(Az, PauliBasis(Pauli(N, Z=[q(2, 1)]))) == 1           # single Z pays
        @test bin_shift(Az, PauliBasis(Pauli(N, Z=[q(1, 1), q(1, 2)]))) == 0  # other sites unseen

        # the diagonal trap: an x-only map sends every Z string to bin 0
        diag_pop = PauliSum(N)
        for _ in 1:40
            zbits = rand(Int128) & ((Int128(1) << N) - 1)
            diag_pop[PauliBasis{N}(zbits, Int128(0))] = 1.0 + 0im
        end
        Bx = BinnedPauliSum(diag_pop, RankMap{N}([cut, RankRow(N, x=_site_qubits([1, 2]))]))
        @test collect(nonempty_bins(Bx)) == [0]
        # adding z-pair rows spreads it
        Amix = RankMap{N}([cut, RankRow(N, z=[q(1, 1), q(1, 2)]),
                           RankRow(N, z=[q(4, 1), q(4, 2)])])
        Bmix = BinnedPauliSum(diag_pop, Amix)
        @test length(collect(nonempty_bins(Bmix))) > 1
        @test check_binning(Bmix)
    end

    @testset "end-to-end: communication structure under real dynamics" begin
        # geometric map: two x-cuts (left column, bottom row) + one z-pair row
        A = RankMap{N}([RankRow(N, x=_site_qubits([1, 3])),
                        RankRow(N, x=_site_qubits([1, 2])),
                        RankRow(N, z=[q(1, 1), q(1, 2)])])
        gens5, angs5 = trotterize(H, 0.05, n_trotter=3, order=2)
        O = PauliSum(N)
        O[PauliBasis(Pauli(N, Z=[q(1, 1)]))] = 1.0 + 0im   # local density-like operator
        B = BinnedPauliSum(O, A)
        circ = compile(B, gens5, angs5, window=1)
        counters = PropagationCounters()
        evolve!(B, circ; counters)
        # protected rotations moved nothing; unprotected moved something at least once
        for i in 1:length(gens5)
            circ.shifts[i] == 0 && @test counters.moved_per_rotation[i] == 0
        end
        @test sum(counters.moved_per_rotation) > 0
        @test check_binning(B)
        # exactness against serial
        @test isapprox(PauliSum(B), evolve(O, gens5, angs5), atol=1e-10)
    end

    @testset "constrained solver reproduces the geometric structure" begin
        # protecting all diagonal generators + interior hops must yield a map
        # where those layers are free; the solver should find such rows
        interior = [G for G in hop_gens if begin
            qs = PauliOperators.get_on_bits(G.x)
            sites = unique((qq + 1) ÷ 2 for qq in qs)
            !((sites[1] in region) ⊻ (sites[2] in region))
        end]
        protected = vcat(diag_gens, interior)
        A = RankMap{N}(3, protected=protected)
        for G in protected
            @test bin_shift(A, G) == 0
        end
        crossing = [G for G in hop_gens if !(G in interior)]
        @test any(bin_shift(A, G) != 0 for G in crossing)
    end
end
