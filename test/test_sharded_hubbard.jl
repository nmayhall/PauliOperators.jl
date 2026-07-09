using PauliOperators
using LinearAlgebra
using Test
using Random

# Small-size smoke test of the 2D Hubbard benchmark path (milestone 5):
# geometric x-cut + z-pair rank map, mixed protected/unprotected Trotter
# layers, sharded evolution vs the serial oracle.
@testset "Sharded 2D Hubbard smoke test" begin
    Random.seed!(29)
    Lx, Ly = 2, 2
    Nsites = Lx * Ly
    N = 2Nsites
    q(s, σ) = 2(s - 1) + σ
    site(ix, iy) = ix + (iy - 1) * Lx

    H = PauliSum(N)
    acc!(ps, w) = for (p, c) in ps
        H[p] = get(H, p, zero(ComplexF64)) + w * c
    end
    for iy in 1:Ly, ix in 1:Lx, (dx, dy) in ((1, 0), (0, 1))
        jx, jy = ix + dx, iy + dy
        (jx <= Lx && jy <= Ly) || continue
        i, j = site(ix, iy), site(jx, jy)
        for σ in 1:2
            ai = jordan_wigner(q(i, σ), N)
            aj = jordan_wigner(q(j, σ), N)
            acc!(ai * aj', -1.0)
            acc!(aj * ai', -1.0)
        end
    end
    for s in 1:Nsites
        nup = jordan_wigner(q(s, 1), N) * jordan_wigner(q(s, 1), N)'
        ndn = jordan_wigner(q(s, 2), N) * jordan_wigner(q(s, 2), N)'
        acc!(nup * ndn, 4.0)
    end
    coeff_clip!(H, 1e-12)

    # left-column x-cut + one z-pair row (splits the diagonal sector while
    # keeping on-site ZZ protected)
    left = [site(1, iy) for iy in 1:Ly]
    A = RankMap{N}([RankRow(N, x=sort!(vcat([[q(s, 1), q(s, 2)] for s in left]...))),
                    RankRow(N, z=[q(site(1, 1), 1), q(site(1, 1), 2)])])

    gens, angs = trotterize(H, 0.05, n_trotter=2, order=2)
    @test any(bin_shift(A, G) == 0 for G in gens)   # protected layers exist
    @test any(bin_shift(A, G) != 0 for G in gens)   # and unprotected ones

    O = PauliSum(N)
    O[PauliBasis(Pauli(N, Z=[q(site(1, 1), 1)]))] = 1.0 + 0.0im
    trunc = CoeffTruncation(1e-8)
    Oref = evolve(O, gens, angs; truncation=trunc)

    maxt = min(4, Threads.nthreads())
    for nt in unique((1, maxt))
        S = ShardedPauliSum(O, A; T=ComplexF64, nthreads=nt, min_capacity=2048)
        cnt = WindowCounters(length(gens))
        evolve!(S, compile(A, gens, angs; window=1); truncation=trunc, counters=cnt)
        @test check_sharding(S)
        @test sum(cnt.early_merges) == 0
        Og = PauliSum(S)
        @test length(Og) == length(Oref)
        @test all(Og[k] == Oref[k] for k in keys(Oref))   # bit-exact, window=1
    end

    # greedy bisection against the evolved population returns a valid,
    # better-balanced map that still protects what it was asked to protect
    S = ShardedPauliSum(O, A; T=ComplexF64, min_capacity=2048)
    evolve!(S, compile(A, gens, angs; window=4); truncation=trunc)
    zz = [G for G in gens if G.x == 0]
    A2 = greedy_bisection_rankmap(S, 3; protected=zz)
    @test nbits(A2) == 3
    @test all(bin_shift(A2, G) == 0 for G in zz)
end
