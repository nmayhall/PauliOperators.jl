# 2D Hubbard reference stress test for the shared-memory sharded engine
# (milestone 5 of the shared-memory design). Exercises the diagonal Z-string
# sector, mixed protected/unprotected Trotter layers, and light-cone growth.
#
# Strong-scaling usage (repeat with increasing --threads):
#   julia --project --threads=1  examples/hubbard2d_sharded_benchmark.jl 4 4
#   julia --project --threads=16 examples/hubbard2d_sharded_benchmark.jl 4 4
#   julia --project --threads=96 examples/hubbard2d_sharded_benchmark.jl 6 6
#
# Args: Lx Ly [n_trotter] [window] [damping alpha] [coeff thresh]
# N = 2·Lx·Ly qubits (spin-orbitals). On Linux add `using ThreadPinning`
# first so pin_engine! actually pins.

using PauliOperators
using LinearAlgebra
using Printf
using Random

Lx     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
Ly     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
nsteps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4
window = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 8
alpha  = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.3
thresh = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 1e-8

Random.seed!(2026)

# ---------------- model ----------------
q(s, σ) = 2(s - 1) + σ
site(ix, iy) = ix + (iy - 1) * Lx

function hubbard_2d(Lx, Ly; t=1.0, U=4.0)
    Nsites = Lx * Ly
    N = 2Nsites
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
            acc!(ai * aj', -t)
            acc!(aj * ai', -t)
        end
    end
    for s in 1:Nsites
        nup = jordan_wigner(q(s, 1), N) * jordan_wigner(q(s, 1), N)'
        ndn = jordan_wigner(q(s, 2), N) * jordan_wigner(q(s, 2), N)'
        acc!(nup * ndn, U)
    end
    coeff_clip!(H, 1e-12)
    return H
end

# ---------------- geometric rank map ----------------
# Vertical x-cut rows (domain decomposition for hopping: only hops crossing
# a cut shift bins; every Z-only interaction term is automatically free)
# plus same-site z-pair rows (split the diagonal Z-string sector while
# keeping on-site ZZ interaction generators protected).
function geometric_rankmap(Lx, Ly; ncuts=min(Lx - 1, 2), nzpairs=2)
    N = 2 * Lx * Ly
    rows = RankRow[]
    for c in 1:ncuts
        xcol = round(Int, c * Lx / (ncuts + 1))
        left = [site(ix, iy) for ix in 1:xcol, iy in 1:Ly]
        push!(rows, RankRow(N, x=sort!(vcat([[q(s, 1), q(s, 2)] for s in vec(left)]...))))
    end
    for k in 1:nzpairs
        s = site(clamp(k, 1, Lx), clamp(k, 1, Ly))
        push!(rows, RankRow(N, z=[q(s, 1), q(s, 2)]))
    end
    return RankMap{N}(rows)
end

# ---------------- run ----------------
N = 2 * Lx * Ly
H = hubbard_2d(Lx, Ly)
gens, angs = trotterize(H, 0.05, n_trotter=nsteps, order=2)
A = geometric_rankmap(Lx, Ly)
nprotected = count(G -> bin_shift(A, G) == 0, gens)

O = PauliSum(N, Float64)
O[PauliBasis(Pauli(N, Z=[q(site((Lx + 1) ÷ 2, (Ly + 1) ÷ 2), 1)]))] = 1.0  # central n↑ probe

circ = compile(A, gens, angs; window)
nw = length(circ.window_subgroups)
trunc = WeightDampedTruncation(alpha, thresh)
nt = Threads.nthreads()

@printf("2D Hubbard %dx%d  (N=%d qubits, %s words)  threads=%d\n",
        Lx, Ly, N, N <= 64 ? "UInt64" : "UInt128", nt)
@printf("rotations=%d (%d protected / %d total)  shards=%d  window=%d\n",
        length(gens), nprotected, length(gens), 1 << length(A.rows), window)

S = ShardedPauliSum(O, A; nthreads=nt, capacity_factor=16.0, append_factor=2.0,
                    min_capacity=4096)
pin_engine!(S)
nwarm = min(2window, length(gens))
evolve!(S, compile(A, gens[1:nwarm], angs[1:nwarm]; window); truncation=trunc)  # JIT warm-up

S = ShardedPauliSum(O, A; nthreads=nt, capacity_factor=16.0, append_factor=2.0,
                    min_capacity=4096)
cnt = WindowCounters(nw)
wall = @elapsed evolve!(S, circ; truncation=trunc, counters=cnt, rebalance_threshold=1.25)

created = sum(cnt.terms_created)
wordbytes = 2 * sizeof(eltype(S.shards[1].z)) + sizeof(eltype(S.shards[1].c))
bytes = sum(cnt.merge_in) * wordbytes * 2 + created * wordbytes

@printf("final terms: %d    created: %d    cross-shard: %d (%.1f%%)\n",
        length(S), created, sum(cnt.cross_appends),
        100 * sum(cnt.cross_appends) / max(created, 1))
@printf("early merges: %d    max shard pop: %d\n",
        sum(cnt.early_merges), maximum(cnt.max_shard_pop))
@printf("wall %.3f s   rotate %.3f s   merge %.3f s\n",
        wall, sum(cnt.t_rotate), sum(cnt.t_merge))
@printf("throughput %.3g created-terms/s    effective traffic %.2f GB/s\n",
        created / wall, bytes / wall / 1e9)
@printf("steady-state GC allocation (windows 2+): %d bytes  <- must be 0\n",
        sum(cnt.allocd[2:end]))

# correctness cross-check at small sizes
if N <= 16
    Oref = evolve(PauliSum{N,ComplexF64}(k => v for (k, v) in O), gens, angs;
                  truncation=trunc)
    d = norm(PauliSum(S) - Oref) / norm(Oref)
    @printf("relative deviation vs serial (truncation cadence M=%d vs 1): %.2e\n",
            window, d)
end
