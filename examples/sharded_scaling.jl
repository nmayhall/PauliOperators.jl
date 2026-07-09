# Strong-scaling harness for the shared-memory sharded engine (milestone 2).
#
# Run with increasing thread counts and compare terms/s and effective
# bandwidth against the node's STREAM triad number, e.g.:
#
#   julia --project --threads=1 examples/sharded_scaling.jl
#   julia --project --threads=8 examples/sharded_scaling.jl
#   julia --project --threads=96 examples/sharded_scaling.jl
#
# Optional args: N nsteps r window  (defaults below). On Linux, load
# ThreadPinning before running for stable placement:
#   julia --project -t 96 -e 'using ThreadPinning; include("examples/sharded_scaling.jl")'

using PauliOperators
using LinearAlgebra
using Printf
using Random

N       = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 24
nsteps  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
r       = length(ARGS) >= 3 ? parse(Int, ARGS[3]) :
          max(2, round(Int, log2(Threads.nthreads())) + 4)   # ~16 shards/thread
window  = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 8

Random.seed!(1)

# Heisenberg chain, single-site Z observable in the middle
H = PauliSum(N)
for i in 1:N-1
    H[PauliBasis(Pauli(N, X=[i, i+1]))] = 1.0
    H[PauliBasis(Pauli(N, Y=[i, i+1]))] = 0.9
    H[PauliBasis(Pauli(N, Z=[i, i+1]))] = 1.1
end
O = PauliSum(N, Float64)
O[PauliBasis(Pauli(N, Z=[N ÷ 2]))] = 1.0

gens, angs = trotterize(H, 0.05, n_trotter=nsteps, order=2)
A = rand(RankMap{N}, r)
circ = compile(A, gens, angs; window)
nw = length(circ.window_subgroups)
trunc = WeightDampedTruncation(0.4, 1e-8)

nt = Threads.nthreads()
S = ShardedPauliSum(O, A; nthreads=nt, capacity_factor=8.0, append_factor=2.0)
pin_engine!(S)

# warm-up (compilation + capacity growth)
evolve!(S, compile(A, gens[1:min(2window, end)], angs[1:min(2window, end)]; window);
        truncation=trunc)

S = ShardedPauliSum(O, A; nthreads=nt, capacity_factor=8.0, append_factor=2.0)
cnt = WindowCounters(nw)
t = @elapsed evolve!(S, circ; truncation=trunc, counters=cnt,
                     rebalance_threshold=1.25)

nterms = length(S)
created = sum(cnt.terms_created)
# bytes swept per rotation ≈ population × (2 bit-words + coefficient)
wordbytes = 2 * sizeof(eltype(S.shards[1].z)) + sizeof(eltype(S.shards[1].c))
bytes = sum(cnt.merge_in) * wordbytes * 2 + created * wordbytes   # crude sweep+merge traffic

@printf("threads=%-3d  N=%d  r=%d (%d shards)  window=%d  rotations=%d\n",
        nt, N, r, 1 << r, window, length(gens))
@printf("final terms: %d   created: %d   early merges: %d\n",
        nterms, created, sum(cnt.early_merges))
@printf("wall: %.3f s   rotate: %.3f s   merge: %.3f s\n",
        t, sum(cnt.t_rotate), sum(cnt.t_merge))
@printf("throughput: %.3g terms/s   effective traffic: %.2f GB/s\n",
        created / t, bytes / t / 1e9)
@printf("steady-state GC allocation (windows 2+): %d bytes  <- must be 0\n",
        sum(cnt.allocd[2:end]))
