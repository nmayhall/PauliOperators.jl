# Scaling-study probe (design doc milestone 6). One run = one CSV row on
# rank 0. Sweep it over rank counts and windows with examples/run_scaling_sweep.jl,
# or under a cluster scheduler:
#
#     mpiexec -n <np> julia --project examples/scaling_study.jl [N n_trotter dt r window strict_exp check]
#
# CSV columns:
#     np,N,rotations,r,window,strict,wall_s,terms,shipped_records,shipped_KB,rebalances,rel_err
# rel_err is vs the serial per-rotation-truncated reference (computed on
# rank 0 when `check` is 1 — feasible only for modest N) else NaN.
using MPI
using PauliOperators
using LinearAlgebra
using Printf
using Random

MPI.Init()
const comm = MPI.COMM_WORLD
const np = MPI.Comm_size(comm)
const me = MPI.Comm_rank(comm)

N          = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
n_trotter  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
dt         = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.1
r          = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : trailing_zeros(np) + 4
window     = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 16
strict_exp = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 7.0
check      = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 1

H = PauliSum(N)
for i in 1:N-1
    H[PauliBasis(Pauli(N, X=[i, i+1]))] = 1.0
    H[PauliBasis(Pauli(N, Y=[i, i+1]))] = 0.9
    H[PauliBasis(Pauli(N, Z=[i, i+1]))] = 1.1
end
O = PauliSum(N)
O[PauliBasis(Pauli(N, Z=[N ÷ 2]))] = 1.0 + 0im
gens, angs = trotterize(H, dt; n_trotter, order=2)
strict = CoeffTruncation(10.0^-strict_exp)
loose = CoeffTruncation(10.0^-(strict_exp + 1))

Random.seed!(60)                       # identical map on every rank
A = rand(RankMap{N}, r)
D = DistributedPauliSum(O, A, comm)
circ = compile(D, gens, angs; window)
counters = PropagationCounters()

# warm up compilation with a short prefix, on a throwaway copy
let Dw = DistributedPauliSum(O, A, comm)
    evolve!(Dw, compile(Dw, gens[1:min(10, end)], angs[1:min(10, end)]; window=1),
            truncation=strict)
end

MPI.Barrier(comm)
t = @elapsed evolve!(D, circ, truncation=strict, local_truncation=loose,
                     rebalance_threshold=1.5; counters)
MPI.Barrier(comm)

shipped = MPI.Allreduce(sum(counters.shipped_per_merge), +, comm)
bytes = MPI.Allreduce(sum(counters.bytes_per_merge), +, comm)
nterms = length(D)
rebalances = counters.merges - cld(length(gens), window)   # extra exchange records

rel = NaN
if check == 1
    G = gather(D)
    if me == 0
        Oref = evolve(O, gens, angs, truncation=strict)
        rel = norm(G - Oref) / norm(Oref)
    end
end

me == 0 && @printf("%d,%d,%d,%d,%d,1e-%g,%.3f,%d,%d,%.1f,%d,%.2e\n",
                   np, N, length(gens), r, window, strict_exp, t, nterms,
                   shipped, bytes / 1024, rebalances, rel)
MPI.Finalize()
