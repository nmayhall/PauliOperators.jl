# Multinode (across-node) Sparse Pauli Dynamics, with on-node multithreading.
#
# Two levels of parallelism:
#   - across nodes: one worker per node, terms sharded by hash (memory scales).
#   - inside a node: each worker's rotation is threaded over its local terms.
#
# Run locally, e.g.:  julia --project examples/distributed_evolution.jl
# or across nodes by adding SSH/SLURM workers before `using PauliOperators`.
#
# Each worker owns the Pauli terms that hash to it and stores an ordinary local
# PauliSum. Every Trotter rotation cos-scales local terms and routes the new
# sin-branch terms (G*p) to their owner workers, then clips with a global
# threshold. Nothing is gathered on the master, so the sparse Pauli sum can exceed
# a single node's memory (target: a 10x10x10 = 1000-qubit Heisenberg lattice).

using Distributed
if nprocs() == 1
    # one worker per node, each with several Julia threads (--threads=N).
    # On a cluster, replace with addprocs(hosts; exeflags="--project=... --threads=N").
    addprocs(4; exeflags="--threads=4")
end
using PauliOperators
@info "workers / threads-per-worker" workers() =
    [remotecall_fetch(Threads.nthreads, p) for p in workers()]

# ---- build a 3D Heisenberg Trotter step as a list of 2-local generators ----
# Lattice: Lx x Ly x Lz spins, nearest-neighbour XX+YY+ZZ couplings.
function heisenberg_generators(Lx, Ly, Lz)
    N = Lx * Ly * Lz
    idx(i, j, k) = ((k - 1) * Ly + (j - 1)) * Lx + i    # 1-based site index
    gens = PauliBasis{N,PauliOperators.uinttype(N)}[]
    for k in 1:Lz, j in 1:Ly, i in 1:Lx
        s = idx(i, j, k)
        neighbours = Int[]
        i < Lx && push!(neighbours, idx(i + 1, j, k))
        j < Ly && push!(neighbours, idx(i, j + 1, k))
        k < Lz && push!(neighbours, idx(i, j, k + 1))
        for t in neighbours
            push!(gens, PauliBasis(Pauli(N; X=[s, t])))
            push!(gens, PauliBasis(Pauli(N; Y=[s, t])))
            push!(gens, PauliBasis(Pauli(N; Z=[s, t])))
        end
    end
    return N, gens
end

Lx, Ly, Lz = 10, 10, 10                      # 1000 qubits
N, gens = heisenberg_generators(Lx, Ly, Lz)
dt = 0.05
angles = fill(dt, length(gens))

@info "system" N n_generators = length(gens) workers = workers()

# Heisenberg-picture evolution of a local observable O(0) = Z_1 Z_2.
O0 = PauliSum(N)
O0[PauliBasis(Pauli(N; Z=[1, 2]))] = 1.0 + 0.0im

dO = distribute(O0; workers=workers())

n_steps = 20
thresh = 1e-6                                # SPD coefficient truncation
for step in 1:n_steps
    evolve!(dO, gens, angles; truncation_thresh=thresh)
    if step % 5 == 0
        println("step $step:  terms = $(length(dO))   |O|₂ = $(round(opnorm2(dO), digits=6))")
        println("           shard split = ", sharded_summary(dO))
    end
end

# ⟨0…0| O(t) |0…0⟩  (gather only the tiny reduction, not the operator)
Olocal = collect_paulisum(dO)                # gather for the final expectation only
ev = expectation_value(Olocal, Ket(N, 0))
println("\n<0|O(t)|0> = ", ev)

destroy!(dO)
