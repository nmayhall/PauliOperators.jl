# Single-machine MPI speedup benchmark: serial vs distributed rank sweep.
#
# Answers "how much faster is this code with n MPI ranks on THIS machine?"
# in three measurements:
#
#   1. serial baseline        — plain `evolve` (the thing to beat)
#   2. hardware ceiling       — n INDEPENDENT concurrent serial runs (zero
#                               communication). Their slowdown vs running
#                               alone is pure memory-system contention, so
#                               n × t_solo / t_concurrent is the maximum
#                               speedup ANY n-way split can achieve here.
#   3. MPI sweep              — actual DistributedPauliSum runs
#
# Setup (once, on the target machine):
#     git checkout distributed2 && julia --project -e 'using Pkg; Pkg.instantiate()'
# Run (expect ~15-30 min with defaults; needs roughly (nmax+1) GB of RAM):
#     julia --project examples/mpi_speedup_benchmark.jl
# Custom workload / rank list:
#     julia --project examples/mpi_speedup_benchmark.jl N n_trotter dt clip_exp window np1,np2,...
#     e.g.  julia --project examples/mpi_speedup_benchmark.jl 30 10 0.15 9 16 1,4,8,12,16
# (On P/E-core Apple Silicon, include a P-core-only count like 12: merge
# barriers run at the slowest core's pace, so all-16 may lose to 12.)
using MPI
using PauliOperators
using LinearAlgebra
using Printf
using Random

# ---------------- mode/parameter parsing ----------------
mode = (!isempty(ARGS) && startswith(ARGS[1], "__")) ? popfirst!(ARGS) : "__driver__"
N        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
nt       = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
dt       = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.15
clip_exp = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 9.0
window   = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 16
nps      = length(ARGS) >= 6 ? parse.(Int, split(ARGS[6], ',')) : [1, 4, 8, 16]
clip = 10.0^-clip_exp

# Nonintegrable NN + NNN chain: spreads fast, so the population actually
# gets large enough for parallelism to matter.
function workload(N, nt, dt)
    H = PauliSum(N)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, X=[i, i+1]))] = 1.0
        H[PauliBasis(Pauli(N, Y=[i, i+1]))] = 0.9
        H[PauliBasis(Pauli(N, Z=[i, i+1]))] = 1.1
    end
    for i in 1:N-2
        H[PauliBasis(Pauli(N, Z=[i, i+2]))] = 0.7
        H[PauliBasis(Pauli(N, X=[i, i+2]))] = 0.5
    end
    O = PauliSum(N)
    O[PauliBasis(Pauli(N, Z=[N ÷ 2]))] = 1.0 + 0im
    gens, angs = trotterize(H, dt; n_trotter=nt, order=2)
    return O, gens, angs
end

# ---------------- child: one serial run ----------------
if mode == "__serial__"
    O, gens, angs = workload(N, nt, dt)
    t = @elapsed Of = evolve(O, gens, angs, truncation=CoeffTruncation(clip))
    @printf("RESULT serial t=%.2f terms=%d norm=%.10f\n", t, length(Of), norm(Of))
    exit(0)
end

# ---------------- child: one distributed run (under mpiexec) ----------------
if mode == "__mpi__"
    MPI.Init()
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    O, gens, angs = workload(N, nt, dt)
    Random.seed!(60)
    A = rand(RankMap{N}, trailing_zeros(np) + 4)     # 16 bins per rank
    D = DistributedPauliSum(O, A, comm)
    circ = compile(D, gens, angs; window)
    counters = PropagationCounters()
    MPI.Barrier(comm)
    t = @elapsed evolve!(D, circ, truncation=CoeffTruncation(clip),
                         local_truncation=CoeffTruncation(clip / 10),
                         rebalance_threshold=1.3; counters)
    MPI.Barrier(comm)
    bytes = MPI.Allreduce(sum(counters.bytes_per_merge), +, comm)
    nterms = length(D)                                # collective: all ranks call
    nrm = norm(D)                                     # collective: all ranks call
    if MPI.Comm_rank(comm) == 0
        @printf("RESULT mpi np=%d t=%.2f terms=%d norm=%.10f mb=%.0f\n",
                np, t, nterms, nrm, bytes / 1024^2)
    end
    MPI.Finalize()
    exit(0)
end

# ---------------- driver ----------------
script = @__FILE__
params = collect(string.((N, nt, dt, clip_exp, window)))   # tuple: no int->float promotion
childcmd(mode, extra=String[]) =
    `$(Base.julia_cmd()) --project=$(Base.active_project()) $script $mode $params $extra`

parsefield(out, key) = begin
    m = match(Regex("$key=([0-9.eE+-]+)"), out)
    m === nothing && error("child produced no '$key=' (output was:\n$out)")
    parse(Float64, m.captures[1])
end

nmax = maximum(nps)
println("workload: N=$N NNN-chain, $nt second-order Trotter steps, dt=$dt, clip 1e-$clip_exp, window=$window")
println("machine:  $(Sys.cpu_info() |> length) CPU threads, $(round(Sys.total_memory()/2^30, digits=0)) GB RAM")
println()

# 1. serial baseline
println("[1/3] serial baseline...")
out = read(childcmd("__serial__"), String)
t_serial = parsefield(out, "t")
terms_serial = Int(parsefield(out, "terms"))
norm_serial = parsefield(out, "norm")
@printf("      %.1fs, %d terms\n\n", t_serial, terms_serial)

# 2. hardware ceiling: nmax independent concurrent serial runs
println("[2/3] hardware ceiling: $nmax independent concurrent serial runs...")
times = Vector{Float64}(undef, nmax)
@sync for i in 1:nmax
    Threads.@spawn times[i] = parsefield(read(childcmd("__serial__"), String), "t")
end
t_conc = sum(times) / nmax
ceiling = nmax * t_serial / t_conc
@printf("      %.1fs each (vs %.1fs alone) -> max possible %d-way speedup ~ %.1fx\n\n",
        t_conc, t_serial, nmax, ceiling)

# 3. MPI sweep
println("[3/3] MPI sweep...")
exe = MPI.mpiexec()
rows = []
for np in nps
    mpiout = read(`$exe -n $np $(childcmd("__mpi__"))`, String)
    t = parsefield(mpiout, "t")
    push!(rows, (np, t, Int(parsefield(mpiout, "terms")), parsefield(mpiout, "norm"),
                 parsefield(mpiout, "mb")))
    @printf("      np=%-3d %.1fs  (%.2fx)\n", np, t, t_serial / t)
end

println("\n================ summary ================")
@printf("serial: %.1fs, %d terms | ceiling for %d-way split: %.1fx\n",
        t_serial, terms_serial, nmax, ceiling)
@printf("%-5s %-9s %-9s %-12s %-10s %-8s\n", "np", "time(s)", "speedup", "% of ceiling", "shipped", "norm ok")
for (np, t, terms, nrm, mb) in rows
    ideal = np * t_serial / t_conc          # contention-adjusted ideal for this np
    @printf("%-5d %-9.1f %-9.2f %-12.0f %-10s %s\n",
            np, t, t_serial / t, 100 * (t_serial / t) / ideal,
            @sprintf("%.0f MB", mb),
            abs(nrm - norm_serial) < 1e-3 ? "yes" : "CHECK (cadence err $(round(abs(nrm-norm_serial), sigdigits=2)))")
end
println("\nnotes: 'norm ok' compares to the serial norm (windowed runs differ at the")
println("cadence-error scale, ~1e-5, which is expected). If the ceiling itself is far")
println("below $nmax, the workload is memory-bandwidth-bound and no communication")
println("scheme can recover the difference on a single machine.")
