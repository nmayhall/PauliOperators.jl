# Serial vs threaded comparison for the sharded engine.
#
# `compare_serial_threaded` runs the SAME circuit through two engines in one
# session — one built with nthreads=1 (serial driver, no barriers) and one
# with nthreads=n (worker pool) — and reports wall time per phase, speedup,
# and result agreement. Works because the engine's thread count is a
# construction parameter, independent of Julia's --threads (which only sets
# the ceiling).
#
# Run directly for a Heisenberg-chain demo:
#   julia --project --threads=8 examples/serial_vs_threaded.jl 24 6 8
#   (args: N n_trotter window [damping alpha] [coeff thresh])
#
# Interpreting the result: the threaded engine only wins once the swept
# population is large enough that per-rotation work dwarfs the barrier
# cost (~10⁴-10⁵ live terms on a laptop). Below that, expect speedup < 1 —
# lower `alpha`/`thresh` to grow the population and watch the crossover.
# On heterogeneous CPUs (Apple Silicon P+E cores) every barrier waits for
# the slowest core: use --threads = number of PERFORMANCE cores.
#
# Or use the function with your own model:
#   include("examples/serial_vs_threaded.jl")
#   compare_serial_threaded(O, A, gens, angs; window=8, truncation=...)

using PauliOperators
using LinearAlgebra
using Printf
using Random

"""
    compare_serial_threaded(O, A, gens, angs;
                            nthreads=Threads.nthreads(), window=8,
                            truncation=NoTruncation(),
                            local_truncation=NoTruncation(),
                            T=Float64, rebalance_threshold=1.25,
                            warmup=true, engine_kwargs...)

Evolve `O` under the rotation sequence `(gens, angs)` twice — on a serial
(`nthreads = 1`) and a threaded (`nthreads`) sharded engine with identical
map `A`, window, capacities, and truncation — and print a comparison.

Returns a NamedTuple with both runs' timings/counters, the speedup, and the
relative norm difference of the final operators. At `window = 1` the two
results are bit-exact (`reldiff == 0`); at `window > 1` they agree to
floating-point reduction order unless capacity-forced early merges fire
(reported — raise `min_capacity`/`append_factor` to eliminate them).

Any `engine_kwargs` (e.g. `min_capacity`, `capacity_factor`,
`append_factor`) are forwarded to both `ShardedPauliSum` constructors.
"""
function compare_serial_threaded(O::PauliSum{N}, A::RankMap{N},
                                 gens::Vector{PauliBasis{N}}, angs::Vector{<:Real};
                                 nthreads::Int=Threads.nthreads(),
                                 window::Int=8,
                                 truncation::TruncationStrategy=NoTruncation(),
                                 local_truncation::TruncationStrategy=NoTruncation(),
                                 T::Type{<:Number}=Float64,
                                 rebalance_threshold::Real=1.25,
                                 warmup::Bool=true,
                                 engine_kwargs...) where {N}
    nthreads <= Threads.nthreads() &&  nthreads >= 1 ||
        error("nthreads=$nthreads not available (Julia has $(Threads.nthreads()); " *
              "restart with --threads=$nthreads or more)")
    circ = compile(A, gens, angs; window)
    nw = length(circ.window_subgroups)

    runs = NamedTuple[]
    for nt in (1, nthreads)
        build() = ShardedPauliSum(O, A; T, nthreads=nt, engine_kwargs...)
        if warmup   # JIT + capacity growth outside the timed run
            nwarm = min(2window, length(gens))
            wcirc = compile(A, gens[1:nwarm], angs[1:nwarm]; window)
            # identical argument types to the timed call, or the timed
            # region pays the JIT for its own kwarg specialization
            evolve!(build(), wcirc; truncation, local_truncation,
                    counters=WindowCounters(length(wcirc.window_subgroups)),
                    rebalance_threshold)
        end
        S = build()
        cnt = WindowCounters(nw)
        wall = @elapsed evolve!(S, circ; truncation, local_truncation,
                                counters=cnt, rebalance_threshold)
        push!(runs, (engine=S, wall=wall, counters=cnt,
                     rotate=sum(cnt.t_rotate), merge=sum(cnt.t_merge),
                     created=sum(cnt.terms_created),
                     early_merges=sum(cnt.early_merges),
                     allocd_steady=sum(cnt.allocd[2:end]),
                     nterms=length(S)))
    end
    serial, threaded = runs

    Os = PauliSum(serial.engine)
    Ot = PauliSum(threaded.engine)
    reldiff = norm(Os - Ot) / max(norm(Os), eps())
    speedup = serial.wall / threaded.wall

    @printf("%-22s %14s %14s\n", "", "serial (nt=1)", "threaded (nt=$nthreads)")
    @printf("%-22s %14.4f %14.4f\n", "wall time (s)", serial.wall, threaded.wall)
    @printf("%-22s %14.4f %14.4f\n", "rotate time (s)", serial.rotate, threaded.rotate)
    @printf("%-22s %14.4f %14.4f\n", "merge time (s)", serial.merge, threaded.merge)
    @printf("%-22s %14d %14d\n", "terms created", serial.created, threaded.created)
    @printf("%-22s %14d %14d\n", "final terms", serial.nterms, threaded.nterms)
    @printf("%-22s %14d %14d\n", "early merges", serial.early_merges, threaded.early_merges)
    @printf("%-22s %14d %14d   <- must be 0\n", "steady-state alloc (B)",
            serial.allocd_steady, threaded.allocd_steady)
    @printf("speedup: %.2fx    relative result difference: %.3e%s\n",
            speedup, reldiff,
            window == 1 && reldiff == 0 ? "   (bit-exact)" : "")
    (serial.early_merges > 0 || threaded.early_merges > 0) &&
        println("note: early merges fired — cadences differ between the runs; " *
                "raise min_capacity/append_factor for a like-for-like comparison")

    return (serial=serial, threaded=threaded, speedup=speedup, reldiff=reldiff)
end

# ---------------- demo when run as a script ----------------
if abspath(PROGRAM_FILE) == @__FILE__
    N      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 24
    nsteps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6
    window = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
    alpha  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.25
    thresh = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1e-9

    Random.seed!(1)
    H = PauliSum(N)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, X=[i, i+1]))] = 1.0
        H[PauliBasis(Pauli(N, Y=[i, i+1]))] = 0.9
        H[PauliBasis(Pauli(N, Z=[i, i+1]))] = 1.1
    end
    gens, angs = trotterize(H, 0.05, n_trotter=nsteps, order=2)

    O = PauliSum(N, Float64)
    O[PauliBasis(Pauli(N, Z=[N ÷ 2]))] = 1.0

    r = max(2, round(Int, log2(max(Threads.nthreads(), 2))) + 4)
    A = rand(RankMap{N}, r)

    @printf("Heisenberg chain N=%d, %d rotations, %d shards, window=%d, --threads=%d\n\n",
            N, length(gens), 1 << r, window, Threads.nthreads())
    compare_serial_threaded(O, A, gens, angs;
                            window,
                            truncation=WeightDampedTruncation(alpha, thresh),
                            min_capacity=1 << 14, append_factor=2.0)
end
