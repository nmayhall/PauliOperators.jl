# Old (Dict-based serial evolve) vs new (sharded engine) comparison.
#
# `compare_old_vs_new` runs the same rotation sequence through three paths:
#   1. the original `evolve(O, gens, angs; truncation)` — Dict storage,
#      per-rotation truncate! (unchanged by the sharded-engine work),
#   2. the sharded engine with nthreads=1,
#   3. the sharded engine with nthreads=n,
# and reports wall time, GC time, allocation, and result agreement.
#
# The old path truncates after EVERY rotation, which equals the engine at
# window=1 (bit-exact for deterministic truncations). window>1 is the
# engine's natural operating point; the reported deviation vs old is then
# the truncation-cadence difference, not an error.
#
# Expectations: at small populations (≲10³ live terms) the old path wins —
# the engine pays fixed per-rotation shard costs. The engine takes over as
# the population grows (no hashing, no per-term allocation, sequential
# sweeps), and the old path's GC time column shows where the Dict churn
# goes at scale.
#
# Run directly for a Heisenberg-chain demo:
#   julia --project --threads=8 examples/old_vs_new.jl 32 20 8 0.2 1e-10
#   (args: N n_trotter window [damping alpha] [coeff thresh])

using PauliOperators
using LinearAlgebra
using Printf
using Random

"""
    compare_old_vs_new(O, gens, angs;
                       A=nothing, r=nothing, nthreads=Threads.nthreads(),
                       window=8, truncation=NoTruncation(),
                       local_truncation=NoTruncation(), T=Float64,
                       rebalance_threshold=1.25, warmup=true,
                       engine_kwargs...)

Evolve `O` under `(gens, angs)` through the original Dict-based serial
`evolve` and through the sharded engine (serial and `nthreads`-threaded),
timing all three. `A` (or its row count `r`) controls the engine's rank
map; by default a random map with ~16 shards per thread is drawn.

Returns a NamedTuple with per-path timings, GC statistics, and each
engine's relative deviation from the old path's result (zero at
`window = 1` with deterministic truncation; cadence-sized otherwise).
"""
function compare_old_vs_new(O::PauliSum{N}, gens::Vector{PauliBasis{N}},
                            angs::Vector{<:Real};
                            A::Union{Nothing,RankMap{N}}=nothing,
                            r::Union{Nothing,Int}=nothing,
                            nthreads::Int=Threads.nthreads(),
                            window::Int=8,
                            truncation::TruncationStrategy=NoTruncation(),
                            local_truncation::TruncationStrategy=NoTruncation(),
                            T::Type{<:Number}=Float64,
                            rebalance_threshold::Real=1.25,
                            warmup::Bool=true,
                            engine_kwargs...) where {N}
    if A === nothing
        r === nothing && (r = max(2, round(Int, log2(max(nthreads, 2))) + 4))
        A = rand(RankMap{N}, r)
    end
    circ = compile(A, gens, angs; window)
    nw = length(circ.window_subgroups)
    nwarm = min(2window, length(gens))

    # ---- old path: Dict PauliSum, truncate! after every rotation ----
    warmup && evolve(O, gens[1:nwarm], angs[1:nwarm]; truncation)
    old = @timed evolve(O, gens, angs; truncation)
    Oref = old.value

    # ---- new engine, serial and threaded ----
    engines = NamedTuple[]
    for nt in unique((1, min(nthreads, Threads.nthreads())))
        build() = ShardedPauliSum(O, A; T, nthreads=nt, engine_kwargs...)
        if warmup
            wcirc = compile(A, gens[1:nwarm], angs[1:nwarm]; window)
            evolve!(build(), wcirc; truncation, local_truncation,
                    counters=WindowCounters(length(wcirc.window_subgroups)),
                    rebalance_threshold)
        end
        S = build()
        cnt = WindowCounters(nw)
        st = @timed evolve!(S, circ; truncation, local_truncation,
                            counters=cnt, rebalance_threshold)
        push!(engines, (nthreads=nt, wall=st.time, gctime=st.gctime,
                        bytes=st.bytes, counters=cnt, nterms=length(S),
                        early_merges=sum(cnt.early_merges),
                        reldiff=norm(PauliSum(S) - Oref) / max(norm(Oref), eps())))
    end

    cols = ["old dict"; ["sharded nt=$(e.nthreads)" for e in engines]]
    @printf("%-24s", "")
    foreach(c -> @printf("%16s", c), cols)
    println()
    function prow(label, fmt, vals)
        @printf("%-24s", label)
        f = Printf.Format(fmt)
        foreach(v -> Printf.format(stdout, f, v), vals)
        println()
    end
    prow("wall time (s)", "%16.4f", [old.time; [e.wall for e in engines]])
    prow("GC time (s)", "%16.4f", [old.gctime; [e.gctime for e in engines]])
    prow("allocated (MB)", "%16.1f", [old.bytes / 1e6; [e.bytes / 1e6 for e in engines]])
    prow("final terms", "%16d", [length(Oref); [e.nterms for e in engines]])
    prow("rel diff vs old", "%16.2e", [0.0; [e.reldiff for e in engines]])
    for e in engines
        @printf("speedup vs old (nt=%d): %.2fx\n", e.nthreads, old.time / e.wall)
    end
    window > 1 &&
        println("note: window=$window — the engine truncates every $window rotations, " *
                "the old path every rotation; rel diff is that cadence, not error " *
                "(set window=1 for a bit-exact check)")
    any(e -> e.early_merges > 0, engines) &&
        println("note: early merges fired; raise min_capacity/append_factor")

    return (old=(wall=old.time, gctime=old.gctime, bytes=old.bytes, result=Oref),
            engines=engines)
end

# ---------------- demo when run as a script ----------------
if abspath(PROGRAM_FILE) == @__FILE__
    N      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32
    nsteps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
    window = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
    alpha  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.2
    thresh = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1e-10

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

    @printf("Heisenberg chain N=%d, %d rotations, window=%d, trunc=WeightDamped(%.3g, %.1e)\n\n",
            N, length(gens), window, alpha, thresh)
    compare_old_vs_new(O, gens, angs;
                       window,
                       truncation=WeightDampedTruncation(alpha, thresh),
                       min_capacity=1 << 14, append_factor=2.0)
end
