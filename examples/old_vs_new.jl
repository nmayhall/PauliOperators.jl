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
# Run directly for a 2D Heisenberg demo (N = Lx·Ly qubits; 2D operator
# populations grow much faster than a chain, so start small):
#   julia --project --threads=8 examples/old_vs_new.jl 5 5 20 8 0.1 1e-10
#   (args: Lx Ly n_trotter window [damping alpha] [coeff thresh])

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

Along the way each path records its live Pauli count (the peak is a table
row) and the expectation value ⟨0…0|O(t)|0…0⟩ at every window boundary;
the traces are written to `<plotfile>.csv` and plotted to `plotfile` when
Plots.jl is available (`plotfile=nothing` disables both). The cheap
per-boundary recordings are included in every path's timing.

Returns a NamedTuple with per-path timings, GC statistics, peak/trace
data, and each engine's relative deviation from the old path's result
(zero at `window = 1` with deterministic truncation; cadence-sized
otherwise).
"""
function compare_old_vs_new(O::PauliSum{N}, gens::Vector{PauliBasis{N}},
                            angs::Vector{<:Real};
                            A::Union{Nothing,RankMap{N}}=nothing,
                            r::Union{Nothing,Int}=nothing,
                            nthreads::Int=Threads.nthreads(),
                            window::Int=1,
                            truncation::TruncationStrategy=NoTruncation(),
                            local_truncation::TruncationStrategy=NoTruncation(),
                            T::Type{<:Number}=Float64,
                            rebalance_threshold::Real=1.25,
                            warmup::Bool=true,
                            plotfile::Union{Nothing,String}="old_vs_new_expectation.png",
                            engine_kwargs...) where {N}
    if A === nothing
        r === nothing && (r = max(2, round(Int, log2(max(nthreads, 2))) + 4))
        A = rand(RankMap{N}, r)
    end
    L = length(gens)
    nwarm = min(2window, L)
    ψ0 = Ket(N, 0)
    boundaries = [i for i in 1:L if i % window == 0 || i == L]

    # ---- old path: Dict PauliSum, truncate! after every rotation ----
    # (this explicit loop is exactly what evolve(O, gens, angs; truncation)
    # does internally, unrolled so population and ⟨0|O|0⟩ can be watched)
    warmup && evolve(O, gens[1:nwarm], angs[1:nwarm]; truncation)
    peak_old = length(O)
    exp_old = Float64[]
    Oref = deepcopy(O)
    old = @timed for i in 1:L
        evolve!(Oref, gens[i], angs[i])
        truncate!(Oref, truncation)
        n = length(Oref)
        n > peak_old && (peak_old = n)
        (i % window == 0 || i == L) &&
            push!(exp_old, real(expectation_value(Oref, ψ0)))
    end

    # ---- new engine, serial and threaded ----
    # driven in window-sized chunks: every chunk ends on a merge boundary
    # the full circuit would also have, so the cadence is identical
    chunks = [lo:min(lo + window - 1, L) for lo in 1:window:L]
    ccircs = [compile(A, gens[rng], angs[rng]; window) for rng in chunks]
    wcirc = compile(A, gens[1:nwarm], angs[1:nwarm]; window)
    engines = NamedTuple[]
    for nt in unique((1, min(nthreads, Threads.nthreads())))
        build() = ShardedPauliSum(O, A; T, nthreads=nt, engine_kwargs...)
        warmup && evolve!(build(), wcirc; truncation, local_truncation,
                          counters=WindowCounters(length(wcirc.window_subgroups)),
                          rebalance_threshold)
        S = build()
        cnts = [WindowCounters(length(cc.window_subgroups)) for cc in ccircs]
        peak = length(S)
        exps = Float64[]
        st = @timed for (cc, cnt) in zip(ccircs, cnts)
            evolve!(S, cc; truncation, local_truncation,
                    counters=cnt, rebalance_threshold)
            n = length(S)
            n > peak && (peak = n)
            push!(exps, real(expectation_value(S, ψ0)))
        end
        push!(engines, (nthreads=nt, wall=st.time, gctime=st.gctime,
                        bytes=st.bytes, nterms=length(S), peak=peak, exps=exps,
                        early_merges=sum(c -> sum(c.early_merges), cnts),
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
    prow("peak terms", "%16d", [peak_old; [e.peak for e in engines]])
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

    res = (old=(wall=old.time, gctime=old.gctime, bytes=old.bytes,
                peak=peak_old, exps=exp_old, result=Oref),
           engines=engines, boundaries=boundaries)
    plotfile === nothing || save_expectation_plot(res; file=plotfile)
    return res
end

"""
    save_expectation_plot(res; file="old_vs_new_expectation.png")

Write the ⟨0…0|O(t)|0…0⟩ traces from a `compare_old_vs_new` result to
`<file>.csv`, and render them to `file` if Plots.jl is installed.
"""
function save_expectation_plot(res; file::String="old_vs_new_expectation.png")
    csv = first(splitext(file)) * ".csv"
    open(csv, "w") do io
        println(io, "rotation,old_dict," *
                    join(("sharded_nt$(e.nthreads)" for e in res.engines), ","))
        for (k, i) in enumerate(res.boundaries)
            println(io, "$i,$(res.old.exps[k])," *
                        join((e.exps[k] for e in res.engines), ","))
        end
    end
    println("expectation traces written to $csv")
    ok = try
        @eval import Plots
        true
    catch
        @warn "Plots.jl not installed — skipping the figure (CSV written)"
        false
    end
    ok || return csv
    Base.invokelatest() do
        p = Plots.plot(xlabel="rotation", ylabel="⟨0…0| O(t) |0…0⟩",
                       title="old vs new expectation trace", legend=:topright)
        Plots.plot!(p, res.boundaries, res.old.exps, label="old dict", lw=3)
        for e in res.engines
            Plots.plot!(p, res.boundaries, e.exps,
                        label="sharded nt=$(e.nthreads)", ls=:dash, lw=1.5)
        end
        Plots.savefig(p, file)
        println("expectation plot written to $file")
    end
    return file
end

# ---------------- demo when run as a script ----------------
if abspath(PROGRAM_FILE) == @__FILE__
    Lx     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
    Ly     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10 
    nsteps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50
    window = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
    alpha  = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.2
    thresh = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 1e-6

    N = Lx * Ly
    N <= 127 || error("Lx·Ly = $N exceeds the 127-qubit limit")
    site(ix, iy) = ix + (iy - 1) * Lx

    Random.seed!(1)
    H = PauliSum(N)
    for iy in 1:Ly, ix in 1:Lx, (dx, dy) in ((1, 0), (0, 1))
        jx, jy = ix + dx, iy + dy
        (jx <= Lx && jy <= Ly) || continue
        i, j = site(ix, iy), site(jx, jy)
        H[PauliBasis(Pauli(N, X=[i, j]))] = 1.0
        H[PauliBasis(Pauli(N, Y=[i, j]))] = 0.9
        H[PauliBasis(Pauli(N, Z=[i, j]))] = 1.1
    end
    gens, angs = trotterize(H, 0.05, n_trotter=nsteps, order=1)

    O = PauliSum(N, Float64)
    O[PauliBasis(Pauli(N, Z=[site((Lx + 1) ÷ 2, (Ly + 1) ÷ 2)]))] = 1.0  # central Z probe

    @printf("2D Heisenberg %dx%d (N=%d), %d rotations, window=%d, trunc=WeightDamped(%.3g, %.1e)\n\n",
            Lx, Ly, N, length(gens), window, alpha, thresh)
    compare_old_vs_new(O, gens, angs;
                       window,
                    #    truncation=CoeffTruncation(thresh),
                       truncation=WeightDampedTruncation(alpha, thresh),
                       min_capacity=1 << 14, append_factor=2.0)
end
