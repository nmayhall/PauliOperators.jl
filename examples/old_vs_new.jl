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
row) and two observable traces:

- the infinite-temperature autocorrelation
  `C(t) = tr(O(t)·O(0)) / tr(O(0)²)` — the standard operator-spreading /
  transport observable (a coefficient lookup, since Paulis are
  trace-orthogonal), and
- `⟨ψ|O(t)|ψ⟩` for the computational-basis ket `state` (default `|0…0⟩`;
  NOTE: the fully polarized state is an exact eigenstate of any
  Heisenberg/XXZ model, so its trace is constant up to Trotter error —
  pass e.g. a Néel ket for nontrivial dynamics).

Traces are written to `<plotfile>.csv` and plotted to `plotfile` when
Plots.jl is available (`plotfile=nothing` disables both). The cheap
recordings are included in every path's timing.

`record_every` sets the sampling stride in rotations (default: every
window boundary). Mid-Trotter-step samples oscillate — only some of the
non-commuting layers have been applied yet — so for a smooth physical
trace pass one full Trotter step (e.g. `length(gens) ÷ n_trotter`),
ideally a multiple of `window` so samples land exactly on merge
boundaries (the engine can only be observed there; off-multiple strides
snap to the next boundary).

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
                            record_every::Union{Nothing,Int}=nothing,
                            state::Union{Nothing,Ket{N}}=nothing,
                            engine_kwargs...) where {N}
    if A === nothing
        r === nothing && (r = max(2, round(Int, log2(max(nthreads, 2))) + 4))
        A = rand(RankMap{N}, r)
    end
    L = length(gens)
    nwarm = min(2window, L)
    ψ0 = state === nothing ? Ket(N, 0) : state
    refterms = collect(O)                      # O(0), for the autocorrelation
    norm0sq = sum(abs2(c) for (_, c) in refterms)
    stride = record_every === nothing ? window : record_every
    # sample points: window boundaries (the engine is only observable on
    # merged state), thinned to the first boundary at/after each stride
    boundaries = [i for i in 1:L if i % window == 0 || i == L]
    recpts = Int[]
    for b in boundaries
        (isempty(recpts) ? b : b - recpts[end]) >= stride && push!(recpts, b)
    end
    (isempty(recpts) || recpts[end] != L) && push!(recpts, L)
    recset = Set(recpts)

    # ---- old path: Dict PauliSum, truncate! after every rotation ----
    # (this explicit loop is exactly what evolve(O, gens, angs; truncation)
    # does internally, unrolled so population and ⟨0|O|0⟩ can be watched)
    warmup && evolve(O, gens[1:nwarm], angs[1:nwarm]; truncation)
    peak_old = length(O)
    exp_old = Float64[]
    corr_old = Float64[]
    Oref = deepcopy(O)
    old = @timed for i in 1:L
        evolve!(Oref, gens[i], angs[i])
        truncate!(Oref, truncation)
        n = length(Oref)
        n > peak_old && (peak_old = n)
        if i in recset
            push!(exp_old, real(expectation_value(Oref, ψ0)))
            push!(corr_old, real(sum(conj(c0) * get(Oref, p, zero(c0))
                                     for (p, c0) in refterms)) / norm0sq)
        end
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
        corrs = Float64[]
        st = @timed for (rng, cc, cnt) in zip(chunks, ccircs, cnts)
            evolve!(S, cc; truncation, local_truncation,
                    counters=cnt, rebalance_threshold)
            n = length(S)
            n > peak && (peak = n)
            if last(rng) in recset
                push!(exps, real(expectation_value(S, ψ0)))
                push!(corrs, real(sum(conj(c0) * S[p] for (p, c0) in refterms)) / norm0sq)
            end
        end
        push!(engines, (nthreads=nt, wall=st.time, gctime=st.gctime,
                        bytes=st.bytes, nterms=length(S), peak=peak,
                        exps=exps, corrs=corrs,
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
                peak=peak_old, exps=exp_old, corrs=corr_old, result=Oref),
           engines=engines, boundaries=recpts)
    plotfile === nothing || save_expectation_plot(res; file=plotfile)
    return res
end

"""
    save_expectation_plot(res; file="old_vs_new_expectation.png")

Write the autocorrelation C(t) and ⟨ψ|O(t)|ψ⟩ traces from a
`compare_old_vs_new` result to `<file>.csv`, and render a two-panel figure
to `file` if Plots.jl is installed.
"""
function save_expectation_plot(res; file::String="old_vs_new_expectation.png")
    csv = first(splitext(file)) * ".csv"
    open(csv, "w") do io
        println(io, "rotation,old_corr,old_expval," *
                    join(("sharded_nt$(e.nthreads)_corr,sharded_nt$(e.nthreads)_expval"
                          for e in res.engines), ","))
        for (k, i) in enumerate(res.boundaries)
            println(io, "$i,$(res.old.corrs[k]),$(res.old.exps[k])," *
                        join(("$(e.corrs[k]),$(e.exps[k])" for e in res.engines), ","))
        end
    end
    println("observable traces written to $csv")
    ok = try
        @eval import Plots
        true
    catch
        @warn "Plots.jl not installed — skipping the figure (CSV written)"
        false
    end
    ok || return csv
    Base.invokelatest() do
        pc = Plots.plot(ylabel="tr(O(t)·O(0)) / tr(O(0)²)",
                        title="infinite-temperature autocorrelation", legend=:topright)
        Plots.plot!(pc, res.boundaries, res.old.corrs, label="old dict", lw=3)
        pe = Plots.plot(xlabel="rotation", ylabel="⟨ψ| O(t) |ψ⟩",
                        title="state expectation value", legend=:topright)
        Plots.plot!(pe, res.boundaries, res.old.exps, label="old dict", lw=3)
        for e in res.engines
            Plots.plot!(pc, res.boundaries, e.corrs,
                        label="sharded nt=$(e.nthreads)", ls=:dash, lw=1.5)
            Plots.plot!(pe, res.boundaries, e.exps,
                        label="sharded nt=$(e.nthreads)", ls=:dash, lw=1.5)
        end
        p = Plots.plot(pc, pe, layout=(2, 1), size=(700, 640))
        Plots.savefig(p, file)
        println("observable plot written to $file")
    end
    return file
end

# ---------------- demo when run as a script ----------------
if abspath(PROGRAM_FILE) == @__FILE__
    Lx     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
    Ly     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10 
    nsteps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100
    window = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10
    alpha  = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.1
    thresh = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 1e-7

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
        H[PauliBasis(Pauli(N, Y=[i, j]))] = 1.0
        H[PauliBasis(Pauli(N, Z=[i, j]))] = 0.1
    end
    gens, angs = trotterize(H, 0.05, n_trotter=nsteps, order=1)

    O = PauliSum(N, Float64)
    O[PauliBasis(Pauli(N, Z=[site((Lx + 1) ÷ 2, (Ly + 1) ÷ 2)]))] = 1.0  # central Z probe

    # Néel (checkerboard) ket: NOT an eigenstate, so ⟨ψ|O(t)|ψ⟩ has real
    # dynamics (the polarized |0…0⟩ is an XXZ eigenstate — constant trace)
    ψ_neel = Ket(N, sum(Int128(1) << (site(ix, iy) - 1)
                        for iy in 1:Ly for ix in 1:Lx if isodd(ix + iy);
                        init=Int128(0)))

    trunc = WeightDampedTruncation(alpha,thresh)
    trunc = CoeffTruncation(thresh)
    trunc = WeightTruncation(3)
    @show trunc
    @printf("2D Heisenberg %dx%d (N=%d), %d rotations, window=%d\n\n",
            Lx, Ly, N, length(gens), window)
    compare_old_vs_new(O, gens, angs;
                       window,
                    #    truncation=CoeffTruncation(thresh),
                       truncation=trunc,
                       state=ψ_neel,
                       record_every=length(gens) ÷ nsteps,   # sample only completed Trotter steps
                       min_capacity=1 << 14, append_factor=2.0)
end
