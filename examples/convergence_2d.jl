#=
2D Heisenberg — Systematic Truncation Convergence Study (no dense reference)
============================================================================

Heisenberg-picture evolution of a single-site observable under a 2D Heisenberg
Hamiltonian at system sizes beyond exact diagonalization. For each of three
truncation strategies we sweep a tightness grid and check convergence by
tightening the threshold:

    1. CoeffTruncation(ε)           — drop terms with |c| ≤ ε
    2. WeightTruncation(k)          — drop terms with Pauli weight > k
    3. MeanFieldTruncation(k, ψ)    — expand weight > k terms in single-site
                                       fluctuations around |ψ⟩

Convergence diagnostics (no exact curve required):

    • within-strategy self-convergence:
          max_t |⟨O⟩_param(t) − ⟨O⟩_tightest(t)|   vs   param / cost
      Tells you whether the strategy has stopped moving as you tighten.

    • cross-strategy agreement:
          max_t |⟨O⟩_tightest_A(t) − ⟨O⟩_tightest_B(t)|
      If the three tightest runs agree, that's strong circumstantial evidence
      they've converged to the true answer (different biases).

Usage:
    julia --project=examples examples/convergence_2d.jl
=#

using PauliOperators
using LinearAlgebra
using Printf

# Bring _apply! into scope so we can add methods to it here (not exported).
import PauliOperators: _apply!


# ── Experimental: CoeffTruncation with mean-field redirect ──────────────────
#
# Like CoeffTruncation(ε), but when a term c·P is dropped (|c| ≤ ε), instead
# of setting it to zero we redirect it to its expectation value on |ψ⟩:
#
#     c · P   →   c · ⟨ψ|P|ψ⟩ · I
#
# For a computational-basis reference, ⟨ψ|P|ψ⟩ is 0 unless P is diagonal
# (no X/Y content), in which case it is ±1 — essentially free to evaluate.
# Preserves ⟨ψ|O|ψ⟩ exactly at the moment of truncation (diff is absorbed
# into the identity coefficient).
struct CoeffTruncationMF{N} <: TruncationStrategy
    thresh::Float64
    reference::Ket{N}
end

function _apply!(O::PauliSum{N,T}, s::CoeffTruncationMF{N}) where {N,T}
    id_key = PauliBasis{N}(Int128(0), Int128(0))
    id_accum = zero(T)
    for (p, c) in collect(O)        # collect to iterate over a snapshot
        p == id_key && continue
        if abs(c) <= s.thresh
            id_accum += c * expectation_value(p, s.reference)
            delete!(O, p)
        end
    end
    if id_accum != zero(T)
        O[id_key] = get(O, id_key, zero(T)) + id_accum
    end
    return O
end


# ── Experimental: WeightTruncation with mean-field redirect ─────────────────
#
# Like WeightTruncation(k), but terms with weight > k are redirected to their
# expectation value on |ψ⟩ rather than dropped:
#
#     c · P   →   c · ⟨ψ|P|ψ⟩ · I      (for weight(P) > k)
#
# Preserves ⟨ψ|O|ψ⟩ exactly at the moment of truncation.
struct WeightTruncationMF{N} <: TruncationStrategy
    max_weight::Int
    reference::Ket{N}
end

function _apply!(O::PauliSum{N,T}, s::WeightTruncationMF{N}) where {N,T}
    id_key = PauliBasis{N}(Int128(0), Int128(0))
    id_accum = zero(T)
    for (p, c) in collect(O)
        p == id_key && continue
        if weight(p) > s.max_weight
            id_accum += c * expectation_value(p, s.reference)
            delete!(O, p)
        end
    end
    if id_accum != zero(T)
        O[id_key] = get(O, id_key, zero(T)) + id_accum
    end
    return O
end


# ── Lattice helpers ──────────────────────────────────────────────────────────

"""
Return the list of nearest-neighbor (i, j) bonds (1-based) for an Lx×Ly open
rectangular lattice, row-major indexing (site at (r, c) → (r-1)*Lx + c).
"""
function nn_bonds_2d(Lx::Int, Ly::Int)
    idx(r, c) = (r - 1) * Lx + c
    bonds = Tuple{Int,Int}[]
    for r in 1:Ly, c in 1:Lx
        if c < Lx
            push!(bonds, (idx(r, c), idx(r, c + 1)))   # horizontal
        end
        if r < Ly
            push!(bonds, (idx(r, c), idx(r + 1, c)))   # vertical
        end
    end
    return bonds
end

"""
2D Heisenberg XXZ Hamiltonian on an Lx×Ly open lattice.

    H = Σ_⟨ij⟩  Jxy (X_i X_j + Y_i Y_j) + Jz Z_i Z_j
"""
function heisenberg_2d(Lx::Int, Ly::Int; Jxy::Real=1.0, Jz::Real=1.0)
    N = Lx * Ly
    H = PauliSum(N, ComplexF64)
    for (i, j) in nn_bonds_2d(Lx, Ly)
        H[PauliBasis(Pauli(N; X=[i, j]))] = Jxy + 0im
        H[PauliBasis(Pauli(N; Y=[i, j]))] = Jxy + 0im
        H[PauliBasis(Pauli(N; Z=[i, j]))] = Jz  + 0im
    end
    return H
end


# ── Trotter driver ──────────────────────────────────────────────────────────

function trotter_ev_curve(H::PauliSum{N,T}, O::PauliSum{N,T}, ψ::Ket{N},
                          times::AbstractVector, dt::Real,
                          truncation::TruncationStrategy) where {N,T}
    generators, angles = trotterize(H, dt; n_trotter=1, order=2)
    Ot = deepcopy(O)

    ev      = zeros(Float64, length(times))
    n_terms = zeros(Int,     length(times))
    ev[1]      = real(expectation_value(Ot, ψ))
    n_terms[1] = length(Ot)
    for step in 2:length(times)
        Ot = evolve(Ot, generators, angles; truncation=truncation)
        ev[step]      = real(expectation_value(Ot, ψ))
        n_terms[step] = length(Ot)
    end
    return ev, n_terms
end


# ── Sweep definition ─────────────────────────────────────────────────────────

struct SweepEntry
    name::String                    # "coeff", "weight", "mf"
    label::String                   # human-readable legend label
    tightness::Float64              # monotone tightness ordinate (larger = tighter)
    strategy::TruncationStrategy
end

"""
Build the full sweep. Within each strategy, entries are ordered loose → tight.
A small floor CoeffTruncation(1e-10) is composed onto weight/MF so the
weight/MF-tightest run is not polluted by numerical chaff.
"""
function build_sweep(ψ::Ket{N}) where {N}
    sweep = SweepEntry[]

    # CoeffTruncation: tightness = -log10(ε)
    for ε in (1e-3, 1e-4, 1e-5)
        push!(sweep, SweepEntry("coeff", "Coeff(ε=$(ε))", -log10(ε),
                                CoeffTruncation(ε)))
    end

    # CoeffTruncationMF (experimental — defined above): tightness = -log10(ε)
    for ε in (1e-3, 1e-4, 1e-5)
        push!(sweep, SweepEntry("coeff_mf", "CoeffMF(ε=$(ε))", -log10(ε),
                                CoeffTruncationMF(ε, ψ)))
    end

    # WeightTruncation: tightness = k
    for k in 1:4
        push!(sweep, SweepEntry("weight", "Weight(k=$k)", float(k),
            CompositeTruncation(WeightTruncation(k), CoeffTruncation(1e-8))))
    end

    # WeightTruncationMF (experimental — defined above): tightness = k
    for k in 1:4
        push!(sweep, SweepEntry("weight_mf", "WeightMF(k=$k)", float(k),
            CompositeTruncation(WeightTruncationMF(k, ψ), CoeffTruncation(1e-8))))
    end

    # MeanFieldTruncation: tightness = k
    for k in 1:4
        push!(sweep, SweepEntry("mf", "MF(k=$k)", float(k),
            CompositeTruncation(MeanFieldTruncation(k, ψ), CoeffTruncation(1e-8))))
    end

    return sweep
end


# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    # Lattice — tweak freely; 4×4 (N=16) is well beyond exact reach (2^16=65536)
    # but still light for Heisenberg-picture Trotter at modest truncation.
    Lx, Ly = 4, 4
    N      = Lx * Ly
    Jxy    = .10
    Jz     = 1.0
    dt     = 0.02
    T_max  = 4.0
    times  = collect(0.0:dt:T_max)

    H = heisenberg_2d(Lx, Ly; Jxy=Jxy, Jz=Jz)

    # Néel-type product reference state (checkerboard on row-major idx)
    bits = UInt128(0)
    for r in 1:Ly, c in 1:Lx
        site = (r - 1) * Lx + c
        if isodd(r + c)
            bits |= (UInt128(1) << (site - 1))
        end
    end
    ψ = Ket(N, Int128(bits))

    # Observable: Z on a central site
    obs_site = (Ly ÷ 2) * Lx + (Lx ÷ 2 + 1)
    O = PauliSum(Pauli(N; Z=[obs_site]))

    println("=" ^ 82)
    println("  2D Heisenberg | $(Lx)×$(Ly) lattice (N=$N), Jxy=$Jxy, Jz=$Jz")
    println("  observable: Z_$(obs_site) on Néel reference, dt=$dt, T=$T_max")
    println("  (no dense reference — convergence by tightening thresholds)")
    println("=" ^ 82)

    sweep = build_sweep(ψ)

    avg_n   = zeros(Float64, length(sweep))
    curves  = Vector{Vector{Float64}}(undef, length(sweep))
    nterms  = Vector{Vector{Int}}(undef,    length(sweep))

    println("\n  Sweep:")
    for (i, s) in enumerate(sweep)
        @printf("    %-18s  ", s.label)
        t0 = time()
        ev, nt = trotter_ev_curve(H, O, ψ, times, dt, s.strategy)
        dt_s = time() - t0
        curves[i] = ev
        nterms[i] = nt
        avg_n[i]  = sum(nt) / length(nt)
        @printf("avg #terms = %10.1f   (%.2fs)\n", avg_n[i], dt_s)
    end

    # ── Within-strategy self-convergence ─────────────────────────────────────
    # For each strategy, compare every run to the tightest run within that
    # strategy. The tightest entry itself registers 0 by construction.
    self_err = zeros(Float64, length(sweep))
    tightest_idx = Dict{String,Int}()
    for name in ("coeff", "coeff_mf", "weight", "weight_mf", "mf")
        idxs = findall(s -> s.name == name, sweep)
        tight = idxs[argmax([sweep[i].tightness for i in idxs])]
        tightest_idx[name] = tight
        ref = curves[tight]
        for i in idxs
            self_err[i] = maximum(abs.(curves[i] .- ref))
        end
    end

    println("\n  Within-strategy self-convergence (vs tightest of same strategy):")
    for name in ("coeff", "coeff_mf", "weight", "weight_mf", "mf")
        idxs = findall(s -> s.name == name, sweep)
        println("    $(uppercase(name)):")
        for i in idxs
            marker = i == tightest_idx[name] ? "  <-- tightest" : ""
            @printf("      %-18s  avg #terms = %10.1f   max|Δ vs tightest| = %.3e%s\n",
                    sweep[i].label, avg_n[i], self_err[i], marker)
        end
    end

    # ── Cross-strategy agreement (tightest vs tightest) ─────────────────────
    println("\n  Cross-strategy agreement at tightest settings:")
    names = ("coeff", "coeff_mf", "weight", "weight_mf", "mf")
    for a in 1:length(names), b in (a+1):length(names)
        ia = tightest_idx[names[a]]
        ib = tightest_idx[names[b]]
        d  = maximum(abs.(curves[ia] .- curves[ib]))
        @printf("    %-8s vs %-8s   max|Δ| = %.3e\n",
                sweep[ia].label, sweep[ib].label, d)
    end
    println()

    return (times=times, sweep=sweep, curves=curves, nterms=nterms,
            avg_n=avg_n, self_err=self_err, tightest_idx=tightest_idx,
            Lx=Lx, Ly=Ly, obs_site=obs_site)
end


results = main()


# ── Optional plot (requires Plots.jl) ────────────────────────────────────────
try
    using Plots

    color_of  = Dict("coeff" => :darkgreen, "coeff_mf"  => :purple,
                     "weight" => :red,       "weight_mf" => :orange,
                     "mf"     => :blue)
    marker_of = Dict("coeff" => :diamond,    "coeff_mf"  => :utriangle,
                     "weight" => :circle,    "weight_mf" => :dtriangle,
                     "mf"     => :square)

    # ---- p1a/p1b/p1c/p1d: ⟨O(t)⟩ — one subplot per strategy ----
    title_of = Dict("coeff"     => "CoeffTruncation",
                    "coeff_mf"  => "CoeffTruncationMF (experimental)",
                    "weight"    => "WeightTruncation",
                    "weight_mf" => "WeightTruncationMF (experimental)",
                    "mf"        => "MeanFieldTruncation")
    strategy_panels = Any[]
    for name in ("coeff", "coeff_mf", "weight", "weight_mf", "mf")
        p = plot(xlabel="t", ylabel="⟨ψ| Z_$(results.obs_site)(t) |ψ⟩",
                 title=title_of[name],
                 legend=:outerright, legendfontsize=6,
                 ylims=(0.9, 1.1))
        idxs = findall(s -> s.name == name, results.sweep)
        tightness = [results.sweep[i].tightness for i in idxs]
        tmin, tmax = extrema(tightness)
        for i in idxs
            α = tmax > tmin ? 0.2 + 0.8 * (results.sweep[i].tightness - tmin) / (tmax - tmin) : 1.0
            plot!(p, results.times, results.curves[i],
                  label=results.sweep[i].label,
                  color=color_of[name], lw=1.0, alpha=α)
        end
        push!(strategy_panels, p)
    end

    # ---- p3: operator size vs t for the tightest run of each strategy ----
    p3 = plot(xlabel="t", ylabel="# Pauli terms",
              title="Operator size over time (tightest run per strategy)",
              yscale=:log10, legend=:bottomright)
    for name in ("coeff", "coeff_mf", "weight", "weight_mf", "mf")
        i = results.tightest_idx[name]
        plot!(p3, results.times, results.nterms[i],
              label=results.sweep[i].label,
              color=color_of[name], lw=1.2,
              marker=marker_of[name], ms=3, msw=0.3)
    end

    fig = plot(strategy_panels..., p3, layout=(6, 1), size=(1000, 2000))
    outfile = joinpath(@__DIR__, "convergence_2d.pdf")
    savefig(fig, outfile)
    println("  Plot saved to: $outfile")
catch e
    if isa(e, ArgumentError) || isa(e, LoadError)
        println("  [Plots.jl not available — skipping plot]")
    else
        rethrow(e)
    end
end
