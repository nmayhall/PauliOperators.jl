#=
Stochastic Pauli Propagation Demo
==================================

Demonstrates that stochastic compression (Russian Roulette) is unbiased,
while hard truncation (coeff_clip!) introduces systematic bias.

We propagate the observable Z₁ through a Trotterized Heisenberg time evolution
and compare three approaches:

  1. Exact: no truncation (feasible for small N)
  2. Hard truncation: coeff_clip! after every gate (biased)
  3. Stochastic compression: stochastic_clip! after every gate (unbiased)

Requirements:
  - PauliOperators.jl (this package)
  - Plots.jl (optional, for generating plots)

Usage:
  julia --project=. examples/stochastic_demo.jl
=#

using PauliOperators
using Printf
using LinearAlgebra
using Random


"""
    heisenberg_1d(N; Jx=1.0, Jy=1.0, Jz=1.0)

Build a 1D Heisenberg Hamiltonian with periodic boundary conditions:

    H = Σᵢ (Jx Xᵢ Xᵢ₊₁ + Jy Yᵢ Yᵢ₊₁ + Jz Zᵢ Zᵢ₊₁) / 4
"""
function heisenberg_1d(N; Jx=1.0, Jy=1.0, Jz=1.0)
    H = PauliSum(N, Float64)
    for i in 1:N
        j = mod1(i + 1, N)   # periodic
        H[PauliBasis(Pauli(N, X=[i, j]))] = Jx / 4
        H[PauliBasis(Pauli(N, Y=[i, j]))] = Jy / 4
        H[PauliBasis(Pauli(N, Z=[i, j]))] = Jz / 4
    end
    return H
end


"""
    run_demo(; N=8, dt=0.1, n_trotter=30, n_samples=1000,
              thresholds=[1e-2, 5e-3, 1e-3])

Run the stochastic Pauli propagation demo.

Compares exact, hard-truncated, and stochastic Heisenberg-picture evolution
of ⟨Z₁(t)⟩ for a 1D Heisenberg model starting from |000...0⟩.
"""
function run_demo(; N::Int=8, dt::Float64=0.1, n_trotter::Int=30,
                    n_samples::Int=10, thresholds=[1e-2, 5e-3, 1e-3])

    t_total = dt * n_trotter

    println("=" ^ 90)
    println("  Stochastic Pauli Propagation Demo")
    println("  N = $N,  dt = $dt,  n_trotter = $n_trotter,  t = $t_total")
    println("  n_samples = $n_samples")
    println("=" ^ 90)

    # ── Setup ────────────────────────────────────────────────────────────────

    H = heisenberg_1d(N)

    # Observable: Z on qubit 1
    O = PauliSum(N, ComplexF64)
    O[PauliBasis(Pauli(N, Z=[1]))] = 1.0

    # Initial stabilizer state: |000...0⟩
    psi = Ket{N}(0)

    # Extract Trotter generators and angles from the Hamiltonian
    generators = PauliBasis{N}[]
    base_angles = Float64[]
    for (p, c) in H
        push!(generators, p)
        push!(base_angles, real(c) * dt)
    end
    n_gates_per_step = length(generators)

    # Full Trotter circuit: repeat the single-step generators n_trotter times
    full_generators = repeat(generators, n_trotter)
    full_angles     = repeat(base_angles, n_trotter)
    n_gates_total   = length(full_generators)

    @printf("  Hamiltonian terms: %d\n", n_gates_per_step)
    @printf("  Total gates:       %d\n", n_gates_total)
    println()

    # ── 1. Exact evolution (no truncation) ───────────────────────────────────

    print("  Computing exact (no truncation)... ")
    O_exact = deepcopy(O)
    for (g, a) in zip(full_generators, full_angles)
        evolve!(O_exact, g, a)
    end
    e_exact = real(expectation_value(O_exact, psi))
    n_terms_exact = length(O_exact)
    @printf("done.  n_terms = %d\n", n_terms_exact)
    @printf("  Exact <Z_1(t)> = %.10f\n\n", e_exact)

    # ── 2. Hard truncation at each threshold ─────────────────────────────────

    println("  Hard truncation (coeff_clip!):")
    e_hard = Float64[]
    n_terms_hard = Int[]
    for eps in thresholds
        O_hard = deepcopy(O)
        for (g, a) in zip(full_generators, full_angles)
            evolve!(O_hard, g, a)
            coeff_clip!(O_hard; thresh=eps)
        end
        ev = real(expectation_value(O_hard, psi))
        push!(e_hard, ev)
        push!(n_terms_hard, length(O_hard))
        @printf("    eps = %.0e :  <Z_1> = %12.8f   bias = %+12.8f   n_terms = %d\n",
                eps, ev, ev - e_exact, length(O_hard))
    end
    println()

    # ── 3. Stochastic compression at each threshold ──────────────────────────

    println("  Stochastic compression (stochastic_clip!, M=$n_samples):")
    e_stoch_mean    = Float64[]
    e_stoch_stderr  = Float64[]
    n_stoch_mean    = Float64[]
    n_stoch_std     = Float64[]
    for eps in thresholds
        result = stochastic_propagate(O, full_generators, full_angles, psi, eps*100;
                                      n_samples=n_samples*100, seed=42, verbose=0)
        push!(e_stoch_mean, result.mean)
        push!(e_stoch_stderr, result.stderr)
        push!(n_stoch_mean, result.n_terms_mean)
        push!(n_stoch_std, result.n_terms_std)
        bias = result.mean - e_exact
        @printf("    eps = %.0e :  <Z_1> = %12.8f   bias = %+12.8f   stderr = %.2e   n_terms = %.0f +/- %.0f\n",
                eps, result.mean, bias, result.stderr, result.n_terms_mean, result.n_terms_std)
    end
    println()

    # ── Summary table ────────────────────────────────────────────────────────

    println("=" ^ 90)
    println("  Summary:  exact = $(@sprintf("%.10f", e_exact))   (n_terms = $n_terms_exact)")
    println("-" ^ 90)
    @printf("  %8s | %12s %12s %8s | %12s %12s %10s %14s\n",
            "thresh", "hard_trunc", "hard_bias", "n_terms",
            "stoch_mean", "stoch_bias", "stderr", "n_terms")
    println("  " * "-"^8 * " | " * "-"^35 * " | " * "-"^51)
    for (i, eps) in enumerate(thresholds)
        hard_bias  = e_hard[i] - e_exact
        stoch_bias = e_stoch_mean[i] - e_exact
        @printf("  %8.0e | %12.8f %+12.8f %8d | %12.8f %+12.8f %10.2e %8.0f +/- %4.0f\n",
                eps, e_hard[i], hard_bias, n_terms_hard[i],
                e_stoch_mean[i], stoch_bias, e_stoch_stderr[i],
                n_stoch_mean[i], n_stoch_std[i])
    end
    println("=" ^ 90)
    println()
    println("  Key observations:")
    println("    - Hard truncation bias is SYSTEMATIC (same sign, grows with threshold)")
    println("    - Stochastic bias is STATISTICAL (fluctuates around zero, ~ stderr)")
    println()

    return (thresholds=thresholds, e_exact=e_exact,
            e_hard=e_hard, n_terms_hard=n_terms_hard,
            e_stoch_mean=e_stoch_mean, e_stoch_stderr=e_stoch_stderr,
            n_stoch_mean=n_stoch_mean, n_stoch_std=n_stoch_std,
            N=N, t_total=t_total)
end


# ── Run ──────────────────────────────────────────────────────────────────────

results = run_demo(N=8, dt=0.1, n_trotter=30, n_samples=1000,
                   thresholds=[1e-2, 5e-3, 1e-3])


# ── Plot (optional, requires Plots.jl) ──────────────────────────────────────

try
    using Plots

    hard_bias_abs  = abs.(results.e_hard .- results.e_exact)
    stoch_bias_abs = abs.(results.e_stoch_mean .- results.e_exact)

    p = plot(
        xscale=:log10, yscale=:log10,
        xlabel="Threshold  eps",
        ylabel="|Bias|  or  Stderr",
        title="Bias Comparison: Hard Truncation vs Stochastic  (N=$(results.N), t=$(results.t_total))",
        legend=:topleft,
        size=(800, 500),
        margin=5Plots.mm
    )

    scatter!(p, results.thresholds, hard_bias_abs,
             label="Hard truncation |bias|",
             marker=:circle, markersize=8, color=:red)

    scatter!(p, results.thresholds, stoch_bias_abs,
             label="Stochastic |bias|",
             marker=:square, markersize=8, color=:blue)

    scatter!(p, results.thresholds, results.e_stoch_stderr,
             label="Stochastic stderr",
             marker=:diamond, markersize=6, color=:green, alpha=0.7)

    eps_range = [minimum(results.thresholds), maximum(results.thresholds)]
    plot!(p, eps_range, eps_range .* 0.5,
          label="~ eps (guide)", linestyle=:dash, color=:gray, linewidth=1)

    outfile = joinpath(@__DIR__, "stochastic_bias_comparison.png")
    savefig(p, outfile)
    println("  Plot saved to: $outfile")

catch e
    if isa(e, ArgumentError) || isa(e, LoadError)
        println("  [Plots.jl not available -- skipping plot generation]")
        println("  Install with: julia -e 'using Pkg; Pkg.add(\"Plots\")'")
    else
        rethrow(e)
    end
end
