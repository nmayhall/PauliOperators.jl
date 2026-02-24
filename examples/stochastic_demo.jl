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
  - Plots.jl (install in your global environment: ]add Plots)

Usage:
  julia --project=. examples/stochastic_demo.jl
=#

using PauliOperators
using Printf
using LinearAlgebra
using Random

# ── Build 1D Heisenberg Hamiltonian ──────────────────────────────────────────
#
#   H = Σᵢ (Jx Xᵢ Xᵢ₊₁ + Jy Yᵢ Yᵢ₊₁ + Jz Zᵢ Zᵢ₊₁) / 4
#
# with periodic boundary conditions.

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

# ── Parameters ───────────────────────────────────────────────────────────────

N = 8                           # qubits
dt = 0.1                        # Trotter step size
n_trotter = 30                  # number of Trotter steps  (total time t = 1.0)
n_samples = 1000                 # stochastic samples per threshold
# thresholds = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
thresholds = [1e-2, 5e-3, 1e-3]

println("=" ^ 78)
println("  Stochastic Pauli Propagation Demo")
println("  N = $N,  dt = $dt,  n_trotter = $n_trotter,  t = $(dt * n_trotter)")
println("  n_samples = $n_samples")
println("=" ^ 78)

# ── Setup ────────────────────────────────────────────────────────────────────

H = heisenberg_1d(N)

# Observable: Z on qubit 1
O = PauliSum(N, ComplexF64)
O[PauliBasis(Pauli(N, Z=[1]))] = 1.0

# Initial stabilizer state: |000...0⟩
psi = Ket{N}(0)

# Extract Trotter generators and angles from the Hamiltonian
#   Each term c_j P_j of H becomes a rotation exp(i c_j dt / 2 P_j) ... exp(...)
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

# ── 1. Exact evolution (no truncation) ───────────────────────────────────────

print("  Computing exact (no truncation)... ")
O_exact = deepcopy(O)
for (g, a) in zip(full_generators, full_angles)
    evolve!(O_exact, g, a)
end
e_exact = real(expectation_value(O_exact, psi))
@printf("done.  n_terms = %d\n", length(O_exact))
@printf("  Exact ⟨Z₁(t)⟩ = %.10f\n\n", e_exact)

# ── 2. Hard truncation at each threshold ─────────────────────────────────────

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
    @printf("    ε = %.0e :  ⟨Z₁⟩ = %12.8f   bias = %+12.8f   n_terms = %d\n",
            eps, ev, ev - e_exact, length(O_hard))
end
println()

# ── 3. Stochastic compression at each threshold ─────────────────────────────

println("  Stochastic compression (stochastic_clip!, M=$n_samples):")
e_stoch_mean   = Float64[]
e_stoch_stderr = Float64[]
for eps in thresholds
    result = stochastic_propagate(O, full_generators, full_angles, psi, eps;
                                  n_samples=n_samples, seed=42, verbose=0)
    push!(e_stoch_mean, result.mean)
    push!(e_stoch_stderr, result.stderr)
    bias = result.mean - e_exact
    @printf("    ε = %.0e :  ⟨Z₁⟩ = %12.8f   bias = %+12.8f   stderr = %.2e\n",
            eps, result.mean, bias, result.stderr)
end
println()

# ── Summary table ────────────────────────────────────────────────────────────

println("=" ^ 78)
println("  Summary:  exact = $(@sprintf("%.10f", e_exact))")
println("-" ^ 78)
@printf("  %10s │ %12s  %12s │ %12s  %12s  %10s\n",
        "threshold", "hard_trunc", "hard_bias", "stoch_mean", "stoch_bias", "stderr")
println("  " * "-"^10 * " ┼ " * "-"^27 * " ┼ " * "-"^37)
for (i, eps) in enumerate(thresholds)
    hard_bias  = e_hard[i] - e_exact
    stoch_bias = e_stoch_mean[i] - e_exact
    @printf("  %10.0e │ %12.8f  %+12.8f │ %12.8f  %+12.8f  %10.2e\n",
            eps, e_hard[i], hard_bias, e_stoch_mean[i], stoch_bias, e_stoch_stderr[i])
end
println("=" ^ 78)
println()
println("  Key observation:")
println("    - Hard truncation bias is SYSTEMATIC (same sign, grows with ε)")
println("    - Stochastic bias is STATISTICAL (fluctuates around zero, ~ stderr)")
println()

# ── Plot ─────────────────────────────────────────────────────────────────────

try
    using Plots

    hard_bias_abs  = abs.(e_hard .- e_exact)
    stoch_bias_abs = abs.(e_stoch_mean .- e_exact)

    p = plot(
        xscale=:log10, yscale=:log10,
        xlabel="Threshold  ε",
        ylabel="|Bias|  or  Stderr",
        title="Bias Comparison: Hard Truncation vs Stochastic  (N=$N, t=$(dt*n_trotter))",
        legend=:topleft,
        size=(800, 500),
        margin=5Plots.mm
    )

    # Hard truncation |bias|
    scatter!(p, thresholds, hard_bias_abs,
             label="Hard truncation |bias|",
             marker=:circle, markersize=8, color=:red)

    # Stochastic |bias| with error bars
    scatter!(p, thresholds, stoch_bias_abs,
             label="Stochastic |bias|",
             marker=:square, markersize=8, color=:blue)

    # Stochastic stderr
    scatter!(p, thresholds, e_stoch_stderr,
             label="Stochastic stderr",
             marker=:diamond, markersize=6, color=:green, alpha=0.7)

    # Guide line: bias ~ ε
    eps_range = [minimum(thresholds), maximum(thresholds)]
    plot!(p, eps_range, eps_range .* 0.5,
          label="~ ε (guide)", linestyle=:dash, color=:gray, linewidth=1)

    outfile = joinpath(@__DIR__, "stochastic_bias_comparison.png")
    savefig(p, outfile)
    println("  Plot saved to: $outfile")

catch e
    if isa(e, ArgumentError) || isa(e, LoadError)
        println("  [Plots.jl not available — skipping plot generation]")
        println("  Install with: julia -e 'using Pkg; Pkg.add(\"Plots\")'")
    else
        rethrow(e)
    end
end
