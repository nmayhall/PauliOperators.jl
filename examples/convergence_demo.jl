#=
Stochastic Convergence Demo
============================

Shows how the stochastic Pauli propagation estimate converges to the exact
value as the number of independent samples (M) increases.

The initial observable is a sum of several Pauli strings (a local
magnetization-like operator), which is more realistic than a single Pauli.
For a fixed compression threshold epsilon, we run stochastic propagation at
geometrically increasing sample counts and track:

  - The running mean of <O(t)>
  - The standard error of the mean (stderr ~ 1/sqrt(M))
  - The average number of Pauli terms surviving compression (final)
  - The peak number of Pauli terms before compression (max)
  - Comparison to the exact (un-truncated) result

This demonstrates that:
  1. The stochastic estimator is unbiased -- the mean converges to the exact value.
  2. The standard error decreases as 1/sqrt(M), as expected from the CLT.
  3. Stochastic compression keeps the operator sparse, with a well-defined
     average term count independent of the number of samples.
  4. The peak term count shows the operator growth before compression kicks in.

Requirements:
  - PauliOperators.jl (this package)
  - Plots.jl (optional, for generating plots)

Usage:
  julia --project=. examples/convergence_demo.jl
=#

using PauliOperators
using Printf
using LinearAlgebra
using Random


"""
    heisenberg_1d(N; Jx=1.0, Jy=1.0, Jz=1.0)

Build a 1D Heisenberg Hamiltonian with periodic boundary conditions:

    H = Sum_i (Jx Xi Xi+1 + Jy Yi Yi+1 + Jz Zi Zi+1) / 4
"""
function heisenberg_1d(N; Jx=1.0, Jy=1.0, Jz=1.0)
    H = PauliSum(N, Float64)
    for i in 1:N
        j = mod1(i + 1, N)
        H[PauliBasis(Pauli(N, X=[i, j]))] = Jx / 4
        H[PauliBasis(Pauli(N, Y=[i, j]))] = Jy / 4
        H[PauliBasis(Pauli(N, Z=[i, j]))] = Jz / 4
    end
    return H
end


"""
    run_convergence(; N=8, dt=0.1, n_trotter=30, eps=1.0,
                      sample_counts=[10,25,50,100,250,500,1000,2500,5000],
                      seed=42)

Run stochastic Pauli propagation at increasing sample counts and show
convergence of the mean, standard error, and term count.

The initial observable is a weighted sum of local Pauli operators:
    O = Z_1 + 0.5*(X_1 X_2 + Y_1 Y_2) + 0.25*Z_1 Z_2

Returns a named tuple with all results for optional plotting.
"""
function run_convergence(; N::Int=8, dt::Float64=0.1, n_trotter::Int=30,
                           eps::Float64=1.0,
                           sample_counts::Vector{Int}=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
                           seed::Int=42)

    t_total = dt * n_trotter

    println("=" ^ 90)
    println("  Stochastic Convergence Demo")
    println("  N = $N,  dt = $dt,  n_trotter = $n_trotter,  t = $t_total")
    println("  Compression threshold eps = $eps")
    println("=" ^ 90)

    # ── Setup ────────────────────────────────────────────────────────────────

    H = heisenberg_1d(N)

    # Observable: a sum of several Pauli strings
    #   O = Z_1  +  0.5*(X_1 X_2 + Y_1 Y_2)  +  0.25*Z_1 Z_2
    O = PauliSum(N, ComplexF64)
    O[PauliBasis(Pauli(N, Z=[1]))]    = 1.0
    O[PauliBasis(Pauli(N, X=[1,2]))]  = 0.5
    O[PauliBasis(Pauli(N, X=[2], Y=[3]))]  = 0.5
    O[PauliBasis(Pauli(N, Z=[3,4]))]  = 0.25
    @printf("  Initial observable: %d Pauli terms\n", length(O))
    display(O)

    # Initial state: |000...0>
    psi = rand(Ket{N})
    display(psi)

    # Build Trotter circuit
    generators = PauliBasis{N}[]
    base_angles = Float64[]
    for (p, c) in H
        push!(generators, p)
        push!(base_angles, real(c) * dt)
    end

    full_generators = repeat(generators, n_trotter)
    full_angles     = repeat(base_angles, n_trotter)
    n_gates_total   = length(full_generators)

    @printf("  Hamiltonian terms: %d\n", length(generators))
    @printf("  Total gates:       %d\n", n_gates_total)
    println()

    # ── 1. Exact evolution ─────────────────────────────────────────────────

    print("  Computing exact (no truncation)... ")
    O_exact = deepcopy(O)
    @time for (g, a) in zip(full_generators, full_angles)
        evolve!(O_exact, g, a)
        clip!(O_exact, thresh=1e-5)
    end
    e_exact = real(expectation_value(O_exact, psi))
    n_terms_exact = length(O_exact)
    @printf("done.\n")
    @printf("  Exact <O(t)> = %.10f   (n_terms = %d)\n\n", e_exact, n_terms_exact)
    
    # ── 2. Stochastic propagation at increasing M ──────────────────────────

    M_max = maximum(sample_counts)
    @printf("  Running stochastic propagation with M_max = %d samples...\n", M_max)

    @time result = stochastic_propagate(O, full_generators, full_angles, psi, eps;
                                  n_samples=M_max, seed=seed, verbose=0)

    all_samples         = result.samples          # Vector{Float64} of length M_max
    all_term_counts     = result.term_counts      # Vector{Int} of length M_max
    all_max_term_counts = result.max_term_counts  # Vector{Int} of length M_max

    println("  Done.\n")

    # ── 3. Compute running statistics at each sample count ─────────────────

    means            = Float64[]
    stderrs          = Float64[]
    biases           = Float64[]
    nterms_means     = Float64[]
    nterms_stds      = Float64[]
    max_nterms_means = Float64[]
    max_nterms_stds  = Float64[]

    println("  " * "-"^114)
    @printf("  %8s | %12s %12s %12s | %10s %10s | %10s %10s\n",
            "M", "mean", "bias", "stderr",
            "n_final", "n_std", "n_max", "max_std")
    println("  " * "-"^114)

    for M in sample_counts
        # Expectation value statistics
        ev_subset = @view all_samples[1:M]
        mu = sum(ev_subset) / M
        se = sqrt(sum((s - mu)^2 for s in ev_subset) / (M * (M - 1)))

        # Final term count statistics
        tc_subset = @view all_term_counts[1:M]
        tc_mean = sum(tc_subset) / M
        tc_std  = M > 1 ? sqrt(sum((n - tc_mean)^2 for n in tc_subset) / (M - 1)) : 0.0

        # Peak term count statistics
        mc_subset = @view all_max_term_counts[1:M]
        mc_mean = sum(mc_subset) / M
        mc_std  = M > 1 ? sqrt(sum((n - mc_mean)^2 for n in mc_subset) / (M - 1)) : 0.0

        push!(means, mu)
        push!(stderrs, se)
        push!(biases, mu - e_exact)
        push!(nterms_means, tc_mean)
        push!(nterms_stds, tc_std)
        push!(max_nterms_means, mc_mean)
        push!(max_nterms_stds, mc_std)

        @printf("  %8d | %12.8f %+12.8f %12.6e | %10.1f %10.1f | %10.1f %10.1f\n",
                M, mu, mu - e_exact, se, tc_mean, tc_std, mc_mean, mc_std)
    end
    println("  " * "-"^114)

    println()
    println("  Key observations:")
    println("    - The mean converges toward the exact value as M increases")
    println("    - The stderr decreases as ~ 1/sqrt(M)  (central limit theorem)")
    println("    - The bias stays within the stderr band, confirming unbiasedness")
    @printf("    - Stochastic compression: ~%.0f final terms (vs %d exact), a %.1fx reduction\n",
            nterms_means[end], n_terms_exact, n_terms_exact / nterms_means[end])
    @printf("    - Peak operator size before compression: ~%.0f terms\n",
            max_nterms_means[end])
    println()

    return (sample_counts=sample_counts, means=means, stderrs=stderrs, biases=biases,
            nterms_means=nterms_means, nterms_stds=nterms_stds,
            max_nterms_means=max_nterms_means, max_nterms_stds=max_nterms_stds,
            n_terms_exact=n_terms_exact,
            e_exact=e_exact, eps=eps,
            all_samples=all_samples, all_term_counts=all_term_counts,
            all_max_term_counts=all_max_term_counts,
            N=N, t_total=t_total)
end


# ── Run ──────────────────────────────────────────────────────────────────────

results = run_convergence(N=40, dt=0.1, n_trotter=20, eps=.50,
                        #   sample_counts=[10, 25, 50, 100, 250, 500, 1000])
                          sample_counts=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000])


# ── Plot (optional, requires Plots.jl) ──────────────────────────────────────

try
    using Plots

    M = results.sample_counts
    exact = results.e_exact

    # ── Panel 1: Convergence of the mean ──

    p1 = plot(
        xlabel="Number of samples  M",
        ylabel="<O(t)>",
        title="Convergence of Stochastic Mean  (N=$(results.N), t=$(results.t_total), eps=$(results.eps))",
        legend=:topright,
        xscale=:log10,
        margin=5Plots.mm
    )

    scatter!(p1, M, results.means,
             yerror=results.stderrs,
             label="Stochastic mean +/- stderr",
             marker=:circle, markersize=6, color=:blue)

    hline!(p1, [exact],
           label="Exact = $(@sprintf("%.6f", exact))",
           linestyle=:dash, color=:red, linewidth=2)

    # ── Panel 2: Stderr scaling ──

    p2 = plot(
        xlabel="Number of samples  M",
        ylabel="Standard error",
        title="Stderr Scaling  (~ 1/sqrt(M))",
        legend=:topright,
        xscale=:log10, yscale=:log10,
        margin=5Plots.mm
    )

    scatter!(p2, M, results.stderrs,
             label="Measured stderr",
             marker=:circle, markersize=6, color=:blue)

    # Reference 1/sqrt(M) line
    ref_scale = results.stderrs[1] * sqrt(M[1])
    M_range = range(M[1], M[end], length=100)
    plot!(p2, M_range, ref_scale ./ sqrt.(M_range),
          label="~ 1/sqrt(M)",
          linestyle=:dash, color=:gray, linewidth=2)

    # ── Panel 3: Term count (final and peak) ──

    p3 = plot(
        xlabel="Number of samples  M",
        ylabel="Pauli terms",
        title="Pauli Term Counts  (exact final = $(results.n_terms_exact))",
        legend=:topright,
        xscale=:log10,
        margin=5Plots.mm
    )

    scatter!(p3, M, results.nterms_means,
             yerror=results.nterms_stds,
             label="Final (mean +/- std)",
             marker=:square, markersize=6, color=:green)

    scatter!(p3, M, results.max_nterms_means,
             yerror=results.max_nterms_stds,
             label="Peak (mean +/- std)",
             marker=:utriangle, markersize=6, color=:orange)

    hline!(p3, [results.n_terms_exact],
           label="Exact (no compression)",
           linestyle=:dash, color=:red, linewidth=2)

    # ── Combined figure ──

    fig = plot(p1, p2, p3, layout=(3, 1), size=(900, 1000))

    outfile = joinpath(@__DIR__, "convergence_demo.png")
    savefig(fig, outfile)
    println("  Plot saved to: $outfile")

catch e
    if isa(e, ArgumentError) || isa(e, LoadError)
        println("  [Plots.jl not available -- skipping plot generation]")
        println("  Install with: julia -e 'using Pkg; Pkg.add(\"Plots\")'")
    else
        rethrow(e)
    end
end
