#=
Mean-Field Factorization vs Weight Truncation — 1D Heisenberg Dynamics
======================================================================

Compares Heisenberg-picture simulations of ⟨Z_i(t)⟩ for the 1D Heisenberg
model on N sites, starting from a product state (domain wall).

    1. Exact           — dense matrix exponentiation of H
    2. Trotter         — second-order Trotter, no truncation (sanity baseline)
    3. Trotter + WeightTruncation(k)       — drops all weight > k terms
    4. Trotter + MeanFieldTruncation(k, ψ) — expands weight > k terms in
                                              single-site fluctuations
                                              around |ψ⟩

MeanFieldTruncation preserves ⟨ψ|·|ψ⟩ *at the moment of truncation*, not the
final EV after subsequent evolution — so method 4 is not exact. But each
truncation step inflates the operator's L2 norm (and hence its variance on ψ),
which eventually limits accuracy. The variance panel visualizes this blow-up.

Usage:
    julia --project=examples examples/mean_field_demo.jl
=#

using PauliOperators
using LinearAlgebra
using Printf


"""
1D Heisenberg XXX Hamiltonian with open boundary conditions:
    H = J Σ_{i=1}^{N-1} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
"""
function heisenberg_1d(N::Int; J::Real=1.0)
    H = PauliSum(N, ComplexF64)
    for i in 1:(N-1)
        H[PauliBasis(Pauli(N; X=[i, i+1]))] = .1J + 0im
        H[PauliBasis(Pauli(N; Y=[i, i+1]))] = .1J + 0im
        H[PauliBasis(Pauli(N; Z=[i, i+1]))] = 2J + 0im
    end
    return H
end


"""
Exact dynamics via eigendecomposition of H: ⟨ψ(t)| O |ψ(t)⟩ and Var_ψ(t)(O).
"""
function exact_ev_curve(H::PauliSum{N}, O::PauliSum{N}, ψ::Ket{N},
                        times::AbstractVector) where {N}
    Hm  = Hermitian(Matrix(H))
    Om  = Matrix(O)
    Om2 = Om * Om
    ψv = zeros(ComplexF64, Int(2^N))
    ψv[Int(ψ.v) + 1] = 1.0

    F = eigen(Hm)
    λ, V = F.values, F.vectors
    c0 = V' * ψv

    ev  = zeros(Float64, length(times))
    var = zeros(Float64, length(times))
    for (k, t) in enumerate(times)
        ψt = V * (cis.(-t .* λ) .* c0)
        ev[k]  = real(ψt' * Om  * ψt)
        var[k] = real(ψt' * Om2 * ψt) - ev[k]^2
    end
    return ev, var
end


"""
Heisenberg-picture Trotter evolution of `O`. One second-order Trotter step
per recorded time (i.e. `times` must be uniformly spaced by `dt`).
Returns (ev_curve, var_curve, n_terms_curve).
"""
function trotter_ev_curve(H::PauliSum{N,T}, O::PauliSum{N,T}, ψ::Ket{N},
                          times::AbstractVector, dt::Real,
                          truncation::TruncationStrategy) where {N,T}
    generators, angles = trotterize(H, dt; n_trotter=1, order=2)
    Ot = deepcopy(O)

    ev      = zeros(Float64, length(times))
    var     = zeros(Float64, length(times))
    n_terms = zeros(Int,     length(times))
    ev[1]      = real(expectation_value(Ot, ψ))
    var[1]     = variance(Ot, ψ)
    n_terms[1] = length(Ot)
    l2 = norm(H)
    for step in 2:length(times)
        Ot = evolve(Ot, generators, angles; truncation=truncation)
        # mul!(Ot, l2 / norm(Ot))
        # @show norm(Ot)
        ev[step]      = real(expectation_value(Ot, ψ))
        var[step]     = variance(Ot, ψ)
        n_terms[step] = length(Ot)
    end
    return ev, var, n_terms
end


function main()
    # ── Setup ────────────────────────────────────────────────────────────────
    N      = 10 
    J      = 1.0
    dt     = 0.04
    T_max  = 4.0
    times  = collect(0.0:dt:T_max)
    k_max  = 3
    obs_site = 1

    H = heisenberg_1d(N; J=J)

    # Domain-wall initial state |111000⟩: qubits 1,2,3 in |1⟩
    ψ = Ket(N, Int128(0b000))
    ψ = Ket(N, Int128(0b1111100000))
    ψ = Ket(N, Int128(0b0101010101))

    # Observable O = Z on `obs_site`
    O = PauliSum(N, ComplexF64)
    O[PauliBasis(Pauli(N; X=[obs_site]))] = 1.0 + 0im
    O = PauliSum(Pauli(N; Z=[1,2]))
    # O += Pauli(N; Z=[3,])
    # mul!(O, 1/norm(O))

    println("=" ^ 82)
    println("  1D Heisenberg | N=$N sites, J=$J, ψ=|111000⟩, observable Z_$obs_site")
    println("  dt=$dt, T=$T_max, truncation weight k=$k_max")
    println("=" ^ 82)
    println()

    print("  exact (dense)...                        ")
    @time ev_exact, var_exact = exact_ev_curve(H, O, ψ, times)

    print("  trotter, no truncation...               ")
    @time ev_tr, var_tr, nt_tr = trotter_ev_curve(H, O, ψ, times, dt, CoeffTruncation(1e-4))

    print("  trotter + weight(k=$k_max)...                 ")
    @time ev_wt, var_wt, nt_wt = trotter_ev_curve(H, O, ψ, times, dt,
        CompositeTruncation(WeightTruncation(k_max), CoeffTruncation(1e-5)))

    print("  trotter + mean-field(k=$k_max, ψ)...          ")
    @time ev_mf, var_mf, nt_mf = trotter_ev_curve(H, O, ψ, times, dt,
        CompositeTruncation(MeanFieldTruncation(k_max, ψ), CoeffTruncation(1e-5)))

    println()
    @printf("  %5s  %+10s  %+10s  %+10s  %+10s    %5s %5s %5s\n",
            "t", "exact", "trotter", "weight($k_max)", "MF($k_max)",
            "#trot", "#wt", "#mf")
    println("  " * "-" ^ 82)
    for (i, t) in enumerate(times)
        @printf("  %5.2f  %+10.6f  %+10.6f  %+10.6f  %+10.6f    %5d %5d %5d\n",
                t, ev_exact[i], ev_tr[i], ev_wt[i], ev_mf[i],
                nt_tr[i], nt_wt[i], nt_mf[i])
    end
    println()

    err_tr = maximum(abs.(ev_tr .- ev_exact))
    err_wt = maximum(abs.(ev_wt .- ev_exact))
    err_mf = maximum(abs.(ev_mf .- ev_exact))
    @printf("  max |Δ| vs exact:  trotter-only        = %.4e\n",            err_tr)
    @printf("                     WeightTruncation(%d)  = %.4e\n",   k_max, err_wt)
    @printf("                     MeanFieldTruncation(%d) = %.4e\n", k_max, err_mf)
    println()

    return (times=times, exact=ev_exact, trotter=ev_tr,
            weight=ev_wt, meanfield=ev_mf,
            var_exact=var_exact, var_tr=var_tr,
            var_wt=var_wt, var_mf=var_mf,
            nt_tr=nt_tr, nt_wt=nt_wt, nt_mf=nt_mf,
            k=k_max, N=N, obs_site=obs_site)
end


results = main()


# ── Optional plot (requires Plots.jl) ────────────────────────────────────────
try
    using Plots

    p1 = plot(results.times, results.exact, label="exact",
              color=:black, lw=2.5, ls=:solid,
              xlabel="t", ylabel="⟨ψ| Z_$(results.obs_site)(t) |ψ⟩",
              title="1D Heisenberg XXX, N=$(results.N), k=$(results.k)",
              legend=:bottomleft)
    plot!(p1, results.times, results.trotter,
          label="Trotter (no trunc)", color=:gray, ls=:dash)
    plot!(p1, results.times, results.weight,
          label="WeightTruncation($(results.k))",
          color=:red,  lw=0, marker=:circle, ms=3, msw=0.3)
    plot!(p1, results.times, results.meanfield,
          label="MeanFieldTruncation($(results.k), ψ)",
          color=:blue, lw=2, marker=:square, ms=3, msw=0.3)

    p2 = plot(results.times, results.var_exact, label="exact",
              color=:black, lw=2.5, ls=:solid,
              xlabel="t", ylabel="Var_ψ[ O(t) ]",
              title="Operator variance on ψ",
              legend=:topleft)
    plot!(p2, results.times, results.var_tr,
          label="Trotter (no trunc)", color=:gray, ls=:dash)
    plot!(p2, results.times, results.var_wt,
          label="WeightTruncation($(results.k))",
          color=:red,  lw=2, marker=:circle, ms=3, msw=0.3)
    plot!(p2, results.times, results.var_mf,
          label="MeanFieldTruncation($(results.k), ψ)",
          color=:blue, lw=2, marker=:square, ms=3, msw=0.3)

    p3 = plot(results.times, results.nt_tr,
              label="Trotter (no trunc)", yscale=:log10,
              xlabel="t", ylabel="# Pauli terms",
              title="Operator size",
              color=:gray, ls=:dash)
    plot!(p3, results.times, results.nt_wt, label="WeightTruncation",
          color=:red,  marker=:circle, ms=3, msw=0.3)
    plot!(p3, results.times, results.nt_mf, label="MeanFieldTruncation",
          color=:blue, marker=:square, ms=3, msw=0.3)

    fig = plot(p1, p2, p3, layout=(3, 1), size=(900, 1000))
    outfile = joinpath(@__DIR__, "mean_field_demo.pdf")
    savefig(fig, outfile)
    println("  Plot saved to: $outfile")
catch e
    if isa(e, ArgumentError) || isa(e, LoadError)
        println("  [Plots.jl not available — skipping plot]")
    else
        rethrow(e)
    end
end
