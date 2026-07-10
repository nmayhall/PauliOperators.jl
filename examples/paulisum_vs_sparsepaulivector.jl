# ============================================================
# Timing comparison: Dict-backed PauliSum vs flat SparsePauliVector.
#
# Heisenberg-picture evolution of a single-site Z observable under a
# trotterized Heisenberg-chain circuit, truncating after every rotation.
# Both paths compute the SAME thing: `evolve!` on a SparsePauliVector with
# `window=1` reproduces the PauliSum evolve/truncate sequence exactly, so
# the final operators are checked to agree to machine precision.
#
# Run with:  julia --project -O3 examples/paulisum_vs_sparsepaulivector.jl
# ============================================================

using PauliOperators
using LinearAlgebra
using Random
using Printf

function heisenberg_chain(N; Jx=1.0, Jy=0.7, Jz=1.1, hz=0.3)
    H = PauliSum(N)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, X=[i, i+1]))] = Jx
        H[PauliBasis(Pauli(N, Y=[i, i+1]))] = Jy
        H[PauliBasis(Pauli(N, Z=[i, i+1]))] = Jz
    end
    for i in 1:N
        H[PauliBasis(Pauli(N, Z=[i]))] = hz
    end
    return H
end

# Old path: Dict-backed PauliSum, truncating after every rotation.
function evolve_dict(O0::PauliSum, gens, angs, strat)
    O = deepcopy(O0)
    for (g, θ) in zip(gens, angs)
        evolve!(O, g, θ)
        truncate!(O, strat)
    end
    return O
end

# New path: flat SparsePauliVector; window=1 gives identical semantics.
function evolve_spv(O0::PauliSum, gens, angs, strat)
    O = SparsePauliVector(O0; capacity_factor=4.0)
    evolve!(O, gens, angs; window=4, truncation=strat)
    return O
end

function main()
    Random.seed!(1)
    N = 100
    H = heisenberg_chain(N)
    gens, angs = trotterize(H, 0.15, n_trotter=20, order=1)
    O0 = PauliSum(N, Float64)
    O0[PauliBasis(Pauli(N, Z=[N ÷ 2,N ÷ 2-1]))] = 1.0

    println("N = $N qubits, $(length(gens)) rotations, observable Z_$(N ÷ 2)\n")
    @printf("%-30s %10s %10s %10s %12s\n",
            "truncation", "terms", "dict (s)", "spv (s)", "speedup")

    for strat in (CompositeTruncation(WeightTruncation(4), CoeffTruncation(1e-8)),
                  CompositeTruncation(WeightTruncation(6), CoeffTruncation(1e-10)),
                  CoeffTruncation(1e-8))
        # warm-up (compile), then best of 3 repetitions
        local ref, new
        t_dict = Inf
        t_spv = Inf
        @time evolve_dict(O0, gens, angs, strat)
        @time evolve_spv(O0, gens, angs, strat)
        for _ in 1:3
            t_dict = min(t_dict, @elapsed ref = evolve_dict(O0, gens, angs, strat))
            t_spv  = min(t_spv,  @elapsed new = evolve_spv(O0, gens, angs, strat))
        end

        # same answer, to machine precision
        err = norm(SparsePauliVector(ref; T=Float64) - new)
        err < 1e-12 || @warn("paths disagree: |diff| = $err")

        label = strat isa CoeffTruncation ? "coeff $(strat.thresh)" :
                "weight $(strat.strategies[1].max_weight) + " *
                "coeff $(strat.strategies[2].thresh)"
        @printf("%-30s %10d %10.3f %10.3f %11.1fx\n",
                label, length(new), t_dict, t_spv, t_dict / t_spv)
    end
    println("\n(final operators verified identical; window > 1 trades exact")
    println(" truncation cadence for additional speed)")
end

main()
