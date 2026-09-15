# ============================================================
# VQE simulation with analytic rotation-circuit gradients.
#
# Minimizes C(θ) = ⟨ψ| U_M'⋯U_1' H U_1⋯U_M |ψ⟩ for a transverse-field Ising
# chain with a hardware-efficient ansatz of Ry rotations and ZZ entanglers,
# using `expectation_value_gradient`: the full analytic gradient at ~3x the
# cost of a single energy evaluation (one forward pass of H, one reverse pass
# each of H and the projected state).
#
# Run with:  julia --project=examples examples/vqe_demo.jl
# ============================================================

using PauliOperators
using LinearAlgebra
using Random
using Printf
using Optim

# H = -J Σ Z_i Z_{i+1} - h Σ X_i
function tfim_chain(N; J=1.0, h=1.2)
    H = PauliSum(N)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, Z=[i, i+1]))] = -J
    end
    for i in 1:N
        H[PauliBasis(Pauli(N, X=[i]))] = -h
    end
    return H
end

# L layers of {Ry on every site, then Y⊗Z entanglers}, plus a final Ry layer.
# All generators are imaginary antisymmetric, so the circuit is real
# orthogonal — well matched to the real TFIM ground state.
function hardware_efficient_ansatz(N, L)
    gens = PauliBasis{N}[]
    for _ in 1:L
        for i in 1:N
            push!(gens, PauliBasis(Pauli(N, Y=[i])))
        end
        for i in 1:N-1
            push!(gens, PauliBasis(Pauli(N, Y=[i], Z=[i+1])))
        end
    end
    for i in 1:N
        push!(gens, PauliBasis(Pauli(N, Y=[i])))
    end
    return gens
end

function main()
    Random.seed!(7)
    N = 6
    L = 4
    H = tfim_chain(N)
    gens = hardware_efficient_ansatz(N, L)
    ψ = Ket(N, 0)
    θ0 = 0.1 * randn(length(gens))          # small generic start near |0…0⟩

    E_exact = eigmin(Hermitian(Matrix(H)))
    println("TFIM chain: N = $N sites, ansatz: $L layers, $(length(gens)) parameters")
    @printf("exact ground energy:  %.10f\n\n", E_exact)

    n_evals = Ref(0)
    function fg!(F, G, θ)
        c, g = expectation_value_gradient(H, gens, collect(θ), ψ)
        n_evals[] += 1
        G === nothing || copyto!(G, g)
        return F === nothing ? nothing : c
    end

    res = Optim.optimize(Optim.NLSolversBase.only_fg!(fg!), θ0, LBFGS(),
                         Optim.Options(g_tol=1e-6, iterations=2000))
    E_vqe = Optim.minimum(res)

    @printf("VQE energy:           %.10f\n", E_vqe)
    @printf("error:                %.3e\n", E_vqe - E_exact)
    @printf("LBFGS iterations:     %d  (%d cost+gradient evaluations)\n",
            Optim.iterations(res), n_evals[])

    # Timing: full gradient vs a single cost evaluation. The :dynamic engine
    # prunes the reverse sweep to mirror the forward pass (~3x expected);
    # :static never deletes Paulis, so its reverse passes run at full size.
    θf = Optim.minimizer(res)
    expectation_value(H, gens, θf, ψ)                       # warm-up
    expectation_value_gradient(H, gens, θf, ψ; method=:dynamic)
    expectation_value_gradient(H, gens, θf, ψ; method=:static)
    t_cost = minimum(@elapsed expectation_value(H, gens, θf, ψ) for _ in 1:5)
    t_dyn = minimum(@elapsed expectation_value_gradient(H, gens, θf, ψ; method=:dynamic) for _ in 1:5)
    t_stat = minimum(@elapsed expectation_value_gradient(H, gens, θf, ψ; method=:static) for _ in 1:5)
    @printf("\ncost eval: %.3f ms   gradient :dynamic: %.3f ms (%.1fx)   :static: %.3f ms (%.1fx)\n",
            1e3 * t_cost, 1e3 * t_dyn, t_dyn / t_cost, 1e3 * t_stat, t_stat / t_cost)

    # Truncated evolution: the gradient's forward pass only truncates newly created
    # sin-branch terms (Paulis are never deleted once introduced), and the
    # gradient matches finite differences of that truncated cost to first
    # order in the threshold.
    c_t, _ = expectation_value_gradient(H, gens, θf, ψ;
                                        truncation=CoeffTruncation(1e-8))
    @printf("cost with CoeffTruncation(1e-8): %.10f\n", c_t)
end

main()
