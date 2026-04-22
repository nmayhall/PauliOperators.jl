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
function get_1d_neel_state_sequence(N)
    g = Vector{PauliBasis{N}}([])
    a = Vector{Float64}([])
    for i in 1:N
        if i%2 == 0
            push!(g, PauliBasis(Pauli(N, X=[i])))
            push!(a, π)
        end
    end
    return g, a 
end

function run()
    N = 6
    H = heisenberg_1d(N)
    Ot = PauliSum(Pauli(N, Z=[1]))
    Oi = zeros(ComplexF64, N)
    ψ = Ket(N,0)
    g, a = get_1d_neel_state_sequence(N)
    H = evolve(H,g,a)

    dt = .1

    Random.seed!(2)

    # QDrift stochastic protocol
    gens, angs = qdrift(H, dt; n_samples=200)
 
    correction = EnergyCorrection(ψ)
    truncation = CoeffTruncation(1e-4)
    nt = 100
    for ti in 1:nt
        # Ot = evolve(Ot, gens, angs;
        #     truncation=CoeffTruncation(1e-6),
        #     correction=EnergyCorrection(ψ))
        for (gi, θi) in zip(gens, angs)
            evolve!(Ot, gi, θi)
            truncate!(Ot, truncation, correction)
        end
        ev = expectation_value(Ot,ψ)
        evc = ev - correction.accumulated_energy
        # @printf(" %12.8f %12.8f %5i\n", real(ev), real(evc), length(Ot))
        @printf(" %12.8f %12.8f\n", real(ev), real(evc))
    end
end

run()