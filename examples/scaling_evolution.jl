# Scaling of Sparse Pauli Dynamics with system size.
#
# Heisenberg-picture evolution of a local observable under a 1D Heisenberg chain,
# swept over N = 100, 200, 400, 800, 1000 qubits with a fixed coefficient
# truncation of 1e-2 for 5 Trotter layers. Prints wall time and the number of
# retained Pauli terms (the memory footprint) at each size, so you can see how
# time and memory grow toward the 10x10x10 = 1000-qubit regime.
#
#     julia --project examples/scaling_evolution.jl

using PauliOperators
using Printf

# One first-order Trotter layer of a 1D Heisenberg chain: XX+YY+ZZ on each bond.
function heisenberg_1d_generators(N)
    T = PauliOperators.uinttype(N)
    gens = PauliBasis{N,T}[]
    for i in 1:N-1
        push!(gens, PauliBasis(Pauli(N; X=[i, i + 1])))
        push!(gens, PauliBasis(Pauli(N; Y=[i, i + 1])))
        push!(gens, PauliBasis(Pauli(N; Z=[i, i + 1])))
    end
    return gens
end

const THRESH   = 1e-2     # SPD coefficient truncation
const N_LAYERS = 5        # Trotter layers ("iterations")
const DT       = 0.1

function run_one(N, thresh=THRESH)
    gens = heisenberg_1d_generators(N)
    angles = fill(DT, length(gens))
    # observable O(0) = Z on the middle site
    O = PauliSum(N)
    O[PauliBasis(Pauli(N; Z=[N ÷ 2]))] = 1.0 + 0.0im

    max_terms = length(O)
    # Truncate once per layer, after applying all the angles (rotations).
    t = @elapsed for layer in 1:N_LAYERS
        for (G, θ) in zip(gens, angles)
            evolve!(O, G, θ)
        end
        coeff_clip!(O, thresh)
        max_terms = max(max_terms, length(O))
    end
    # coefficient memory ≈ terms * (2*ceil(N/64)*8 basis bytes + 16 coeff bytes)
    bytes_per_term = 2 * cld(N, 64) * 8 + 16
    mem_mb = length(O) * bytes_per_term / 1e6
    return (t=t, terms=length(O), max_terms=max_terms, mem_mb=mem_mb,
            norm=sqrt(sum(abs2, values(O))))
end

@printf("\n1D Heisenberg SPD scaling  (thresh=%.0e, %d Trotter layers, dt=%.2f)\n",
        THRESH, N_LAYERS, DT)
@printf("%-8s %10s %12s %12s %12s %10s\n",
        "N", "time(s)", "terms", "peak_terms", "mem(MB)", "|O|2")
@printf("%s\n", repeat("-", 70))
run_one(64)   # warm up compilation (not reported)
for N in (100, 200, 400, 800, 1000)
    r = run_one(N, THRESH)
    @printf("%-8d %10.3f %12d %12d %12.3f %10.6f\n",
            N, r.t, r.terms, r.max_terms, r.mem_mb, r.norm)
    flush(stdout)
end

# Truncation threshold is the memory/accuracy knob. Tightening it grows the
# retained-term count (and time); this is what decides single-node vs multinode.
@printf("\nTruncation sweep at N=1000 (%d Trotter layers)\n", N_LAYERS)
@printf("%-10s %10s %12s %12s %10s\n", "thresh", "time(s)", "terms", "mem(MB)", "|O|2")
@printf("%s\n", repeat("-", 60))
for th in (1e-2, 1e-3, 1e-4, 1e-5)
    r = run_one(1000, th)
    @printf("%-10.0e %10.3f %12d %12.3f %10.6f\n", th, r.t, r.terms, r.mem_mb, r.norm)
    flush(stdout)
end
