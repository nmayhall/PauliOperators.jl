"""
    trotterize(H::PauliSum{N,W,T}, dt::Real; n_trotter::Int=1, order::Int=1)

Decompose the time evolution operator exp(-i dt H) into a sequence of Pauli rotations
using the Lie-Trotter-Suzuki product formula.

Returns `(generators::Vector{PauliBasis{N}}, angles::Vector{Float64})` suitable for
passing to `evolve`.

For a Hamiltonian H = Σ_k c_k P_k, first-order Trotter gives:
    exp(-i dt H) ≈ Π_k exp(-i dt c_k P_k)

Each factor exp(-i dt c_k P_k) = exp(i θ/2 P_k) with θ = -2 dt Re(c_k) in the
Heisenberg picture convention used by `evolve`.

# Arguments
- `H::PauliSum{N,W,T}`: Hamiltonian (should be Hermitian, so coefficients are real)
- `dt::Real`: time step
- `n_trotter::Int=1`: number of Trotter steps (dt is divided by n_trotter)
- `order::Int=1`: Trotter order (1 = first-order, 2 = second-order symmetric)
"""
function trotterize(H::AnyPauliSum{N,W,T}, dt::Real; n_trotter::Int=1, order::Int=1) where {N,W,T}
    order in (1, 2) || throw(ArgumentError("Only order=1 and order=2 Trotter decompositions are supported"))

    generators = PauliBasis{N,W}[]
    angles = Float64[]
    step_dt = dt / n_trotter

    terms = collect(H)

    if order == 1
        for _ in 1:n_trotter
            for (P, c) in terms
                push!(generators, P)
                push!(angles, 2 * step_dt * real(c))
            end
        end
    elseif order == 2
        # Symmetric (Strang) splitting: half-step forward, then half-step backward
        for _ in 1:n_trotter
            for (P, c) in terms
                push!(generators, P)
                push!(angles, step_dt * real(c))  # half step
            end
            for (P, c) in reverse(terms)
                push!(generators, P)
                push!(angles, step_dt * real(c))  # half step
            end
        end
    end

    return generators, angles
end

"""
    qdrift(H::PauliSum{N,W,T}, dt::Real; n_samples::Int=1, rng::AbstractRNG=Random.default_rng())

Decompose time evolution using the QDrift protocol (Campbell, 2019).

Randomly samples Pauli terms from H with probability proportional to |c_k|,
producing a sequence of rotations that approximates exp(-i dt H) in expectation.

Each sampled term P_k is rotated by angle θ = -2 dt λ sign(c_k), where λ = Σ|c_k|
is the 1-norm of the coefficients.

Returns `(generators::Vector{PauliBasis{N}}, angles::Vector{Float64})`.

# Arguments
- `H::PauliSum{N,W,T}`: Hamiltonian
- `dt::Real`: time step
- `n_samples::Int=1`: number of random samples (more = better approximation)
- `rng::AbstractRNG`: random number generator
"""
function qdrift(H::AnyPauliSum{N,W,T}, dt::Real; n_samples::Int=1,
                rng::AbstractRNG=Random.default_rng()) where {N,W,T}
    terms = collect(H)
    coeffs = [real(c) for (_, c) in terms]
    abs_coeffs = abs.(coeffs)
    λ = sum(abs_coeffs)
    # λ = norm(H,1)
    probs = abs_coeffs ./ λ

    # Build cumulative distribution for sampling
    cum_probs = cumsum(probs)

    generators = PauliBasis{N,W}[]
    angles = Float64[]

    for _ in 1:n_samples
        r = rand(rng)
        idx = searchsortedfirst(cum_probs, r)
        idx = min(idx, length(terms))  # safety clamp

        P = terms[idx][1]
        c = coeffs[idx]
        push!(generators, P)
        push!(angles, 2 * dt * λ * sign(c) / n_samples)
    end

    return generators, angles
end
