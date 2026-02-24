"""
    stochastic_propagate(O0, generators, angles, ψ, ε; kwargs...)

Run M independent stochastic Pauli propagation trials.

Each trial:
1. Propagates O0 through the Pauli rotation gates (evolve!)
2. Merges terms sharing the same Pauli string (automatic via Dict)
3. Applies stochastic compression (stochastic_clip!) to control growth
4. Evaluates the expectation value ⟨ψ|O(t)|ψ⟩

# Arguments
- `O0::PauliSum{N,T}`: initial operator to propagate
- `generators::Vector{PauliBasis{N}}`: Pauli rotation generators
- `angles::Vector{<:Real}`: rotation angles (θ in exp(iθ/2 G) O exp(-iθ/2 G))
- `ψ::Ket{N}`: stabilizer state for expectation value evaluation
- `ε::Real`: stochastic compression threshold

# Keyword Arguments
- `n_samples::Int=100`: number of independent runs
- `max_weight::Int=N`: maximum Pauli weight (N disables weight clipping)
- `compress_every::Int=1`: apply compression every k gates (1=per gate)
- `seed::Int=0`: base RNG seed (run m uses seed+m)
- `verbose::Int=1`: print level

# Returns
Named tuple `(mean, stderr, samples)` where:
- `mean`: average expectation value across runs
- `stderr`: standard error of the mean
- `samples`: Vector{Float64} of individual run results
"""
function stochastic_propagate(
    O0::PauliSum{N,T},
    generators::Vector{PauliBasis{N}},
    angles::Vector{<:Real},
    ψ::Ket{N},
    ε::Real;
    n_samples::Int = 100,
    max_weight::Int = N,
    compress_every::Int = 1,
    seed::Int = 0,
    verbose::Int = 1
) where {N,T}

    ng = length(generators)
    ng == length(angles) || throw(DimensionMismatch("generators and angles must have same length"))

    samples = Vector{Float64}(undef, n_samples)

    Threads.@threads for m in 1:n_samples
        rng = Random.Xoshiro(seed + m)
        Ot = deepcopy(O0)

        for (idx, (gi, θi)) in enumerate(zip(generators, angles))
            evolve!(Ot, gi, θi)

            if idx % compress_every == 0
                stochastic_clip!(Ot, ε; rng=rng)
                weight_clip!(Ot, max_weight)
            end
        end

        # Final compression if last gate wasn't a compression point
        if ng % compress_every != 0
            stochastic_clip!(Ot, ε; rng=rng)
            weight_clip!(Ot, max_weight)
        end

        samples[m] = real(expectation_value(Ot, ψ))
    end

    μ = sum(samples) / n_samples
    σ = sqrt(sum((s - μ)^2 for s in samples) / (n_samples * (n_samples - 1)))

    if verbose >= 1
        @printf(" SPP: mean = %12.8f  stderr = %12.8f  n_samples = %i\n", μ, σ, n_samples)
    end

    return (mean=μ, stderr=σ, samples=samples)
end
