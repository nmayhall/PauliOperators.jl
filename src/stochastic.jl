"""
    stochastic_clip!(ps::PauliSum{N,T}, ε::Real; rng=Random.default_rng())

Unbiased stochastic compression (Russian Roulette) of a PauliSum.

For each term (basis, c):
- If |c| >= ε: keep unchanged
- If |c| < ε: with probability |c|/ε, promote to ε·sign(c); otherwise delete

This is unbiased: E[c̃] = c for every term.
"""
function stochastic_clip!(ps::PauliSum{N,T}, ε::Real;
                          rng::AbstractRNG=Random.default_rng()) where {N,T}
    to_delete = PauliBasis{N}[]

    for (basis, c) in ps
        ac = abs(c)
        ac < ε || continue

        if rand(rng) < ac / ε
            ps[basis] = ε * (c / ac)   # promote (preserves phase for complex T)
        else
            push!(to_delete, basis)     # mark for deletion
        end
    end

    for basis in to_delete
        delete!(ps, basis)
    end
    return ps
end
