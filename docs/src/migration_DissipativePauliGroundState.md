# Migration Guide: DissipativePauliGroundState.jl

This document describes the changes needed to update DissipativePauliGroundState.jl to use PauliOperators v3 and remove redundant code.

## Overview

DissipativePauliGroundState.jl currently:
1. Imports `commute` from PauliOperators (now exported)
2. Extends `PauliOperators.clip!` with its own signature
3. Redefines `evolve!` for PauliSum (direct conflict)
4. Reimplements weight-based clipping and adaptive truncation

The goal is to remove these redefinitions and use PauliOperators' built-in truncation and evolution.

## Version Constraint

Update `Project.toml`:
```toml
[compat]
PauliOperators = "3"
```

## Import Changes

```julia
# Old:
using PauliOperators
using PauliOperators: commute
import PauliOperators: clip!

# New (commute is now exported, no special import needed):
using PauliOperators
```

## Functions to Remove

### `evolve!` (CRITICAL — name conflict)

DissipativePauliGroundState defines its own `evolve!(O::PauliSum{N,T}, G::PauliBasis{N}, θ)`. This directly conflicts with the PauliOperators export.

**Action**: Remove the local definition. PauliOperators' `evolve!` is functionally identical.

If there are behavioral differences (e.g., different truncation applied inside the loop), refactor to use the sequence evolution API instead:

```julia
# Old pattern:
function my_evolve!(O, generators, angles; thresh)
    for (g, θ) in zip(generators, angles)
        evolve!(O, g, θ)
        clip!(O; thresh=thresh)
    end
end

# New pattern:
O = evolve(O, generators, angles;
           truncation=CoeffTruncation(thresh))
```

### `clip!` extension

DissipativePauliGroundState extends `PauliOperators.clip!`:
```julia
function PauliOperators.clip!(ps::PauliSum{N,T}; thresh=1e-16) where {N,T}
    filter!(p -> abs(p.second) > thresh, ps)
end
```

PauliOperators already provides `clip!` for PauliSum. Remove this extension if the behavior is identical.

### Adaptive truncation

If DissipativePauliGroundState has adaptive clipping logic, replace it with:
```julia
strat = AdaptiveTruncation(max_terms, min_thresh)
truncate!(O, strat)
```

### Weight-based clipping

Replace any manual weight clipping with:
```julia
truncate!(O, WeightTruncation(max_weight))
# or
weight_clip!(O, max_weight)
```

## Using TruncationStrategy + CorrectionAccumulator

DissipativePauliGroundState likely tracks energy corrections during truncation. This is now built into PauliOperators:

```julia
# Track energy shift from truncation
corr = EnergyCorrection(ψ)
truncate!(O, strat, corr)
println("Accumulated energy correction: ", corr.accumulated_energy)

# Track both energy and variance
corr = EnergyVarianceCorrection(ψ)
truncate!(O, strat, corr)
```

## Evolution with Truncation

Replace manual evolution-truncation loops with the sequence API:

```julia
# Old:
for step in 1:n_steps
    for (g, θ) in zip(generators, angles)
        evolve!(O, g, θ)
        clip!(O; thresh=thresh)
    end
end

# New:
for step in 1:n_steps
    O = evolve(O, generators, angles;
               truncation=CoeffTruncation(thresh),
               correction=corr)
end
```

## New PauliOperators Features Available

After migrating, the package gains access to:
- `commutator(A, B)` — optimized, no need for `A*B - B*A`
- `variance(O, ψ)` — built-in variance computation
- `norm(O, p)` — standard p-norm API
- `trotterize(H, dt)` — Trotter decomposition separated from evolution
- `find_top_k(O, k)` — efficient top-k term selection
- `get_weight_counts(O)` — weight distribution analysis

## Testing

1. After removing redundant code, verify that the Lindbladian evolution produces the same results
2. Check that truncation with `CorrectionAccumulator` matches previous energy correction tracking
3. Run convergence tests to ensure numerical agreement
