# Migration Guide: DBF.jl

This document describes the changes needed to update DBF.jl to use PauliOperators v3 and remove redundant code.

## Overview

DBF.jl currently redefines many functions that are now provided by PauliOperators. The goal is to remove these redefinitions and use PauliOperators directly, reducing maintenance burden and ensuring consistency.

## Version Constraint

Update `Project.toml`:
```toml
[compat]
PauliOperators = "3"
```

## Functions to Remove

### `src/helpers.jl`

The following functions are now provided by PauliOperators and should be deleted from DBF:

| DBF function | PauliOperators replacement | Notes |
|---|---|---|
| `weight(p)` | `weight(p)` | Identical |
| `coeff_clip!(ps; thresh)` | `coeff_clip!(ps; thresh)` | Identical, already exported |
| `weight_clip!(ps, w)` | `weight_clip!(ps, w)` | Identical, already exported |
| `inner_product(A, B)` | `inner_product(A, B)` | Identical |
| `norm(ps)` | `norm(ps)` | Use `LinearAlgebra.norm(ps, p)` for p-norms |
| `l1_norm(ps)` | `norm(ps, 1)` | Unified p-norm API |
| `l4_norm(ps)` | `norm(ps, 4)` | Unified p-norm API |
| `offdiag(ps)` | `offdiag(ps)` | Identical |
| `diag(ps)` | `diag(ps)` | Now `LinearAlgebra.diag` |
| `get_weight_counts(O)` | `get_weight_counts(O)` | Identical |
| `get_weight_probs(O)` | `get_weight_probs(O)` | Identical |
| `get_mweight_counts(O)` | `get_majorana_weight_counts(O)` | Renamed |
| `get_mweight_probs(O)` | `get_majorana_weight_probs(O)` | Renamed |
| `find_top_k(dict, k)` | `find_top_k(O, k)` | Identical algorithm |
| `largest(ps)` | `largest(ps)` | Identical |
| `largest_diag(ps)` | `largest_diag(ps)` | Identical |
| `majorana_weight(p)` | `majorana_weight(p)` | Identical |

### `src/helpers.jl` — Subspace matrices

Remove the `Matrix(O, S::Vector{Ket})` and `Vector(K, S::Vector{Ket})` methods. PauliOperators now provides these.

### `src/adapt.jl`

| DBF function | PauliOperators replacement | Notes |
|---|---|---|
| `variance(O, ψ)` | `variance(O, ψ)` | Identical |
| `covariance(A, B, ψ)` | `covariance(A, B, ψ)` | Identical |

### `src/evolve.jl`

DBF defines its own `evolve` and `evolve!` methods. These are already exported by PauliOperators (and were in v2). Verify that DBF's versions have identical behavior, then remove them:

```julia
# Remove from DBF — use PauliOperators.evolve instead
# evolve(O::PauliSum, G::PauliBasis, θ)
# evolve!(O::PauliSum, G::PauliBasis, θ)
```

### `src/gates.jl` or gate-related code

Remove local gate definitions. PauliOperators now provides:
- `hadamard(O, qubit)`, `cnot(O, control, target)`
- `X_gate`, `Y_gate`, `Z_gate`, `S_gate`, `T_gate`
- `hadamard_to_paulis`, `cnot_to_paulis`, `X_gate_to_paulis`, `Z_gate_to_paulis`

### Truncation code

DBF should remove any local `TruncationStrategy` definitions and use PauliOperators' truncation system directly:
- `truncate!(O, strategy)` replaces manual clip calls
- `CoeffTruncation`, `WeightTruncation`, `AdaptiveTruncation`, etc.
- `EnergyCorrection`, `EnergyVarianceCorrection` for error tracking

## Import Changes

If DBF currently does `using PauliOperators`, the new exports will be available automatically. If there are name conflicts with local definitions that remain, use explicit imports:

```julia
using PauliOperators
# If you need to keep a local `evolve` variant:
import PauliOperators: evolve  # then extend with new signatures
```

## Sequence Evolution

Replace any manual evolution loops:
```julia
# Old pattern in DBF:
for (gi, θi) in zip(generators, angles)
    evolve!(O, gi, θi)
    coeff_clip!(O; thresh=thresh)
    weight_clip!(O, max_weight)
end

# New pattern using PauliOperators:
O = evolve(O, generators, angles;
           truncation=CompositeTruncation(
               CoeffTruncation(thresh),
               WeightTruncation(max_weight)))
```

## Trotter Decomposition

Replace monolithic Trotter evolution with the decompose-then-evolve pattern:
```julia
# Old: custom trotter_evolve function
# New:
gens, angs = trotterize(H, dt; n_trotter=10, order=2)
O = evolve(O, gens, angs; truncation=strat)
```

## Testing

After removing redundant code, run DBF's test suite to verify everything still works. Pay attention to:
1. Any test that constructs PauliOperators types directly
2. Tests that compare against dense matrices (verify numerical agreement)
3. Evolution tests (sign conventions should match)
