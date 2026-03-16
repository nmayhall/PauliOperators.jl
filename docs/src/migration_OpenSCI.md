# Migration Guide: OpenSCI.jl

This document describes the changes needed to update OpenSCI.jl from PauliOperators v1 (custom `spinboson` branch) to PauliOperators v3.

## Overview

OpenSCI.jl currently depends on:
1. PauliOperators v1 via the custom `spinboson` branch
2. DBF.jl for evolution utilities (`DBF.evolve`, `DBF.coeff_clip!`, `DBF.weight_clip!`)

The goal is to update to PauliOperators v3, remove the DBF dependency for basic utilities, and use PauliOperators' built-in evolution and truncation.

## Version Constraint

Update `Project.toml`:
```toml
[deps]
PauliOperators = "be646426-..."

[compat]
PauliOperators = "3"
```

Remove the custom GitHub URL / `spinboson` branch reference if present.

## v1 to v3 Breaking Changes

PauliOperators v2 introduced structural changes from v1. Key differences:

1. **Type parameter**: `PauliSum{N}` became `PauliSum{N,T}` with explicit coefficient type
2. **Constructor**: `PauliSum(N)` now defaults to `ComplexF64` coefficients
3. **PauliBasis**: The internal representation may differ; string constructors like `PauliBasis("ZI")` should still work

Audit all PauliOperators type usage in OpenSCI and update as needed.

## Removing DBF Dependency for Utilities

### `src/evolve.jl`

Replace DBF utility calls with PauliOperators:

```julia
# Old (OpenSCI/src/evolve.jl):
Ot = DBF.evolve(Ot, gi, θi)
DBF.coeff_clip!(Ot, thresh=thresh)
DBF.weight_clip!(Ot, max_weight)

# New — use PauliOperators directly:
using PauliOperators: evolve!, coeff_clip!, weight_clip!

evolve!(Ot, gi, θi)
coeff_clip!(Ot; thresh=thresh)
weight_clip!(Ot, max_weight)
```

Or use the sequence evolution API:
```julia
# New — single call with truncation:
Ot = evolve(O, generators, angles;
            truncation=CompositeTruncation(
                CoeffTruncation(thresh),
                WeightTruncation(max_weight)))
```

### Other DBF utilities

Check if OpenSCI uses any other DBF functions that are now in PauliOperators:
- `DBF.weight(p)` → `weight(p)`
- `DBF.inner_product(A, B)` → `inner_product(A, B)`
- `DBF.norm(O)` → `norm(O)`

If DBF is still needed for DBF-specific functionality (e.g., the double-bracket flow algorithm itself), keep it as a dependency but remove reliance on it for basic operator utilities.

## New PauliOperators Features Available

After migrating, OpenSCI gains access to:
- `variance(O, ψ)`, `covariance(A, B, ψ)` for adaptive algorithms
- `commutator(A, B)`, `anticommutator(A, B)` (optimized single-pass)
- `trotterize(H, dt)`, `qdrift(H, dt)` for decomposition
- `TruncationStrategy` type system for flexible truncation
- Quantum gates: `hadamard`, `cnot`, `X_gate`, etc.
- Subspace matrices: `Matrix(O, basis_kets)`

## Testing

1. Run OpenSCI tests after updating the PauliOperators dependency
2. Verify that evolution results match previous behavior (sign conventions are consistent)
3. Test any code that was using the `spinboson` branch-specific features
