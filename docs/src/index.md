```@meta
CurrentModule = PauliOperators
```

# PauliOperators

Documentation for [PauliOperators](https://github.com/nmayhall/PauliOperators.jl), a
Julia package for efficient manipulation of Pauli operators and quantum states using
symplectic bitstring representations, supporting up to 1024 qubits. The bitstring
storage word is sized to the register (`UInt64` up to 64 qubits, through `UInt1024`
via BitIntegers.jl), so small systems pay nothing for the large-system capability.

## Where to look

The [README](https://github.com/nmayhall/PauliOperators.jl) is the **user manual**: it
covers installation, every type, and every user-facing feature with examples. Start
there.

This site covers what the README doesn't — the lower-level material:

- **[Pauli Representation](representation.md)** — the symplectic ``(z|x)`` encoding,
  the exact phase conventions (`symplectic_phase`, `coeff`, why Y prints as `y`), bit
  ordering, and the bit-level product/commutator identities the kernels are built on.
- **[Data Structures & Performance](data_structures.md)** — how the `Dict`-backed
  `PauliSum` and the flat, zero-allocation `SparsePauliVector` are laid out, how the
  windowed evolution engine works, and how to choose between them.
- **[Truncation](truncation.md)** — the weight measures, the exact semantics of every
  truncation strategy, error tracking with correction accumulators, and how to extend
  the system.
- **[Migrating to v4](migration_v4.md)** — the word-type parameterization
  (`PauliBasis{N,W}`, `PauliSum{N,W,T}`, >128 qubits): what breaks, what doesn't, and
  the mechanical signature rewrites.
- **Migration Guides** — updating downstream packages to PauliOperators v3.
- **Reference** — docstrings for all [types](types.md) and [functions](functions.md).
