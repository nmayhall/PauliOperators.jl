# PauliOperators

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nmayhall.github.io/PauliOperators.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nmayhall.github.io/PauliOperators.jl/dev/)
[![Build Status](https://github.com/nmayhall/PauliOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nmayhall/PauliOperators.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nmayhall/PauliOperators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nmayhall/PauliOperators.jl)
[![Build Status (v3-dev)](https://github.com/nmayhall/PauliOperators.jl/actions/workflows/CI.yml/badge.svg?branch=v3-dev)](https://github.com/nmayhall/PauliOperators.jl/actions/workflows/CI.yml?query=branch%3Av3-dev)
[![Coverage (v3-dev)](https://codecov.io/gh/nmayhall/PauliOperators.jl/branch/v3-dev/graph/badge.svg)](https://codecov.io/gh/nmayhall/PauliOperators.jl/tree/v3-dev)

A Julia package for efficient manipulation of Pauli operators and quantum states using bitstring representations. The package uses symplectic representation to encode tensor products of Pauli operators as pairs of binary strings, enabling fast operations on systems with up to 128 qubits.

## Installation

```julia
using Pkg
Pkg.Registry.add(url="git@github.com:mayhallgroup/MayhallJuliaRegistry.git")  # one-time setup
Pkg.add("PauliOperators")
```

## Pauli Representation

The format for an arbitrary Pauli operator is:

$$
\begin{align}
P_n =& i^\theta Z^{z_1} X^{x_1} ⊗ Z^{z_2} X^{x_2} ⊗ ⋯ ⊗ Z^{z_N} X^{x_N}  \\
=& i^\theta \bigotimes_j Z^{z_j} X^{x_j}
\end{align}
$$

where the $z$ and $x$ strings are encoded as bitwise representations of `Int128` integers, supporting up to 128 qubits. The $Y$ operator (or rather $iY$) is represented by setting both $x_i$ and $z_i$ to 1.

## Types

### Pauli Operators

- **`PauliBasis{N}`**: Hermitian, positive Pauli basis elements. These form the basis for linear combinations and are used as keys in `PauliSum` dictionaries.
```julia
  struct PauliBasis{N}
      z::Int128  # Z operator bitstring
      x::Int128  # X operator bitstring
  end
```

- **`Pauli{N}`**: General Pauli operator with an arbitrary complex coefficient, allowing representation of scaled and phased Pauli strings.
```julia
  struct Pauli{N}
      s::ComplexF64           # Coefficient
      z::Int128               # Z operator bitstring
      x::Int128               # X operator bitstring
  end
```

- **`PauliSum{N,T}`**: Linear combination of Pauli operators, implemented as a dictionary mapping `PauliBasis{N}` to coefficients of type `T`.
```julia
  PauliSum{N,T} = Dict{PauliBasis{N}, T}
```

### Quantum States

- **`Ket{N}` and `Bra{N}`**: Computational basis states represented as bitstrings.
```julia
  struct Ket{N}
      v::Int128  # Occupation number bitstring
  end

  struct Bra{N}
      v::Int128
  end
```

- **`KetSum{N,T}`**: Linear combination of computational basis states.
```julia
  KetSum{N,T} = Dict{Ket{N}, T}
```

### Density Matrix Elements

- **`DyadBasis{N}`**: Basis elements for density matrices and operators, representing outer products $|i⟩⟨j|$.
```julia
  struct DyadBasis{N}
      ket::Ket{N}
      bra::Bra{N}
  end
```

- **`Dyad{N}`**: Scaled dyads with complex coefficients.
```julia
  struct Dyad{N}
      s::ComplexF64
      ket::Ket{N}
      bra::Bra{N}
  end
```

- **`DyadSum{N,T}`**: Linear combination of dyads, useful for representing density matrices and general operators.
```julia
  DyadSum{N,T} = Dict{DyadBasis{N}, T}
```

## Key Features

- **Efficient Operations**: Pauli multiplication, addition, and tensor products using bitwise operations
- **Zero Allocations**: Core operations are allocation-free when the number of qubits `N` is a compile-time constant
- **Flexible Arithmetic**: Overloaded operators (`*`, `+`, `⊗`, `⊕`) for intuitive manipulation
- **Quantum Computations**:
  - `expectation_value(operator, state)` - compute expectation values
  - `matrix_element(bra, operator, ket)` - compute matrix elements
  - Conversion to dense matrix representations for verification
- **Quantum Channels**: Heisenberg-picture single-qubit Pauli channels applied i.i.d. across qubits — `depolarizing_channel!`, `dephasing_channel!`/`phase_flip_channel!`, `bit_flip_channel!`, `bit_phase_flip_channel!`, and a general `pauli_channel!(O, pX, pY, pZ)`. Each is a single popcount per `PauliSum` term — no expansion of the sum.

## Quick Example
```julia
using PauliOperators

# Create Pauli operators (3 qubits)
X = Pauli("XII")
Y = Pauli("IYI")
Z = Pauli("IIZ")

# Build a Hamiltonian
H = PauliSum(3, ComplexF64)
H[PauliBasis("XII")] = -1.0
H[PauliBasis("IYI")] = -0.5
H[PauliBasis("IIZ")] = -0.3

# Create a quantum state
ψ = Ket([1, 0, 1])  # |101⟩

# Compute expectation value
E = expectation_value(H, ψ)

# Tensor products
ZZ = Z ⊗ Z  # Create a 6-qubit operator

# Direct sums
H_total = H ⊕ H  # Combine two 3-qubit Hamiltonians
```

## Norms and Algebra

Standard `LinearAlgebra.norm` is extended for PauliSum and KetSum with support for arbitrary p-norms:
```julia
using LinearAlgebra

norm(H)        # L2 norm (default)
norm(H, 1)     # L1 norm (sum of |c_k|)
norm(H, Inf)   # L∞ norm (max |c_k|)
isapprox(H1, H2; atol=1e-10)  # Approximate equality
```

Optimized commutator and anticommutator that exploit the single-pass `P_i P_j = phase * P_k` identity, skipping commuting/anti-commuting pairs:
```julia
C = commutator(A, B)      # [A, B]
A_plus = anticommutator(A, B)  # {A, B}
```

Statistical quantities measured against a reference state:
```julia
v = variance(O, ψ)           # ⟨O²⟩ - ⟨O⟩²
c = covariance(A, B, ψ)      # ⟨A†B⟩ - ⟨A†⟩⟨B⟩
```

## Truncation Strategies

An abstract `TruncationStrategy` type system with pluggable error tracking via `CorrectionAccumulator`:

```julia
# Available strategies
strat = CoeffTruncation(1e-6)             # Drop terms with |c| < threshold
strat = WeightTruncation(3)               # Keep terms with Pauli weight ≤ 3
strat = MajoranaWeightTruncation(4)       # Keep terms with Majorana weight ≤ 4
strat = AdaptiveTruncation(1000, 1e-8)    # Keep at most 1000 terms, min threshold 1e-8
strat = CompositeTruncation(s1, s2, ...)  # Apply multiple strategies in sequence

# Stochastic strategies
strat = StochasticCoeffTruncation(ε)      # Randomly round small coefficients
strat = StochasticSamplingTruncation(k)   # Importance-sample to k terms

# Apply truncation (in-place)
truncate!(O, strat)

# Track energy/variance corrections across truncation steps
corr = EnergyCorrection(ψ)
truncate!(O, strat, corr)
println(corr.accumulated_energy)
```

## Evolution

### Single-generator evolution
Heisenberg picture: $O(\theta) = e^{i\theta/2\, G}\, O\, e^{-i\theta/2\, G}$

Schr&ouml;dinger picture: $|K(\theta)\rangle = e^{-i\theta/2\, G}|K\rangle$

```julia
# Heisenberg picture (PauliSum)
O_evolved = evolve(O, G, θ)
evolve!(O, G, θ)  # in-place

# Schrödinger picture (KetSum)
K_evolved = evolve(K, G, θ)
```

### Sequence evolution with truncation
```julia
O_out = evolve(O, generators, angles;
               truncation=CoeffTruncation(1e-8),
               correction=EnergyCorrection(ψ))
```

Note: for KetSum sequence evolution, `trotterize`/`qdrift` sequences must be reversed:
```julia
gens, angs = trotterize(H, dt)
K_out = evolve(K, reverse(gens), reverse(angs))
```

### Trotter and QDrift decomposition
Decomposition is separated from evolution. Functions return `(generators, angles)` tuples that are passed to `evolve`:
```julia
# First-order Trotter
gens, angs = trotterize(H, dt; n_trotter=10)
O_out = evolve(O, gens, angs)

# Second-order (symmetric) Trotter
gens, angs = trotterize(H, dt; order=2)

# QDrift stochastic protocol
gens, angs = qdrift(H, dt; n_samples=100)
```

### Quantum gates

Clifford gates act directly on the Pauli basis via a symplectic tableau (no Pauli rotations,
no `cos`/`sin` branching). Every gate has a typed `CliffordGate` representation and a
named-function alias. Both forms work on `PauliSum` (Heisenberg picture), `KetSum`
(Schr&ouml;dinger picture), and a bare `Ket` (promoted to a single-term `KetSum`).

```julia
# Named-function aliases (backwards-compatible)
O = hadamard(O, q)
O = cnot(O, ctrl, tgt)
O = X_gate(O, q);  O = Y_gate(O, q);  O = Z_gate(O, q)
O = S_gate(O, q)             # phase gate, diag(1, i)
O = T_gate(O, q)             # not a Clifford; still rotation-based

# Equivalent typed gate objects (apply with `apply`)
O = apply(Hadamard(q), O)
O = apply(CNOT(ctrl, tgt), O)
O = apply(PhaseGate(q), O)   # S      = diag(1,  i)
O = apply(PhaseDg(q), O)     # S†     = diag(1, -i)
O = apply(SqrtX(q), O)       # √X
O = apply(SqrtY(q), O)       # √Y
O = apply(CZ(ctrl, tgt), O)
O = apply(SWAP(a, b), O)
```

#### Arbitrary N-qubit Cliffords via `CliffordTableau`

`CliffordTableau{N}` stores a Clifford as its action on each generator (the Aaronson–Gottesman
symplectic tableau) and applies it to a `PauliSum` in O(N) per Pauli term.

```julia
# Build a tableau from a primitive gate
C = CliffordTableau{N}(Hadamard(2))

# Build one from a sequence of primitives (applied left-to-right)
C = CliffordTableau{N}([Hadamard(1), CNOT(1, 2), PhaseGate(3), CZ(2, 3)])

# Compose two tableaux (C1 * C2 means "C2 first, then C1")
C12 = C1 * C2

# `*` requires at least one operand to be a CliffordTableau (so N is known).
# Composing two primitives directly raises an error — lift one first:
C = CliffordTableau{N}(CNOT(1, 2)) * Hadamard(1)
# ...or list them in left-to-right (applied) order:
C = CliffordTableau{N}([Hadamard(1), CNOT(1, 2)])

# Invert
Cinv = adjoint(C)

# Apply to a PauliSum (Heisenberg) or to a single PauliBasis (returns sign, image)
O = apply(C, O)
sgn, p_image = apply(C, p::PauliBasis)
```

#### Random Cliffords and dense matrices

```julia
# Random N-qubit Clifford (approximately uniform — composes O(N²) random primitives)
C = rand(CliffordTableau{4})
C = rand(rng, CliffordTableau{4})                # with a fixed RNG for reproducibility
C = rand(CliffordTableau{4}; depth=1000)         # increase depth for better uniformity

# Dense 2^N × 2^N unitary realising the Clifford (up to a global phase).
# Practical for N up to ~10–12; cost is O(4^N).
U = Matrix(C)
```

The current `rand` is not provably uniform — it composes a random circuit of depth
`O(N²)` and relies on rapid mixing of the random walk on the Clifford group. For
applications that need strict uniformity (e.g. randomized benchmarking, t-design
protocols), a Bravyi-Maslov sampler is a planned follow-up.

#### Identifying a Clifford from its dense unitary

For small blocks you can pass a 2<sup>K</sup>&times;2<sup>K</sup> unitary matrix
and the target qubits; the tableau is identified by checking each generator's image
against the list of Hermitian Paulis. Practical for K up to ~4.

```julia
# Identify a 1-qubit Clifford from its 2×2 matrix
H_matrix = [1 1; 1 -1] / sqrt(2)
C = CliffordTableau{1}(H_matrix)   # == CliffordTableau{1}(Hadamard(1))

# Lift a small block onto target qubits of a larger system: this acts as CNOT
# between qubits 3 and 7 of a 10-qubit system, no matter the layout in between.
CNOT_4x4 = ComplexF64[1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]
C = CliffordTableau{10}(CNOT_4x4, [3, 7])

# Non-Clifford input is rejected
T_matrix = [1 0; 0 exp(im*π/4)]
CliffordTableau{1}(T_matrix)   # throws ArgumentError
```

In all cases, the same `apply(gate, ...)` interface is used; primitives accept `PauliSum`,
`KetSum`, or `Ket` operands. Note: `apply(::CliffordTableau, ::KetSum)` (or `::Ket`) is **not**
directly supported &mdash; apply the constituent primitive gates to the state instead
(or decompose the tableau into primitives).

## Analysis Utilities

### Weight distribution
```julia
counts = get_weight_counts(O)           # terms per Pauli weight
probs  = get_weight_probs(O)            # |c|² per Pauli weight
counts = get_majorana_weight_counts(O)  # terms per Majorana weight
probs  = get_majorana_weight_probs(O)   # |c|² per Majorana weight
```

### Operator inspection
```julia
top = find_top_k(O, 10)    # 10 largest terms by |c|, sorted
big = largest(O)            # single largest term as PauliSum
ld  = largest_diag(O)       # largest diagonal (Z-only) term

O_diag = diag(O)            # diagonal terms only (x == 0)
O_off  = offdiag(O)         # off-diagonal terms only (x != 0)
```

### Clipping and filtering
```julia
coeff_clip!(O; thresh=1e-8)          # drop terms with |c| < thresh
weight_clip!(O, max_weight)          # drop terms with weight > max_weight
majorana_weight_clip!(O, max_weight) # drop by Majorana weight
stochastic_clip!(O, ε)               # stochastic rounding
```

### Subspace matrices
Construct matrix representations in a subspace spanned by selected kets:
```julia
S = [Ket(N, 0), Ket(N, 1), Ket(N, 3)]  # subspace basis
M = Matrix(O, S)     # M[i,j] = ⟨S[i]|O|S[j]⟩
v = Vector(K, S)     # v[i] = K[S[i]]
```

## Quantum Channels

Single-qubit Pauli channels are applied i.i.d. across qubits in the Heisenberg picture, `O ↦ Σₖ Kₖ† O Kₖ`. Pauli channels are diagonal in the Pauli basis, so each `PauliBasis` term is rescaled by a closed-form factor — no new terms are produced.

| Channel | Convention | Per-qubit factor on suppressed Paulis |
| --- | --- | --- |
| `depolarizing_channel!(O, p)` | `pI=1−p, pX=pY=pZ=p/3` | `1 − 4p/3` on X, Y, Z |
| `dephasing_channel!(O, p)` (alias `phase_flip_channel!`) | `pI=1−p, pZ=p` | `1 − 2p` on X, Y |
| `bit_flip_channel!(O, p)` | `pI=1−p, pX=p` | `1 − 2p` on Y, Z |
| `bit_phase_flip_channel!(O, p)` | `pI=1−p, pY=p` | `1 − 2p` on X, Z |
| `pauli_channel!(O, pX, pY, pZ)` | general | `λ_X = 1−2(pY+pZ)`, etc. |

Each channel has an in-place form (`!`) and an allocating form (returns a new `PauliSum`). All accept a `qubits` keyword (default: all `1:N`) to restrict the action to a subset.

```julia
O = PauliSum(3)
O[PauliBasis("XYZ")] = 1.0 + 0im
O[PauliBasis("III")] = 1.0 + 0im

depolarizing_channel!(O, 0.1)              # acts on all qubits
dephasing_channel!(O, 0.05; qubits=[1, 3]) # acts only on qubits 1 and 3
```

The convention follows Nielsen-Chuang (Kraus probabilities). Note the consequences at the endpoints:

- **Depolarizing**: `p = 3/4` is the fully-depolarizing point (X, Y, Z → 0); `p = 1` gives `λ = −1/3`.
- **Dephasing / bit-flip / bit-phase-flip**: `p = 1/2` is the fully-decohering point (suppressed Paulis → 0); `p = 1` is unitary conjugation by Z / X / Y, which preserves magnitudes and only flips signs.

### Weight-decay equivalence

The i.i.d. depolarizing channel is exactly an exponential-in-weight CPTP damper. Apply

```julia
p = depolarizing_p_for_weight_decay(γ, Δt)   # = (3/4)·(1 − exp(−γ·Δt))
depolarizing_channel!(O, p)
```

and each Pauli `P` is scaled by `exp(−γ·Δt · w(P))`, where `w(P)` is the number of non-identity qubits. (A *thresholded* weight damper of the form "only damp when `w > lmax`" is **not CPTP in general** and is intentionally not provided.)

## Performance

The package is designed for high performance with zero-allocation operations:
- Pauli multiplication and addition
- Expectation value calculations
- State manipulations

When `N` (number of qubits) is known at compile time, all core operations are performed without heap allocations, making this package suitable for performance-critical quantum simulations.
