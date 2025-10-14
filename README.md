# PauliOperators

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nmayhall.github.io/PauliOperators.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nmayhall.github.io/PauliOperators.jl/dev/)
[![Build Status](https://github.com/nmayhall/PauliOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nmayhall/PauliOperators.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nmayhall/PauliOperators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nmayhall/PauliOperators.jl)

A Julia package for efficient manipulation of Pauli operators and quantum states using bitstring representations. The package uses symplectic representation to encode tensor products of Pauli operators as pairs of binary strings, enabling fast operations on systems with up to 128 qubits.

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

## Performance

The package is designed for high performance with zero-allocation operations:
- Pauli multiplication and addition
- Expectation value calculations
- State manipulations

When `N` (number of qubits) is known at compile time, all core operations are performed without heap allocations, making this package suitable for performance-critical quantum simulations.