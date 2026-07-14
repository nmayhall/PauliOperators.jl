```@meta
CurrentModule = PauliOperators
```

# Pauli Representation and Phase Conventions

This page documents the symplectic (bitstring) encoding underlying every type in the
package, and the exact phase conventions used. The [README](https://github.com/nmayhall/PauliOperators.jl)
covers *how to use* the types; this page explains *how they work* — read it if you are
implementing new kernels, debugging phases, or interfacing bit-level data with another code.

## The ZX factorization

Every single-qubit Pauli can be written as a product of a Z part and an X part:

| Pauli | ``z`` bit | ``x`` bit | ZX form |
|:-----:|:---------:|:---------:|:--------|
| ``I`` | 0 | 0 | ``I`` |
| ``X`` | 0 | 1 | ``X`` |
| ``Z`` | 1 | 0 | ``Z`` |
| ``Y`` | 1 | 1 | ``ZX = iY`` |

Note the last row: ``ZX = iY``, not ``Y``. This factor of ``i`` per Y site is the origin
of every phase convention in the package.

An ``N``-qubit Pauli string is then encoded as **two integer bitstrings** ``z`` and ``x``,
stored in an unsigned word `W` sized to the register by [`word_type`](@ref) (`UInt64`
up to 64 qubits, `UInt128` up to 128, then `UInt256`/`UInt512`/`UInt1024` from
BitIntegers.jl up to 1024 qubits), where bit ``j`` of each describes site ``j``. We
write the *bare bitstring operator* as

```math
(z|x) \;=\; \bigotimes_{j=1}^{N} Z^{z_j} X^{x_j}.
```

Because each Y site contributes one factor of ``i``, the bare bitstring operator differs
from the true (Hermitian) Pauli string ``P = P_1 \otimes \cdots \otimes P_N`` by a power
of ``i``:

```math
(z|x) = i^{\,n_Y}\, P, \qquad n_Y = \mathrm{popcount}(z \wedge x),
```

where ``n_Y`` is the number of Y sites.

## Bit ordering

Site 1 is the **least-significant bit** of `z`, `x`, and `Ket.v`, and the **first
character** of string constructors:

```julia
p = PauliBasis("XIZ")   # X on site 1 (LSB), Z on site 3
p.x == 0b001            # true
p.z == 0b100            # true

ψ = Ket([1, 0, 1])      # site 1 occupied, site 3 occupied
ψ.v == 0b101            # true (== 5)
```

In dense-matrix form (`Matrix(p)`), site ``N`` is the most-significant qubit of the
standard basis index, i.e. `Matrix(PauliBasis("XIZ")) == kron(Z, I, X)`.

## The symplectic phase

The package defines the **symplectic phase** ``\theta_s`` as the power of ``i`` that
cancels the Y-site phases and recovers the Hermitian Pauli string from the bare
bitstring form:

```math
P = i^{\theta_s}\,(z|x), \qquad \theta_s = (-n_Y) \bmod 4.
```

This is what [`symplectic_phase`](@ref) computes:

```julia
symplectic_phase(p) == (4 - count_ones(p.z & p.x) % 4) % 4
```

The two Pauli types differ only in how they treat this phase:

- **`PauliBasis{N}(z, x)`** *is defined as* ``i^{\theta_s}(z|x) = P`` — always the
  Hermitian, coefficient-free Pauli string. This makes it a canonical dictionary key:
  two `PauliBasis` values are equal iff they are the same Pauli string, with no phase
  ambiguity. This is why `PauliSum` uses `PauliBasis` keys with the phase folded into
  the coefficient value.

- **`Pauli{N}(s, z, x)`** *is defined as* ``s \cdot (z|x) = s\, i^{-\theta_s} P``. The
  scalar `s` multiplies the *bare bitstring form*, not the Hermitian string. The
  effective coefficient in front of the Hermitian string is what [`coeff`](@ref)
  returns:

  ```math
  \mathrm{coeff}(p) = s \cdot i^{-\theta_s} = s \cdot i^{\,n_Y}.
  ```

Storing the scalar against the bare form keeps multiplication cheap (see below) and
makes `Pauli` closed under multiplication; `coeff`/`PauliBasis(p)` convert to the
Hermitian convention at the boundary.

!!! note "Display convention"
    `string(::Pauli)` prints Y sites as a lowercase `y`, meaning ``iY`` — the bare
    ZX-form site operator — because the printed scalar `s` multiplies the bare form.
    `string(::PauliBasis)` prints `Y`, since a `PauliBasis` *is* the Hermitian string.

## Multiplication is XOR plus a popcount

Different sites commute, so multiplying two bare bitstring operators only requires
commuting each ``X^{x_1}`` past each ``Z^{z_2}`` *within* each site. Every such swap
contributes a sign ``(-1)``, giving

```math
(z_1|x_1)\,(z_2|x_2) \;=\; (-1)^{\,\mathrm{popcount}(x_1 \wedge z_2)}\; \bigl(z_1 \oplus z_2 \,\big|\, x_1 \oplus x_2\bigr).
```

This is the entire product rule: **two XORs and one popcount**. In code
(`Base.:*(::Pauli, ::Pauli)`):

```julia
x = p1.x ⊻ p2.x
z = p1.z ⊻ p2.z
s = p1.s * p2.s * 1im^(2 * count_ones(p1.x & p2.z) % 4)
```

Two immediate corollaries:

- **Commutation test.** Reversing the product order swaps the roles of the popcounts,
  so two Paulis commute iff
  ``\mathrm{popcount}(x_1 \wedge z_2) \equiv \mathrm{popcount}(z_1 \wedge x_2) \pmod 2``.
  This is exactly what [`commute`](@ref) evaluates — no multiplication needed.

- **The fused-phase identity.** For two `PauliBasis` (Hermitian) strings, the product
  is again a Pauli string up to a power of ``i``:

  ```math
  P_a P_b = i^{k}\, P_c, \qquad
  k = \bigl(n_Y^{(c)} - n_Y^{(a)} - n_Y^{(b)} + 2\,\mathrm{popcount}(x_a \wedge z_b)\bigr) \bmod 4,
  ```

  with ``z_c = z_a \oplus z_b``, ``x_c = x_a \oplus x_b``. This identity — the phase
  computed *purely from bits*, in one pass — is the workhorse of the optimized kernels:
  `commutator`, `anticommutator`, and the `SparsePauliVector` rotation and
  multiplication kernels all use it. For a commutator, ``[P_a, P_b] = (i^k - i^{k'})P_c``
  collapses to ``2 i^k P_c`` when the operators anticommute (``k`` odd ⇒ the
  coefficient is ``\pm 2i``), so commuting pairs are skipped by the parity test and no
  intermediate products are formed.

## Action on computational-basis states

A `Ket{N}` is an occupation bitstring ``|k\rangle``. Applying a Pauli in ZX form: the
X part flips bits, the Z part contributes a sign read off the flipped state:

```math
(z|x)\,|k\rangle = (-1)^{\,\mathrm{popcount}(z \wedge (k \oplus x))}\; |k \oplus x\rangle .
```

Because a Pauli maps one basis state to exactly one basis state, `p * k` returns a
`(coefficient, Ket)` **tuple**, not a `KetSum`. From this rule the bit-level formulas
for observables follow directly:

- **Expectation value** ``\langle k|P|k\rangle``: nonzero only when ``x = 0``
  (diagonal Pauli), in which case it equals ``(-1)^{\mathrm{popcount}(z \wedge k)}
  \cdot \mathrm{coeff}(P)``. This is why `diag`/`offdiag` filter on `p.x == 0`, and
  why expectation values against product states cost one popcount per diagonal term.

- **Matrix element** ``\langle b|P|k\rangle``: nonzero only when ``b = k \oplus x``,
  i.e. exactly one bra connects to a given ket through a given Pauli.

## The storage word

Every bitstring-carrying type is parameterized on its storage word:
`PauliBasis{N,W}`, `Pauli{N,W}`, `Ket{N,W}`, `Bra{N,W}`, `Dyad{N,W}`,
`DyadBasis{N,W}`, with sum aliases `PauliSum{N,W,T} = Dict{PauliBasis{N,W},T}` and
likewise for `KetSum`/`DyadSum`. The canonical word for a register is
[`word_type`](@ref)`(N)`; registers beyond 1024 qubits throw an `ArgumentError`.

Three rules keep this free of overhead and surprises:

- **`W` is inferred, never computed, on hot paths.** Constructors like
  `PauliBasis{N}(z::W, x::W)` take `W` from their arguments by dispatch; `word_type`
  is only consulted when building from scratch (string constructors, `rand`, empty
  sums). All kernels (`⊻`, `&`, `count_ones`, shifts) are generic over `W<:Unsigned`
  and compile to straight-line code at every width — the BitIntegers types are
  primitive isbits types, so `Dict` hashing and the zero-allocation contract are
  unaffected.
- **Bits above `N` are always zero** (constructor invariant). The typed fast paths
  rely on it and do not re-mask; the `Integer` convenience constructors mask for you.
  Negative `Integer` inputs are two's-complement reinterpreted then masked, so the
  historical `PauliBasis{128}(Int128(-1), Int128(-1))` still means "all 128 bits on".
- **Words never mix.** Binary operations require both operands to share `W`; a
  mismatch is a `MethodError` rather than a silent integer promotion (which would
  corrupt `Dict` key equality). In practice you only ever meet mixed words by
  constructing a non-canonical `W` deliberately.

One consequence of parametric invariance to keep in mind when writing signatures:
`Vector{PauliBasis{N}}` does **not** match a `Vector{PauliBasis{N,UInt64}}` — spell
container element types fully, e.g. `generators::Vector{PauliBasis{N,W}}`.
