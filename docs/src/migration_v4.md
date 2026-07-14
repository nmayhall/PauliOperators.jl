```@meta
CurrentModule = PauliOperators
```

# Migrating to v4 (word-type parameterization)

v4.0.0 extends the package from 128 to **1024 qubits** by parameterizing every
bitstring-carrying type on its unsigned storage word `W`, chosen from `N` by
[`word_type`](@ref):

| Qubits | Word |
|:---|:---|
| ``N \le 64`` | `UInt64` |
| ``N \le 128`` | `UInt128` |
| ``N \le 256`` | `UInt256` (BitIntegers.jl) |
| ``N \le 512`` | `UInt512` |
| ``N \le 1024`` | `UInt1024` |

The type changes:

| v3 | v4 |
|:---|:---|
| `PauliBasis{N}` with `z::Int128, x::Int128` | `PauliBasis{N,W}` with `z::W, x::W` |
| `Pauli{N}` | `Pauli{N,W}` |
| `Ket{N}` / `Bra{N}` with `v::Int128` | `Ket{N,W}` / `Bra{N,W}` with `v::W` |
| `Dyad{N}` / `DyadBasis{N}` | `Dyad{N,W}` / `DyadBasis{N,W}` |
| `PauliSum{N,T} = Dict{PauliBasis{N},T}` | `PauliSum{N,W,T} = Dict{PauliBasis{N,W},T}` |
| `KetSum{N,T}` / `DyadSum{N,T}` | `KetSum{N,W,T}` / `DyadSum{N,W,T}` |
| `AnyPauliSum{N,T}` | `AnyPauliSum{N,W,T}` |

This also makes small systems *faster*: `N ≤ 64` now stores `UInt64` bitstrings,
halving `Dict` key size versus the old fixed `Int128` (measured: −11% on lookups,
−8 to −18% on `evolve!` sweeps; see `bench/RESULTS.md` in the repo).

## What does not change

**All value-level code.** `W` is inferred everywhere:

```julia
PauliSum(N)                  # picks W = word_type(N)
Pauli("XYZIZ")               # W from the string length
PauliBasis{N}(z, x)          # W from word_type(N); any Integer accepted
Ket(N, v); rand(Ket{N})      # unchanged
rand(PauliSum{N}; n_paulis)  # unchanged
evolve!, truncate!, trotterize, expectation_value, ...   # unchanged calls
```

Constructors still accept any `Integer` (including the old `Int128` literals);
negative values are two's-complement reinterpreted and masked to `N` bits, so
`PauliBasis{128}(Int128(-1), Int128(-1))` still means "all 128 bits on".

## What you must change

**1. Three-parameter sum aliases in signatures.** A method written against the v3
two-parameter alias now binds `W := T` and will not match (you get a `MethodError`
at call time, never silent wrong behavior *unless* the method body also used `T` —
so rewrite, don't leave these):

```julia
# v3
function f(O::PauliSum{N,T}) where {N,T} ... end
# v4 — either carry W:
function f(O::PauliSum{N,W,T}) where {N,W,T} ... end
# or, if the body never names W or T:
function f(O::PauliSum{N}) where {N} ... end     # still matches every W and T
```

**2. Container element types are invariant.** `Vector{PauliBasis{N}}` is a vector
of the *UnionAll* and does not match concrete vectors anymore:

```julia
# v3
g(gens::Vector{PauliBasis{N}}) where {N}
# v4
g(gens::Vector{PauliBasis{N,W}}) where {N,W}
```

The same applies to `Dict`/`Set`/`Pair`/`Adjoint` type parameters. For a union
inside an invariant position, close over `W` explicitly, e.g.
`Adjoint{<:Any, <:(PauliSum{N,W,T} where W)}`.

**3. Raw field arithmetic.** `p.z`/`p.x`/`k.v` are now unsigned (`UInt64`,
`UInt128`, `UInt256`, ...) instead of `Int128`. Bit operations (`⊻`, `&`, `|`,
shifts, `count_ones`) are unaffected. Things to audit:

- literals: replace `Int128(1) << i` with `one(W) << i` (get `W` from a type
  parameter, or `typeof(p.z)`);
- masks: `typemax(W) >> (8*sizeof(W) - n)` builds an `n`-bit mask (exported helper:
  `PauliOperators._nbit_mask(W, n)`);
- hardcoded `Dict{Int128, ...}` caches keyed by bitstrings should key on `W`;
- anything relying on signed behavior of the old fields (rare — the package itself
  had exactly one such site).

**4. Never mix words.** Operations between operands of different `W` are
`MethodError`s by design — don't "fix" one by converting a bitstring with `%`;
construct both objects at the same `N` (canonical `W`) instead.

## New capabilities

```julia
H = PauliSum(200)                      # Dict path at 200 qubits, UInt256 words
v = SparsePauliVector(H; T=Float64)    # flat engine, same W
rand(Pauli{1000})                      # UInt1024 words
jordan_wigner(150, 300)                # masks built at word width
Matrix(O, S)                           # subspace matrices work above 128 qubits
```

Dense conversions (`Matrix(::PauliSum)` without a subspace, `Vector(::Ket)`) remain
inherently limited to small `N` — they materialize ``2^N`` objects regardless of the
word width.

Note that v3 did **not** error above 128 qubits — it silently wrapped
`Int128(2)^N` masks and truncated every bitstring to 128 bits, computing wrong
results. v4 computes correctly through `N = 1024` and throws an `ArgumentError`
beyond.
