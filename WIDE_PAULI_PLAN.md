# Plan: wide (multi-word) Pauli representation + flexible coefficient types

Branch: `wide-pauli-multiword`. Goal: run Pauli-operator (Heisenberg-picture)
evolution past 128 qubits — target a 10x10x10 = 1000-qubit Heisenberg lattice —
and allow different coefficient float types. Prerequisite for any multinode SPD.

## Why this is needed

`PauliBasis{N}` and `Pauli{N}` store the symplectic bitstrings `z`, `x` as scalar
`Int128` (128 bits). That hard-caps the library at N <= 128 qubits. Every bit
kernel (`count_ones(z & x)`, `z ⊻ z`, `z & x`, `two^idx`, `<<`) is scalar. 1000
qubits needs 1000 bits per string, so the type must widen. This is independent of
multinode: nothing above 128 sites runs at all today.

## Reference: how other libraries do it

- **PauliStrings.jl** (Julia, does SPD-style Pauli dynamics) is the closest match.
  `PauliString{N,T<:Unsigned}` with `v,w::T`; a `uinttype(N)` helper picks the
  width: `UInt8`..native `UInt128`, then `BitIntegers.UInt256/512/1024`, then
  dynamically-defined wider ints. Coefficients are a separate type parameter
  (`Operator{P,T<:Number}`, default `ComplexF64`).
- **Qiskit** `PauliList`/`SparsePauliOp`: packed boolean `z`,`x` arrays.
- **Stim**: SIMD word-packed bit arrays for thousands of qubits.

Takeaway: the idiomatic Julia route is **BitIntegers.jl** — fixed-width unsigned
integers of arbitrary width that support `count_ones`/`&`/`|`/`⊻`/`<<`/`^`
transparently. Our existing scalar kernels then work almost verbatim; no
hand-rolled `NTuple{W,UInt64}` needed.

## Design

1. **Parameterize the integer width.**
   - `struct PauliBasis{N,T<:Unsigned}; z::T; x::T; end`
   - `struct Pauli{N,T<:Unsigned}; s::ComplexF64; z::T; x::T; end`
   - Likewise `Ket{N,T}`, `Bra{N,T}`, `Dyad`/`DyadBasis`.
   - Add `uinttype(N)` (mirror PauliStrings): `UInt8` for N<=8, `nextpow(2,N)`
     native up to 128, else `BitIntegers.UInt{256,512,1024,...}`.
   - Backward-compatible convenience constructors keep `PauliBasis{N}(z,x)` and
     `Pauli{N}(...)` working by defaulting `T = uinttype(N)` and converting.
   - Method signatures written as `::PauliBasis{N}` still match (UnionAll over T),
     so most method bodies are untouched; only constructors and `Int128(...)`
     literals change to `T(...)` / `uinttype(N)(...)`.

2. **Coefficient float type (the "different float types" ask).**
   - `PauliSum{N,T} = Dict{PauliBasis{N,uinttype(N)},T}` already parameterizes the
     coefficient `T`. Keep that; ensure `evolve`/`sum!`/`clip` preserve `T`
     (Float32 / ComplexF32 / Float64 / ComplexF64) instead of hardcoding
     `ComplexF64`/`0.0` literals.
   - `Pauli.s` stays `ComplexF64` (phase bookkeeping); coefficient precision lives
     in the sum's `T`. (Optional later: parameterize `Pauli` scalar too.)

3. **Container.** Keep the `Dict`-based `PauliSum` for now (minimal churn). Note:
   PauliStrings uses struct-of-arrays (`Vector{P}` + `Vector{T}`), which is faster
   and more memory-compact for the large-sum SPD regime and is the natural shard
   unit for multinode — a candidate follow-up, not part of this branch.

## Step list (incremental, test after each)

1. Add `BitIntegers` to `Project.toml` deps + compat. Add `uinttype(N)` (new
   `src/width.jl` or into `helpers.jl`). [low risk, non-breaking]
2. Parameterize `type_PauliBasis.jl` and `type_Pauli.jl` to `{N,T}`; replace
   `Int128` literals with `T`/`uinttype(N)`; keep back-compat constructors.
3. Propagate through `type_Ket.jl`, `type_Bra`, `type_Dyad*`, `type_KetSum`,
   `type_DyadSum`, `type_PauliSum.jl`.
4. Fix `Int128` literals in kernels: `commutator.jl`, `multiplication.jl`,
   `addition.jl`, `clip.jl`, `transformations.jl`, `channels.jl`, `analysis.jl`.
5. Make coefficient-type generic in `evolve`/`sum!`/`clip`/`addition`
   (no hardcoded `ComplexF64`/`0.0`; use `zero(T)` / `T`).
6. Run the existing suite (`test/runtests.jl`) at N<=128 to prove no regression,
   then add tests at N=200 and N=1000 (`UInt256`, `UInt1024`) for
   construct/multiply/commute/evolve/clip.
7. Memory/term-count estimator for SPD at target threshold (feeds the
   multinode-or-not decision).

## After this branch

- Decide multinode level: sample-parallel (exists) vs sharded single Pauli sum.
- If sharded: shard `PauliSum` by hash of `PauliBasis` across nodes; distributed
  `evolve!` routes sin-branch terms `G·p` to owners, merges, clips — same
  hash-sharded pattern as `DistributedTPSCIstate` in TPSChem.jl.
