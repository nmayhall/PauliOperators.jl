```@meta
CurrentModule = PauliOperators
```

# Data Structures & Performance

The package provides two interchangeable storage engines for sums of Pauli operators:
the `Dict`-backed [`PauliSum`](@ref) and the flat, preallocated
[`SparsePauliVector`](@ref). They implement the same API and produce identical results;
they differ in memory layout and therefore in performance characteristics. This page
explains both layouts, how the fast evolution path works, and how to choose.

## PauliSum: hash-map storage

`PauliSum{N,W,T}` is literally a `Dict{PauliBasis{N,W}, T}`. Each term is a key–value
pair; the [phase conventions](representation.md) make `PauliBasis` a canonical key, so
equal Pauli strings always collide into one entry and addition is `mergewith(+)`. The
key word `W` follows [`word_type`](@ref)`(N)`, so a 40-qubit sum hashes 16-byte keys
while a 1000-qubit sum hashes 256-byte keys — both isbits, both allocation-free.

This is the right default. It is ideal for *constructing and manipulating* operators:

- `O[p] = c`, `haskey`, `delete!` are amortized **O(1)**;
- incremental construction needs no capacity planning;
- every function in the package accepts it.

Its costs show up in *hot loops over all terms*, as in Pauli propagation, where each
rotation touches every term:

- terms are scattered across the hash table's memory, so iteration is cache-unfriendly
  and order is unspecified;
- every insert hashes a `2*sizeof(W)`-byte key and may trigger a rehash (allocation);
- coefficients and keys live in the same slot array, so bandwidth is wasted when a
  kernel needs only keys or only coefficients.

## SparsePauliVector: flat sorted storage

`SparsePauliVector{N,W,T}` stores the same mathematical object as three **parallel
arrays** (structure-of-arrays), sorted by the packed `(z, x)` key:

```
live buffer      z[1:n], x[1:n], c[1:n]     sorted by (z,x), duplicate-free
append buffer    az/ax/ac[1:an]             unsorted, evolve-time sin branches
scratch buffer   sz/sx/sc                   merge output (pointer-swapped with live)
workspace        ws                         sort/merge staging
```

Two type parameters do bandwidth work:

- `W` is the packed key word — the same size-matched `word_type(N)` used by every
  type in the package (`UInt64` when ``N \le 64``, through `UInt1024` at 1024 qubits).
  Constructors choose it automatically.
- `T` is the coefficient type. For Hermitian operators, `T = Float64` is sufficient
  and halves coefficient bandwidth relative to `ComplexF64`. Converting a complex-typed
  `PauliSum` with `T=Float64` checks that all coefficients are numerically real.

The **design contract** (enforced by the test suite): after warm-up, the steady-state
hot path — rotation sweeps, merges, clips, reductions — allocates **zero bytes**. All
storage is preallocated at construction and reused; buffers grow only at window
boundaries, by chunked doubling, when the term population genuinely needs the room.

Because the live keys are sorted:

- lookup (`getindex`, `haskey`, `get`) is a binary search, **O(log n)**;
- `setindex!`/`delete!` shift the tail, **O(n)** — fine for occasional edits, wrong
  for bulk construction. Build with the `PauliSum` API and convert, or accumulate with
  `sum!`;
- addition, `inner_product`, and `isapprox` are two-pointer merges/walks over sorted
  arrays — linear time, allocation-free, no hashing.

### Conversion

```julia
H = PauliSum(N, ComplexF64)          # build with the convenient Dict API
# ... fill H ...
v = SparsePauliVector(H; T=Float64)  # flat storage, real coefficients
# ... hot loop on v ...
O = PauliSum(v)                      # gather back when convenience matters
```

`SparsePauliVector(H; capacity_factor=2.0)` sizes the live buffer with headroom
relative to `length(H)`, so growth between truncations does not immediately force
reallocation.

Methods that only iterate `(PauliBasis, coefficient)` pairs are defined once for the
union `AnyPauliSum{N,W,T} = Union{PauliSum{N,W,T}, SparsePauliVector{N,W,T}}` —
this is why analysis utilities, `truncate!`, and expectation values accept both types.

## The windowed evolution algorithm

The reason `SparsePauliVector` exists is the Heisenberg-picture rotation
``O \mapsto e^{i\theta/2\, G}\, O\, e^{-i\theta/2\, G}`` applied thousands of times in
sequence. Per generator `G`, each term `P` either commutes with `G` (untouched) or
anticommutes, splitting into a cosine and a sine branch:

```math
P \;\mapsto\; \cos\theta\, P \;\pm\; \sin\theta\, P',
\qquad P' = \text{bits } G \oplus P .
```

On flat storage this is a **linear sweep**: commuting terms are skipped by the parity
test (two popcounts), anticommuting terms are cosine-scaled in place, and the sine
branch — key, sign, and coefficient computed purely from bits via the fused-phase
identity (see [Pauli Representation](representation.md)) — is *appended* to the append
buffer rather than deduplicated immediately. Deduplication is deferred to a **merge
boundary**: the appends are sorted and merged into the sorted live buffer with a single
two-pointer pass, summing duplicate keys and applying the truncation filter to each
output term as it is produced. The scratch and live buffers are then pointer-swapped —
no copy.

When the pending appends come from a *single* rotation (always the case at
`window = 1`), even the sort is avoided: the appends were generated by scanning the
sorted live buffer and XORing each key with the generator mask, so they are already
sorted with respect to `key ⊻ mask`. One streaming block-swap pass per set mask bit
restores natural order — `weight(G)` linear passes instead of an O(m log m)
comparison sort. Multi-rotation merges (`window > 1`) fall back to the comparison
sort.

`evolve!(v, generators, angles; window=k, ...)` controls the cadence:

- **`window = 1`** (default) merges after every rotation and reproduces the `Dict`
  path `evolve!(O, g, θ); truncate!(O, strategy, correction)` **exactly** — same terms,
  same drop decisions, same correction accumulation. Use this when you need bit-exact
  parity with `PauliSum` results.
- **`window > 1`** performs `k` rotation sweeps between merges, amortizing the sort
  and dedup over the window at the cost of a laxer truncation cadence (the population
  transiently carries unmerged duplicates).

Two supporting knobs:

- `local_truncation`: a cheap deterministic filter applied per appended term at
  rotation time (weight cutoffs are exact there; coefficient cutoffs act on unmerged
  duplicates). It bounds append growth inside a window without waiting for the merge.
- If a rotation's worst-case appends cannot fit the append buffer, an **early merge**
  is triggered automatically — harmless, it only changes when truncation happens — and
  buffers grow at the boundary if genuinely needed.

### Instrumentation

Pass a `WindowCounters(nwindows)` to `evolve!` to record, per window: terms created,
merge input/output sizes, rotation and merge wall time, early merges, and — the
zero-allocation contract made checkable — the exact bytes allocated
(`counters.allocd`, which should be all zeros after warm-up):

```julia
counters = PauliOperators.WindowCounters(cld(length(gens), window))
evolve!(v, gens, angs; window, truncation=strat, counters)
@assert all(==(0), counters.allocd[2:end])
```

## Truncation inside the kernels

Deterministic strategies (`CoeffTruncation`, the weight cutoffs, the damped variants,
and compositions of them) are **compiled once** into a branch-light per-term predicate
(`MergeFilter`) and fused into the merge and compaction kernels — no dynamic dispatch
per term. Stochastic and adaptive strategies cannot be evaluated per-term in isolation
and instead run through the generic `truncate!` machinery at merge boundaries. See
[Truncation](truncation.md) for the semantics of each strategy.

## Choosing an engine

| Situation | Use |
|:---|:---|
| Building/editing operators, one-off algebra, readability | `PauliSum` |
| Long rotation sequences (Pauli propagation, Trotter sweeps) | `SparsePauliVector` |
| Hermitian operator, real coefficients suffice | `SparsePauliVector` with `T=Float64` |
| Frequent random single-term inserts/deletes in a hot loop | `PauliSum` (flat `setindex!` is O(n)) |
| Need allocation-free `commutator!`/`sum!`/reductions | `SparsePauliVector` |

The practical workflow is almost always: **build as `PauliSum`, convert, evolve,
measure, convert back if needed**. Expectation values, norms, and the analysis
utilities work directly on the flat form (allocation-free), so conversion back is only
for convenience.
