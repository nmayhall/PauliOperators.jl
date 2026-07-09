# Shared-Memory Sharded Engine

`ShardedPauliSum` is a multithreaded, single-node engine for Heisenberg-picture
Pauli propagation of large `PauliSum`s (10⁸–10¹⁰ terms) under long sequences of
low-weight Pauli rotations. It replaces `Dict` storage with flat, preallocated
structure-of-arrays shards partitioned by a GF(2) `RankMap`, and treats **zero
steady-state allocation as a correctness property**: GC stop-the-world pauses
serialize all threads, so at high thread counts even a small GC fraction
destroys scaling. The test suite fails if the rotation or merge kernels
allocate, or if a post-warm-up window shows a nonzero `Base.gc_num()` delta.

## Quick start

```julia
using PauliOperators

N = 24
H = ...                                        # your Hamiltonian, PauliSum{N}
gens, angs = trotterize(H, 0.05, n_trotter=10, order=2)

O = PauliSum(N, Float64)                       # real coefficients halve bandwidth
O[PauliBasis(Pauli(N, Z=[N ÷ 2]))] = 1.0       # observable to propagate

A    = rand(RankMap{N}, 8)                     # 256 shards
S    = ShardedPauliSum(O, A; nthreads=Threads.nthreads(), min_capacity=1 << 16)
circ = compile(A, gens, angs; window=8)        # merge every 8 rotations

pin_engine!(S)                                 # no-op unless `using ThreadPinning`
evolve!(S, circ;
        truncation       = WeightDampedTruncation(0.3, 1e-8),  # strict, at merges
        local_truncation = CoeffTruncation(1e-10),             # loose, at append
        rebalance_threshold = 1.25,
        counters = WindowCounters(length(circ.window_subgroups)))

result = PauliSum(S)                           # gather back to a Dict-based sum
```

Julia must be started with enough threads (`--threads=...`) for the
`nthreads` the engine was built with.

## How it works

- **Sharding.** Every term lives in shard `bin_index(A, P) + 1`. Because
  `bin(G·P) = bin(P) ⊻ bin(G)`, a rotation moves sin-branch terms from shard
  `k` to exactly one partner shard `k ⊻ bin_shift(A, G)`, known at compile
  time. Duplicates always co-locate, so dedup and coefficient truncation are
  shard-local.
- **Storage.** Each shard holds a sorted live buffer (strictly increasing
  `(z, x)` keys), an unsorted append buffer segmented by source thread, and a
  scratch buffer. Merges are sort-based (radix-swappable in-place quicksort +
  two-pointer merge), never hash-based. Buffers grow by chunked doubling at
  window boundaries only.
- **Threading.** Shard ownership (`S.owner`) is the only load-balancing
  mechanism. A fixed pool of workers (spawned once per `evolve!`) runs the
  windowed loop under a sense-reversing spin barrier (with GC safepoints).
  Every write is single-writer by construction: owners sweep owned shards,
  thread `t` appends only into segment `t`, and sweep bounds are snapshotted
  by owners in a barrier-protected phase.
- **Cadence.** Within a window, rotations only append (cos branch scales in
  place); at the boundary each owner sorts and merges its shards under the
  strict truncation. `window` (the `compile` keyword `M`) is the primary
  tuning knob: larger M amortizes merges but delays strict truncation.
  `window = 1` reproduces the serial `evolve` **bit-exactly** for any thread
  count; `window > 1` agrees up to floating-point reduction order plus the
  documented truncation-cadence difference. Capacity pressure triggers early
  merges (extra truncation events — semantically harmless, but they make runs
  with different `nthreads` diverge slightly under truncation; size
  `min_capacity`/`append_factor` generously to avoid them).

## Choosing the rank map `A`

- **Size**: aim for 8–32 shards per thread, i.e. `r ≈ log2(nthreads) + 4`
  rows. More shards = finer rebalancing granularity but more merge overhead.
- **Random maps** (`rand(RankMap{N}, r)`) give near-perfect statistical
  balance and are the right default for generic circuits.
- **Protected generators**: `RankMap{N}(r; protected=gens)` draws rows with
  even overlap against every listed generator mask, making those rotations
  communication-free (`bin_shift == 0`, zero cross-shard appends — tested as
  an invariant). Protecting a frequent Trotter family (e.g. all diagonal ZZ
  layers) removes its cross-shard traffic entirely, at the cost of
  constraining the row space (each independent protected mask removes one
  dimension from the 2N-dimensional GF(2) row space).
- **Geometric rows** for lattice models: a row watching the x-slots of all
  sites on one side of a spatial cut is a domain decomposition — only
  hopping terms crossing the cut shift shards, and every Z-only interaction
  term is automatically protected. Watch out for the **diagonal trap**: an
  x-only map sends every Z-string to shard 0. Add same-site z-pair rows
  (`RankRow(N, z=[q↑, q↓])`) to split the diagonal sector while keeping
  on-site ZZ generators protected. See
  `examples/hubbard2d_sharded_benchmark.jl` and `test/test_geometric_rankmap.jl`.
- **Greedy bisection**: `greedy_bisection_rankmap(S, r; protected=...)` draws
  candidate rows against the *live population* and picks the best-balancing
  ones. Build a fresh engine with the returned map (changing `A` is a
  reconstruction; ownership rebalancing via `rebalance_threshold` is the
  cheap in-flight knob).

## Truncation semantics

- `truncation` (strict) applies during merges to fully-merged coefficients —
  equivalent to serial per-rotation `truncate!` at `window = 1`.
- `local_truncation` (loose) applies at append time: weight-type cutoffs are
  exact there; coefficient cutoffs act on unmerged duplicates and should be
  well below the strict threshold. Its losses bypass correction accumulators.
- `AdaptiveTruncation` picks its global threshold from tree-reduced 64-bin
  |c| exponent histograms with a one-window lag, re-clipping immediately if
  the population overshoots 2× the budget. The threshold is quantized to
  power-of-two bin edges, so the kept count lands within a factor-of-2 band
  of the serial exact top-k semantics.
- `EnergyCorrection` accumulates ⟨ψ|·|ψ⟩ changes at every truncation event
  (window boundaries and early merges), reduced across threads. Stochastic
  strategies and `EnergyVarianceCorrection` are unsupported and error.

## Instrumentation

Pass `counters = WindowCounters(nwindows)` to `evolve!` for per-window terms
created, cross-shard appends, merge input/output sizes, per-phase wall time,
shard-population extrema, early-merge count, and the `Base.gc_num()`
allocation delta — **any nonzero `allocd` entry after the first window in
steady state is a bug**, not a tuning issue.

## Distributed (MPI) seam — audit notes

The engine is milestones 1–2 of the larger distributed architecture with
shared memory as the transport. The seams a future MPI backend plugs into,
without touching kernels:

1. **`merge_shards!` is the only boundary function**: on one node it
   sort-merges in place; a distributed method ships non-owned segments first
   (the append segments are already contiguous, isbits send buffers).
2. **`owner::Vector{Int32}` generalizes to (node, thread)** — the shard
   table is the routing layer, exactly as `bin_owner` is for
   `DistributedPauliSum` on the `distributed2` branch.
3. **`CompiledCircuit.window_subgroups`** (the GF(2) span of each window's
   shifts) is already the exchange schedule: it bounds which shard
   displacements — hence which XOR partners — can hold traffic at each
   boundary.
4. **Reductions** (`norm`, `expectation_value`, histograms, corrections) are
   per-thread partials + a tree reduce; an `Allreduce` slots in above the
   thread reduction unchanged.
