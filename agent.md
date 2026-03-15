# PauliOperators.jl Improvement Tracker

## Project Goals
1. Cleaner API for downstream packages (DBF.jl, OpenSCI.jl, DissipativePauliGroundState.jl)
2. Remove redundant code across packages
3. Fill API gaps (missing operations, unexported symbols)
4. Abstract TruncationStrategy type system with error tracking
5. Systematize evolution methods (single Pauli, sequences, Trotter, QDrift, gates)
6. Create private Julia registry

---

## Phase 1: API Cleanup & Missing Operations (v2.1.0)
**Status: COMPLETE** (all 8401 tests passing)

- [x] Export `commute`, `otimes`, `osum` from module
- [x] Add `KetSum + KetSum`, `KetSum - KetSum` (`src/addition.jl`)
- [x] Add `LinearAlgebra.norm(O, p=2)` for PauliSum and KetSum (`src/norms.jl`)
- [x] Add `Base.isapprox` for PauliSum (`src/norms.jl`)
- [x] Add `diag(::PauliSum)`, `offdiag(::PauliSum)` filters (`src/clip.jl`)
- [x] Add `variance(O::PauliSum, ψ::Ket)`, `covariance(A, B, ψ)` (`src/statistics.jl`)
- [x] Add `majorana_weight`, `majorana_weight_clip!` (`src/clip.jl`)
- [x] Add optimized `commutator(A, B)`, `anticommutator(A, B)` (`src/commutator.jl`)
- [x] Add `clip!` for KetSum (`src/clip.jl`; DyadSum already had it)
- [x] Tests for all additions (`test/test_phase1.jl`)

## Phase 2: Abstract Truncation Strategy System (v2.2.0)
**Status: COMPLETE** (all 8444 tests passing)

- [x] Define `abstract type TruncationStrategy` hierarchy in `src/truncation.jl`
  - NoTruncation, CoeffTruncation, WeightTruncation, MajoranaWeightTruncation
  - StochasticCoeffTruncation (wraps stochastic_clip!), StochasticSamplingTruncation (importance sampling)
  - AdaptiveTruncation (from DissipativePGS), CompositeTruncation
- [x] Implement `_apply!(O::PauliSum, s::TruncationStrategy)` for each type
- [x] Define `abstract type CorrectionAccumulator` hierarchy
  - NoCorrection, EnergyCorrection, EnergyVarianceCorrection
- [x] Implement `truncate!(O, strategy, correction)` protocol
- [x] Tests for truncation system (`test/test_truncation.jl`)

## Phase 3: Systematize Evolution Methods (v2.3.0)
**Status: COMPLETE** (all 8468 tests passing)

- [x] Add KetSum evolution (Schrödinger picture): `evolve(K::KetSum, G::PauliBasis, θ)` (`src/evolve.jl`)
- [x] Add sequence evolution: `evolve(O, generators, angles; truncation, correction)` (`src/evolve.jl`)
- [x] Add `trotterize(H, dt)` and `qdrift(H, dt)` decompositions (`src/decompose.jl`)
- [x] Add quantum gate helpers: hadamard, cnot, X/Y/Z/S/T gates + `_to_paulis` helpers (`src/gates.jl`)
- [x] Tests for all evolution methods (`test/test_evolution.jl`)
- [ ] Refactor `stochastic_propagate` to use TruncationStrategy (deferred)

## Phase 4: Utility Functions & Analysis (v2.4.0)
**Status: NOT STARTED**

- [ ] Weight distribution analysis: `get_weight_counts`, `get_weight_probs`
- [ ] Top-k selection: `find_top_k`, `largest`, `largest_diag`
- [ ] Subspace matrix construction: `Matrix(O, S::Vector{Ket})`
- [ ] Standardize `Base.show` methods (currently mixed show/display)

## Phase 5: Downstream Package Cleanup (v2.5.0)
**Status: NOT STARTED**

- [ ] DBF.jl: remove all redundant code, use PauliOperators directly
- [ ] OpenSCI.jl: update from v1/spinboson branch to v2.5 main, remove DBF dependency
- [ ] DissipativePauliGroundState.jl: remove duplicated evolution/truncation/utils
- [ ] Add docstrings to all exported functions
- [ ] Expand Documenter.jl docs

## Phase 6: Private Registry & Future Architecture (v3.0.0)
**Status: NOT STARTED**

- [ ] Create MayhallRegistry (private Julia registry)
- [ ] Register all packages with proper version bounds
- [ ] Consider: struct wrappers for Sum types (breaking change)
- [ ] Consider: abstract type hierarchy (AbstractPauliOperator, AbstractPauliState)

---

## Redundancy Map (what to absorb from downstream)

| Function | Source Package | Target File in PauliOperators |
|----------|--------------|-------------------------------|
| `weight()` | DBF, DissipativePGS | already in `clip.jl` |
| `coeff_clip!` | DBF | already in `clip.jl` |
| `weight_clip!` | DBF, DissipativePGS | already in `clip.jl` |
| `inner_product()` | DBF | already in `inner_product.jl` |
| `norm(O, p)` | DBF (`norm`, `l1_norm`, `l4_norm`) | new `norms.jl` (unified via standard `p`-norm API) |
| `diag()`, `offdiag()` | DBF | `clip.jl` or `filters.jl` |
| `variance()`, `covariance()` | DBF | new `statistics.jl` |
| `majorana_weight()` | DBF | `clip.jl` |
| `adaptive_clip!()` | DissipativePGS | `truncation.jl` |
| `trotter_evolve()` | DissipativePGS | `evolve_sequence.jl` |
| `TruncationStrategy` hierarchy | DBF | new `truncation.jl` |
| Gate helpers (H, CNOT, etc.) | DBF | new `gates.jl` |
| KetSum +/- | DBF | `addition.jl` |
| KetSum evolve | DBF | `evolve.jl` |

---

## Notes & Decisions
- Existing `clip!`, `coeff_clip!`, `weight_clip!`, `stochastic_clip!` remain unchanged for backward compat
- TruncationStrategy is an opt-in parallel API
- Phase 4 can run in parallel with Phase 3 (independent)
- Phase 5 depends on all prior phases
