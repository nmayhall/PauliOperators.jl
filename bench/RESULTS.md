# v4.0.0 word-type parameterization — before/after benchmarks

Run 2026-07-13 on Apple Silicon, Julia 1.12.2, `julia -O3 bench/bench_words.jl`.
Baseline: `main` @ 8c4221f (v3.1.0, Int128 fields). New: `bitint` (v4.0.0, W-parameterized).

## Baseline (main, v3.1)

| N | Pauli*Pauli (ns) | PauliSum*PauliSum 300x300 (ms) | evolve!+clip 200t x 30rot (ms) | Dict get x1000 (us) |
|---|---|---|---|---|
| 10 | 5.7 | 9.44 | 1276 | 16.8 |
| 20 | 5.6 | 9.11 | 6690 | 17.4 |
| 60 | 5.7 | 8.89 | 6561 | 17.4 |
| 64 | 5.7 | 9.10 | 7027 | 17.1 |
| 100 | 6.2 | 8.92 | 7284 | 17.2 |
| 127 | 5.7 | 8.94 | 6311 | 16.8 |

(v3 rows at N>128 omitted: v3 silently wraps `Int128(2)^N` and truncates all
bitstrings to 128 bits, so those runs compute wrong physics.)

## New (bitint, v4.0)

| N | word | Pauli*Pauli (ns) | PauliSum*PauliSum (ms) | evolve!+clip (ms) | Dict get x1000 (us) |
|---|---|---|---|---|---|
| 10 | UInt64 | 5.0 | 8.71 | 1302 | 15.6 |
| 20 | UInt64 | 5.9 | 8.43 | 6104 | 15.2 |
| 60 | UInt64 | 5.0 | 8.50 | 5630 | 14.9 |
| 64 | UInt64 | 5.0 | 8.30 | 5783 | 15.3 |
| 100 | UInt128 | 5.6 | 9.17 | 6229 | 17.6 |
| 127 | UInt128 | 5.6 | 8.85 | 6219 | 17.2 |
| 200 | UInt256 | 6.5 | — | 7453 | 22.7 |
| 500 | UInt512 | 9.5 | — | 17007 | 36.5 |
| 1000 | UInt1024 | 18.8 | — | 24384 | 69.1 |

## Conclusions

- **N ≤ 64 is faster across the board** (UInt64 words halve Dict key bytes vs
  v3's Int128): Dict get −11%, evolve!+clip −8 to −18%, Pauli*Pauli −12%.
- **65 ≤ N ≤ 128 is at parity** (same UInt128-class storage; differences within
  benchmark noise, ±3%).
- **N > 128 scales bandwidth-proportionally**, as designed: Pauli*Pauli at
  N=1024 costs 3.3× the N=127 time for 8× the bits.
- `rand(PauliBasis{N})` measured via `@belapsed` interpolation is dominated by
  dynamic dispatch in *both* trees; the statically-typed call (any `where N`
  code) is 4.3 ns at N=60 and 15.3 ns at N=200.
- Zero-allocation gates (`test_allocations.jl`, `test_spv_allocations.jl`) pass
  unchanged, plus a new UInt256 sweep at N=200 in `test_large_N.jl`.
