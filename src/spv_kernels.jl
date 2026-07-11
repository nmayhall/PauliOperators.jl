# ============================================================
# SparsePauliVector kernels: rotation sweep, append sort, sorted merge,
# in-place compaction, histograms. Ported from the shared-memory sharded
# engine (branch shared1) — a SparsePauliVector is a single un-sharded
# shard, and these kernels are the seam the future threaded engine reuses.
#
# These are the steady-state hot path and MUST allocate zero bytes — the
# test suite enforces this with @ballocated. Rules: isbits arguments and
# plain Vectors only, no closures, no strings, no dynamic dispatch.
# ============================================================

# Lexicographic (z, x) key order, z-major. Works on 2-tuples and on the
# (z, x, c) workspace triples (compares the first two fields).
@inline _key_lt(a::Tuple, b::Tuple) =
    (a[1] < b[1]) | ((a[1] == b[1]) & (a[2] < b[2]))
@inline _key_eq(a::Tuple, b::Tuple) = (a[1] == b[1]) & (a[2] == b[2])

# ------------------------------------------------------------
# Truncation filter, compiled once per evolve!/clip from a
# TruncationStrategy — an isbits predicate evaluated per term with no
# dynamic dispatch.
# ------------------------------------------------------------

"""
    MergeFilter

Compiled truncation predicate for the SparsePauliVector kernels. Sentinels
disable individual checks: `typemax(Int)` for the weight cutoffs, negative
thresholds for the coefficient cutoffs (`thresh = -1.0` keeps exact zeros,
matching `NoTruncation`; `coeff_clip!` semantics are "drop |c| <= thresh").
Built from a `TruncationStrategy` by `_compile_filter`.
"""
struct MergeFilter
    wmax::Int
    xwmax::Int
    mwmax::Int
    thresh::Float64
    alpha_w::Float64
    dthresh_w::Float64
    alpha_xw::Float64
    dthresh_xw::Float64
end

const NOFILTER = MergeFilter(typemax(Int), typemax(Int), typemax(Int), -1.0,
                             0.0, -1.0, 0.0, -1.0)

# Branchless-suffix-parity Majorana weight on packed words; the word-level
# analogue of `majorana_weight(::PauliBasis)` (see clip.jl for the
# derivation). The shift cascade covers 8*sizeof(W) bits, so it stays
# correct for wider (BitIntegers.jl) word types.
@inline function _majorana_weight_bits(z::W, x::W) where {W<:Unsigned}
    zonly = z & ~x
    S = x
    shift = 1
    while shift < 8 * sizeof(W)
        S ⊻= S >> shift
        shift <<= 1
    end
    ctrl = ~(S ⊻ x)
    return count_ones(x) + 2 * count_ones(zonly & ctrl) +
           2 * count_ones(~(z | x) & ~ctrl)
end

@inline function should_drop(f::MergeFilter, z::W, x::W, absc::Float64) where {W<:Unsigned}
    absc <= f.thresh && return true
    w = count_ones(z | x)
    w > f.wmax && return true
    count_ones(x) > f.xwmax && return true
    if f.mwmax != typemax(Int)
        _majorana_weight_bits(z, x) > f.mwmax && return true
    end
    if f.dthresh_w >= 0.0
        absc * exp(-f.alpha_w * w) <= f.dthresh_w && return true
    end
    if f.dthresh_xw >= 0.0
        absc * exp(-f.alpha_xw * count_ones(x)) <= f.dthresh_xw && return true
    end
    return false
end

_compile_filter(::NoTruncation) = NOFILTER
_compile_filter(s::CoeffTruncation) =
    MergeFilter(typemax(Int), typemax(Int), typemax(Int), s.thresh, 0.0, -1.0, 0.0, -1.0)
_compile_filter(s::WeightTruncation) =
    MergeFilter(s.max_weight, typemax(Int), typemax(Int), -1.0, 0.0, -1.0, 0.0, -1.0)
_compile_filter(s::XWeightTruncation) =
    MergeFilter(typemax(Int), s.max_weight, typemax(Int), -1.0, 0.0, -1.0, 0.0, -1.0)
_compile_filter(s::MajoranaWeightTruncation) =
    MergeFilter(typemax(Int), typemax(Int), s.max_weight, -1.0, 0.0, -1.0, 0.0, -1.0)
_compile_filter(s::WeightDampedTruncation) =
    MergeFilter(typemax(Int), typemax(Int), typemax(Int), -1.0, s.alpha, s.thresh, 0.0, -1.0)
_compile_filter(s::XWeightDampedTruncation) =
    MergeFilter(typemax(Int), typemax(Int), typemax(Int), -1.0, 0.0, -1.0, s.alpha, s.thresh)
_compile_filter(s::TruncationStrategy) =
    error("$(typeof(s)) cannot be compiled to a MergeFilter (internal misuse: " *
          "check _is_compilable before compiling)")

function _combine_filters(f1::MergeFilter, f2::MergeFilter)
    (f1.dthresh_w >= 0 && f2.dthresh_w >= 0) &&
        error("cannot combine two weight-damped truncations in one composite")
    (f1.dthresh_xw >= 0 && f2.dthresh_xw >= 0) &&
        error("cannot combine two x-weight-damped truncations in one composite")
    dw  = f1.dthresh_w  >= 0 ? f1 : f2
    dxw = f1.dthresh_xw >= 0 ? f1 : f2
    return MergeFilter(min(f1.wmax, f2.wmax), min(f1.xwmax, f2.xwmax),
                       min(f1.mwmax, f2.mwmax), max(f1.thresh, f2.thresh),
                       dw.alpha_w, dw.dthresh_w, dxw.alpha_xw, dxw.dthresh_xw)
end

_compile_filter(s::CompositeTruncation) =
    foldl(_combine_filters, map(_compile_filter, s.strategies); init=NOFILTER)

# Strategies whose exact semantics compile into a single per-term
# MergeFilter predicate. Everything else (stochastic, adaptive, user
# subtypes) goes through the generic `_apply!` at merge boundaries.
_is_compilable(::Union{NoTruncation, CoeffTruncation, WeightTruncation,
                       XWeightTruncation, MajoranaWeightTruncation,
                       WeightDampedTruncation, XWeightDampedTruncation}) = true
_is_compilable(::TruncationStrategy) = false
function _is_compilable(s::CompositeTruncation)
    all(_is_compilable, s.strategies) || return false
    count(x -> x isa WeightDampedTruncation, collect(s.strategies)) <= 1 || return false
    count(x -> x isa XWeightDampedTruncation, collect(s.strategies)) <= 1 || return false
    return true
end

# ------------------------------------------------------------
# Rotation kernel
# ------------------------------------------------------------

"""
    _rotate_range!(z, x, c, lo, hi, gz, gx, n_g, cosθ, sinθ,
                   dz, dx, dc, cur, cap_end, f) -> (cur, created, overflowed)

Sweep terms `lo:hi` of one buffer under the rotation `exp(iθ/2 G)·O·exp(-iθ/2 G)`
(the `evolve!(O, G, θ)` convention — cos(θ)/sin(θ), half-angle absorbed):
commuting terms untouched; anticommuting terms cos-scaled in place, with
the sin branch (bits `G ⊻ P`, sign `i·i^k = ±1` computed purely from bits —
the fused-phase identity from commutator.jl) appended at `cur` in the
destination arrays unless the local filter drops it. `n_g` is
`count_ones(gz & gx)`, precomputed once per rotation.

Zero-allocation hot path. `overflowed` only fires if the driver's capacity
precheck was skipped or wrong; the driver treats it as an error.
"""
@inline function _rotate_range!(z::Vector{W}, x::Vector{W}, c::Vector{T},
                                lo::Int, hi::Int,
                                gz::W, gx::W, n_g::Int, cosθ::Float64, sinθ::Float64,
                                dz::Vector{W}, dx::Vector{W}, dc::Vector{T},
                                cur::Int, cap_end::Int,
                                f::MergeFilter) where {W<:Unsigned, T<:Number}
    created = 0
    overflowed = false
    @inbounds for i in lo:hi
        zi = z[i]
        xi = x[i]
        m1 = count_ones(gx & zi)
        m2 = count_ones(gz & xi)
        iseven(m1 - m2) && continue
        zp = gz ⊻ zi
        xp = gx ⊻ xi
        k = (count_ones(zp & xp) - n_g - count_ones(zi & xi) + 2 * m1) & 3
        cnew = (T(k - 2) * sinθ) * c[i]
        c[i] *= cosθ
        should_drop(f, zp, xp, abs(cnew)) && continue
        if cur > cap_end
            overflowed = true
            continue
        end
        dz[cur] = zp
        dx[cur] = xp
        dc[cur] = cnew
        cur += 1
        created += 1
    end
    return cur, created, overflowed
end

# ------------------------------------------------------------
# Sort + merge (window boundary)
# ------------------------------------------------------------

"""
Gather the pending appends `az/ax/ac[1:an]` into the workspace as
(z, x, c) triples. Returns the count. Allocation-free.
"""
function _gather_append!(v::SparsePauliVector)
    m = v.an
    @inbounds for i in 1:m
        v.ws[i] = (v.az[i], v.ax[i], v.ac[i])
    end
    return m
end

"""
In-place quicksort (median-of-3, insertion sort below 24, recurse-smaller /
iterate-larger) of tuples by their first two fields — the (z, x) key for
merge-workspace triples. Hand-rolled because Base's default QuickSort
allocates scratch; this is the swap point for a future radix sort.
Allocation-free.
"""
function _sort_ws!(ws::Vector{TT}, lo::Int, hi::Int) where {TT<:Tuple}
    @inbounds while hi - lo >= 24
        mid = (lo + hi) >>> 1
        if _key_lt(ws[mid], ws[lo])
            ws[mid], ws[lo] = ws[lo], ws[mid]
        end
        if _key_lt(ws[hi], ws[mid])
            ws[hi], ws[mid] = ws[mid], ws[hi]
            if _key_lt(ws[mid], ws[lo])
                ws[mid], ws[lo] = ws[lo], ws[mid]
            end
        end
        pivot = ws[mid]
        i, j = lo, hi
        while i <= j
            while _key_lt(ws[i], pivot)
                i += 1
            end
            while _key_lt(pivot, ws[j])
                j -= 1
            end
            if i <= j
                ws[i], ws[j] = ws[j], ws[i]
                i += 1
                j -= 1
            end
        end
        if j - lo < hi - i
            _sort_ws!(ws, lo, j)
            lo = i
        else
            _sort_ws!(ws, i, hi)
            hi = j
        end
    end
    @inbounds for k in lo+1:hi
        v = ws[k]
        m = k - 1
        while m >= lo && _key_lt(v, ws[m])
            ws[m+1] = ws[m]
            m -= 1
        end
        ws[m+1] = v
    end
    return ws
end

# Word / high-prefix accessors for _unshuffle_ws! passes. The combined sort
# key is (z, x) lexicographic: z is the high word. The "high prefix" above
# bit b of the z word ignores x entirely; above bit b of the x word it
# includes all of z.
@inline _uw(t::Tuple{W,W,T}, ::Val{1}) where {W,T} = t[1]
@inline _uw(t::Tuple{W,W,T}, ::Val{2}) where {W,T} = t[2]
@inline _uhp(t::Tuple{W,W,T}, hb::W, ::Val{1}) where {W,T} = (t[1] & hb, zero(W))
@inline _uhp(t::Tuple{W,W,T}, hb::W, ::Val{2}) where {W,T} = (t[1], t[2] & hb)

# One streaming block-swap pass of the unshuffle cascade: input sorted by
# key ⊻ bit(word, b) (higher bits already in natural order). Within each run
# of equal higher bits, the chunk with bit = 1 precedes the chunk with
# bit = 0 — emit them swapped. Stable, allocation-free, O(m).
function _unshuffle_pass!(src::Vector{Tuple{W,W,T}}, dst::Vector{Tuple{W,W,T}},
                          m::Int, b::Int, wv::Val) where {W,T}
    hb = ~((one(W) << (b + 1)) - one(W))   # bits above b (0 for the top bit)
    i = 1
    @inbounds while i <= m
        hp = _uhp(src[i], hb, wv)
        j = i
        while j + 1 <= m && _uhp(src[j+1], hb, wv) == hp
            j += 1
        end
        f = i
        while f <= j && (_uw(src[f], wv) >> b) & one(W) == one(W)
            f += 1
        end
        k = i
        for s in f:j
            dst[k] = src[s]; k += 1
        end
        for s in i:f-1
            dst[k] = src[s]; k += 1
        end
        i = j + 1
    end
    return nothing
end

"""
    _unshuffle_ws!(v, m, gz, gx)

Sort `v.ws[1:m]` by (z, x) key, exploiting rotation structure: sin-branch
appends are generated by scanning the sorted live buffer and XORing each
key with the generator mask `(gz, gx)`, so the gathered triples are already
sorted with respect to `key ⊻ mask`. One streaming block-swap pass per set
mask bit (highest combined bit first) restores natural order —
`weight(G)` linear passes instead of an O(m log m) comparison sort.
Ping-pongs between `v.ws` and the lazily grown `v.ws2`; the result always
ends in `v.ws`. Falls back to nothing-to-do for the identity mask.
"""
function _unshuffle_ws!(v::SparsePauliVector{N,W,T}, m::Int, gz::W, gx::W) where {N,W,T}
    m <= 1 && return v.ws
    count_ones(gz) + count_ones(gx) == 0 && return v.ws
    length(v.ws2) < m && resize!(v.ws2, length(v.ws))
    src, dst = v.ws, v.ws2
    flips = 0
    for b in (8 * sizeof(W) - 1):-1:0
        (gz >> b) & one(W) == one(W) || continue
        _unshuffle_pass!(src, dst, m, b, Val(1))
        src, dst = dst, src
        flips += 1
    end
    for b in (8 * sizeof(W) - 1):-1:0
        (gx >> b) & one(W) == one(W) || continue
        _unshuffle_pass!(src, dst, m, b, Val(2))
        src, dst = dst, src
        flips += 1
    end
    isodd(flips) && copyto!(v.ws, 1, v.ws2, 1, m)
    return v.ws
end

# Single merge pass: live + m sorted workspace triples → scratch. Returns
# (out, overflowed); on overflow nothing is swapped and live is untouched
# (the merge only reads live/ws and writes scratch), so the caller can grow
# and retry.
function _try_merge!(v::SparsePauliVector{N,W,T}, m::Int, f::MergeFilter) where {N,W,T}
    n = v.n
    z, x, c = v.z, v.x, v.c
    sz, sx, sc = v.sz, v.sx, v.sc
    ws = v.ws
    cap = length(sz)
    out = 0
    i = 1
    j = 1
    @inbounds while i <= n || j <= m
        local kz::W, kx::W
        local acc::T
        if j > m || (i <= n && !_key_lt(ws[j], (z[i], x[i])))
            kz = z[i]
            kx = x[i]
            acc = c[i]
            i += 1
        else
            kz, kx, acc = ws[j]
            j += 1
        end
        while j <= m && _key_eq(ws[j], (kz, kx))
            acc += ws[j][3]
            j += 1
        end
        should_drop(f, kz, kx, abs(acc)) && continue
        out += 1
        out <= cap || return out, true
        sz[out] = kz
        sx[out] = kx
        sc[out] = acc
    end
    v.z, v.sz = sz, z
    v.x, v.sx = sx, x
    v.c, v.sc = sc, c
    v.n = out
    return out, false
end

"""
    _merge_spv!(v, m, f) -> (n_in, n_out)

Two-pointer merge of the sorted live buffer with `m` sorted workspace
triples into scratch, summing coefficients of equal keys (equal-key *runs*
in the appends are possible across rotations of one window), dropping
outputs the strict filter rejects, then swapping scratch and live (pointer
swap). Restores the sorted, duplicate-free live invariant and resets the
append cursor. Allocation-free in steady state; grows the live buffers
(chunked doubling) if the merged population exceeds capacity — a boundary
allocation, never a hot-loop one.
"""
function _merge_spv!(v::SparsePauliVector{N,W,T}, m::Int, f::MergeFilter) where {N,W,T}
    n_in = v.n + m
    out, ovf = _try_merge!(v, m, f)
    while ovf
        _grow_live!(v, v.n + m)
        out, ovf = _try_merge!(v, m, f)
    end
    v.an = 0
    return n_in, out
end

# ------------------------------------------------------------
# Boundary utilities: in-place compaction, coefficient histograms,
# expectation values (all allocation-free)
# ------------------------------------------------------------

# In-place filter of the live buffer (order-preserving, so the sorted
# invariant survives). Backs the clip family and the compilable _apply!.
function _compact_spv!(v::SparsePauliVector{N,W,T}, f::MergeFilter) where {N,W,T}
    out = 0
    @inbounds for i in 1:v.n
        should_drop(f, v.z[i], v.x[i], abs(v.c[i])) && continue
        out += 1
        if out != i
            v.z[out] = v.z[i]
            v.x[out] = v.x[i]
            v.c[out] = v.c[i]
        end
    end
    v.n = out
    return v
end

# |c| exponent histogram over the live buffer: bin b holds coefficients
# with 2^(b-61) <= |c| < 2^(b-60), i.e. exponent(|c|) clamped to [-60, 3].
@inline _hist_bin(absc::Float64) = clamp(exponent(absc), -60, 3) + 61

function _hist_spv!(hist::Vector{Int}, v::SparsePauliVector)
    @inbounds for i in 1:v.n
        a = abs(v.c[i])
        a == 0.0 && continue
        hist[_hist_bin(a)] += 1
    end
    return v.n
end

# Largest coefficient threshold (a bin edge) keeping at most max_terms
# terms: everything in the returned threshold's bin and below is dropped.
function _hist_threshold(hist::Vector{Int}, max_terms::Int)
    kept = 0
    @inbounds for b in _HIST_BINS:-1:1
        kept += hist[b]
        if kept > max_terms
            # drop bin b entirely: threshold just below the bin's upper edge
            return prevfloat(2.0^(b - 60))
        end
    end
    return -1.0   # everything fits
end

# ⟨ψ|·|ψ⟩ over the live buffer AND pending appends (the pre-merge state is
# live + appends with duplicates unmerged; expectation is linear, so
# summing them is exact). Computational-basis kets only: a term
# contributes c · (-1)^popcount(z & ψ) iff x == 0.
function _expectation_spv(v::SparsePauliVector{N,W,T}, kv::W) where {N,W,T}
    acc = zero(T)
    @inbounds for i in 1:v.n
        v.x[i] == zero(W) || continue
        acc += (1 - 2 * (count_ones(v.z[i] & kv) & 1)) * v.c[i]
    end
    @inbounds for i in 1:v.an
        v.ax[i] == zero(W) || continue
        acc += (1 - 2 * (count_ones(v.az[i] & kv) & 1)) * v.ac[i]
    end
    return acc
end
