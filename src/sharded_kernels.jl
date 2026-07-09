# ============================================================
# Sharded engine kernels: rotation sweep, append sort, sorted merge.
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
# Truncation filter, compiled once per evolve! from a TruncationStrategy —
# an isbits predicate evaluated per term with no dynamic dispatch.
# ------------------------------------------------------------

"""
    MergeFilter

Compiled truncation predicate for the sharded kernels. Sentinels disable
individual checks: `typemax(Int)` for the weight cutoffs, negative
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
# analogue of `majorana_weight(::PauliBasis)` (see clip.jl for the derivation).
@inline function _majorana_weight_bits(z::W, x::W) where {W<:Unsigned}
    zonly = z & ~x
    S = x
    S ⊻= S >> 1
    S ⊻= S >> 2
    S ⊻= S >> 4
    S ⊻= S >> 8
    S ⊻= S >> 16
    S ⊻= S >> 32
    S ⊻= S >> 64          # no-op at W == UInt64
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
    error("$(typeof(s)) is not supported by the sharded engine")

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

# ------------------------------------------------------------
# Rotation kernel
# ------------------------------------------------------------

"""
    _rotate_range!(z, x, c, lo, hi, gz, gx, n_g, cosθ, sinθ,
                   dz, dx, dc, cur, seg_end, f) -> (cur, created, overflowed)

Sweep terms `lo:hi` of one buffer under the rotation `exp(iθG)`: commuting
terms untouched; anticommuting terms cos-scaled in place, with the sin
branch (bits `G ⊻ P`, sign `i·i^k = ±1` computed purely from bits — the
fused-phase identity from commutator.jl) appended at `cur` in the
destination arrays unless the local filter drops it. `n_g` is
`count_ones(gz & gx)`, precomputed once per rotation.

Zero-allocation hot path. `overflowed` only fires if the driver's capacity
precheck was skipped or wrong; the driver treats it as an error.
"""
@inline function _rotate_range!(z::Vector{W}, x::Vector{W}, c::Vector{T},
                                lo::Int, hi::Int,
                                gz::W, gx::W, n_g::Int, cosθ::Float64, sinθ::Float64,
                                dz::Vector{W}, dx::Vector{W}, dc::Vector{T},
                                cur::Int, seg_end::Int,
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
        if cur > seg_end
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

"""
    _rotate_shard!(S, k, t, s_G, gz, gx, n_g, cosθ, sinθ, f) -> (created, overflowed)

Rotate shard `k` as thread `t`: sweep its live buffer and every append
segment up to the frozen `mark`, appending sin branches into thread `t`'s
segment of the partner shard `k ⊻ s_G` (destination resolved once — all of
a shard's sin branches share one partner; that is the rank-map property).
Appends land at `cur ≥ mark`, so swept and written ranges never overlap.
"""
function _rotate_shard!(S::ShardedPauliSum{N,W,T}, k::Int, t::Int, s_G::Int,
                        gz::W, gx::W, n_g::Int, cosθ::Float64, sinθ::Float64,
                        f::MergeFilter) where {N,W,T}
    src = S.shards[k]
    j = ((k - 1) ⊻ s_G) + 1
    dst = S.shards[j]
    curt = S.cur[t]
    cur = curt[j]
    seg_end = dst.seg_lo[t+1] - 1
    created = 0
    overflowed = false

    cur, cr, ov = _rotate_range!(src.z, src.x, src.c, 1, src.n,
                                 gz, gx, n_g, cosθ, sinθ,
                                 dst.az, dst.ax, dst.ac, cur, seg_end, f)
    created += cr
    overflowed |= ov
    @inbounds for seg in 1:S.nthreads
        lo = src.seg_lo[seg]
        hi = S.mark[seg][k] - 1
        cur, cr, ov = _rotate_range!(src.az, src.ax, src.ac, lo, hi,
                                     gz, gx, n_g, cosθ, sinθ,
                                     dst.az, dst.ax, dst.ac, cur, seg_end, f)
        created += cr
        overflowed |= ov
    end
    curt[j] = cur
    return created, overflowed
end

# Worst-case capacity precheck for rotating shard k as thread t under shift
# s_G: every swept term could anticommute and append to the partner segment.
function _precheck_shard(S::ShardedPauliSum, k::Int, t::Int, s_G::Int)
    src = S.shards[k]
    swept = src.n
    @inbounds for seg in 1:S.nthreads
        swept += S.mark[seg][k] - src.seg_lo[seg]
    end
    swept == 0 && return true
    j = ((k - 1) ⊻ s_G) + 1
    dst = S.shards[j]
    free = dst.seg_lo[t+1] - S.cur[t][j]
    return swept <= free
end

# ------------------------------------------------------------
# Sort + merge (window boundary)
# ------------------------------------------------------------

"""
Gather shard `j`'s pending append segments (each up to its cursor) into the
workspace as (z, x, c) triples. Returns the count. Allocation-free.
"""
function _gather_append!(sh::Shard{W,T}, cur::Vector{Vector{Int}}, j::Int,
                         nsegs::Int) where {W,T}
    m = 0
    @inbounds for t in 1:nsegs
        for i in sh.seg_lo[t]:(cur[t][j] - 1)
            m += 1
            sh.ws[m] = (sh.az[i], sh.ax[i], sh.ac[i])
        end
    end
    return m
end

"""
In-place quicksort (median-of-3, insertion sort below 24, recurse-smaller /
iterate-larger) of workspace triples by (z, x) key. Hand-rolled because
Base's default QuickSort allocates scratch; this is the swap point for a
future radix sort. Allocation-free.
"""
function _sort_ws!(ws::Vector{Tuple{W,W,T}}, lo::Int, hi::Int) where {W,T}
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

"""
    _merge_shard!(sh, m, f) -> (n_in, n_out)

Two-pointer merge of the sorted live buffer with `m` sorted workspace
triples into scratch, summing coefficients of equal keys (live first, then
appends in sorted order — equal-key *runs* in the appends are possible
across rotations of one window), dropping outputs the strict filter
rejects, then swapping scratch and live (pointer swap). Restores the
sorted, duplicate-free live invariant. Allocation-free.
"""
function _merge_shard!(sh::Shard{W,T}, m::Int, f::MergeFilter) where {W,T}
    n = sh.n
    z, x, c = sh.z, sh.x, sh.c
    sz, sx, sc = sh.sz, sh.sx, sh.sc
    ws = sh.ws
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
        out <= cap || error("shard live capacity exceeded during merge; " *
                            "raise capacity_factor or tighten truncation")
        sz[out] = kz
        sx[out] = kx
        sc[out] = acc
    end
    sh.z, sh.sz = sz, z
    sh.x, sh.sx = sx, x
    sh.c, sh.sc = sc, c
    sh.n = out
    return n + m, out
end
