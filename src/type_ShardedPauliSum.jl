# ============================================================
# ShardedPauliSum — flat, preallocated, structure-of-arrays storage for
# shared-memory (multithreaded) Pauli propagation.
#
# Design contract (see the shared-memory design doc): the steady-state hot
# path (rotation sweeps and window-boundary merges) allocates zero bytes.
# All storage is preallocated here, at construction or window boundaries,
# and reused. No Dict, no per-term heap objects, isbits data only.
# ============================================================

"""
    _word_type(N)

Packed bit-word type for an `N`-qubit register: `UInt64` when the bits fit
(halves bit bandwidth), `UInt128` otherwise.
"""
_word_type(N::Integer) = N <= 64 ? UInt64 : UInt128

@inline _pack(::Type{W}, p::PauliBasis) where {W<:Unsigned} = (p.z % W, p.x % W)
@inline _unpack(::Type{PauliBasis{N}}, z::W, x::W) where {N,W<:Unsigned} =
    PauliBasis{N}(Int128(z), Int128(x))

"""
Shard index (0-based) of packed bits under packed rank-map rows — the
native-width analogue of `bin_index(A, p)`. Used for construction and
invariant checks; hot-path routing XORs precomputed shifts instead.
"""
@inline function _shard_index(row_z::Vector{W}, row_x::Vector{W}, z::W, x::W) where {W<:Unsigned}
    b = 0
    @inbounds for i in eachindex(row_z)
        b |= ((count_ones(z & row_z[i]) ⊻ count_ones(x & row_x[i])) & 1) << (i - 1)
    end
    return b
end

"""
    Shard{W,T}

One shard's storage. Three parallel-array buffer sets:

- live `z/x/c[1:n]`: current terms, **sorted by strictly increasing `(z,x)`
  key** after every merge, duplicate-free.
- append `az/ax/ac`: sin-branch terms created during a window, unsorted,
  segmented by *source thread*: segment `t` is `seg_lo[t]:seg_lo[t+1]-1`,
  single-writer (only thread `t` appends there), cursor kept outside the
  shard in the engine's per-thread cursor rows.
- scratch `sz/sx/sc`: merge output; swapped with live by field assignment
  (pointer swap, never a copy).

`ws` is the preallocated merge workspace (append terms gathered as packed
`(z, x, c)` triples for sorting); `hist` is a fixed-size |c| exponent
histogram used for adaptive thresholds.
"""
mutable struct Shard{W<:Unsigned, T<:Number}
    z::Vector{W}
    x::Vector{W}
    c::Vector{T}
    n::Int
    az::Vector{W}
    ax::Vector{W}
    ac::Vector{T}
    seg_lo::Vector{Int}          # length nsegments+1 (fence-post)
    sweep_hi::Vector{Int}        # per-segment sweep bound, snapshotted by the
                                 # OWNER from stable cursors in the precheck
                                 # phase (mid-rotation appends land above it)
    sz::Vector{W}
    sx::Vector{W}
    sc::Vector{T}
    ws::Vector{Tuple{W,W,T}}
    hist::Vector{Int}
end

const _HIST_BINS = 64

function _alloc_shard(::Type{W}, ::Type{T}, live_cap::Int, seg_size::Int, nsegs::Int) where {W,T}
    append_cap = seg_size * nsegs
    seg_lo = [1 + (t - 1) * seg_size for t in 1:nsegs+1]
    return Shard{W,T}(
        zeros(W, live_cap), zeros(W, live_cap), zeros(T, live_cap), 0,
        zeros(W, append_cap), zeros(W, append_cap), zeros(T, append_cap), seg_lo,
        copy(seg_lo[1:nsegs]),
        zeros(W, live_cap), zeros(W, live_cap), zeros(T, live_cap),
        # ws must hold all appends at a merge AND the live terms during the
        # construction-time sort
        Vector{Tuple{W,W,T}}(undef, max(append_cap, live_cap)), zeros(Int, _HIST_BINS))
end

"""
    ShardedConfig

Engine configuration: *initial* per-shard capacities and the debug flag.
Buffers grow by chunked doubling at window boundaries only (never inside a
rotation loop); once the population plateaus under truncation, no further
growth occurs and the steady-state hot path allocates zero bytes.
"""
struct ShardedConfig
    live_cap::Int          # initial live-buffer capacity per shard
    seg_size::Int          # initial append segment size (per source thread, per shard)
    debug::Bool
end

# Window-boundary growth (the only allocation points after construction).
function _grow_live!(sh::Shard{W,T}, need::Int) where {W,T}
    newcap = max(2 * length(sh.z), need)
    resize!(sh.z, newcap)
    resize!(sh.x, newcap)
    resize!(sh.c, newcap)
    resize!(sh.sz, newcap)
    resize!(sh.sx, newcap)
    resize!(sh.sc, newcap)
    return sh
end

# Only valid when the shard's append buffer is empty (right after a merge).
function _grow_append!(sh::Shard{W,T}, nsegs::Int, min_seg::Int) where {W,T}
    seg_size = sh.seg_lo[2] - sh.seg_lo[1]
    new_seg = max(2 * seg_size, min_seg)
    resize!(sh.az, new_seg * nsegs)
    resize!(sh.ax, new_seg * nsegs)
    resize!(sh.ac, new_seg * nsegs)
    resize!(sh.ws, new_seg * nsegs)
    for t in 1:nsegs+1
        sh.seg_lo[t] = 1 + (t - 1) * new_seg
    end
    return sh
end

"""
    ShardedPauliSum{N,W,T}

A `PauliSum` partitioned into `2^r` shards by a GF(2) `RankMap` (shard id =
`bin_index(A, p) + 1`), stored as flat preallocated structure-of-arrays
buffers (see [`Shard`](@ref)). `W` is the packed bit-word type
(`UInt64` for `N ≤ 64`), `T` the coefficient type (`Float64` suffices for
Hermitian dynamics and halves coefficient bandwidth).

Threading state: `owner[j]` maps shard `j` to its owning thread (the ONLY
load-balancing mechanism — terms never move between shards except by
rotation), and `cur[t][j]` is thread `t`'s append cursor for its segment of
shard `j`. Each cursor row is written only by thread `t` and allocated by
it (first-touch), so cursor traffic never falsely shares cache lines.
Sweep bounds are snapshotted from the cursors into each shard's `sweep_hi`
by its owner during the barrier-protected precheck phase, so mid-rotation
cursor movement on other threads can never change what a sweep visits.
"""
mutable struct ShardedPauliSum{N, W<:Unsigned, T<:Number}
    A::RankMap{N}
    row_z::Vector{W}
    row_x::Vector{W}
    shards::Vector{Shard{W,T}}
    owner::Vector{Int32}
    cur::Vector{Vector{Int}}     # cur[t][shard]: next free slot in segment t
    nthreads::Int
    version::Int
    cfg::ShardedConfig
end

nshards(S::ShardedPauliSum) = length(S.shards)

_engine_coeff(::Type{T}, c::Number, tol::Float64) where {T<:Complex} = convert(T, c)
function _engine_coeff(::Type{T}, c::Number, tol::Float64) where {T<:Real}
    abs(imag(c)) <= tol * max(1.0, abs(c)) ||
        error("coefficient $c has an imaginary part; use a complex coefficient " *
              "type (T=ComplexF64) or provide Hermitian (real-coefficient) input")
    return convert(T, real(c))
end

"""
    ShardedPauliSum(O::PauliSum{N}, A::RankMap{N};
                    T=Float64, nthreads=1, capacity_factor=4.0,
                    append_factor=1.0, debug=false, imag_tol=1e-10)

Scatter `O` into `2^nbits(A)` shards of flat preallocated buffers.

`capacity_factor` sizes each shard's initial live buffer relative to the
current heaviest shard (headroom for population growth between
truncations); `append_factor` sizes the initial per-window append buffer
relative to the live buffer. Exhaustion during evolution triggers an early
merge and, if the population genuinely needs more room, chunked buffer
doubling at the window boundary — never a hot-loop reallocation. Size
generously (or truncate) so steady state never grows: the allocation tests
require post-plateau windows to allocate zero bytes.

Real `T` requires (numerically) real coefficients: terms with
`|imag(c)| > imag_tol·max(1,|c|)` are an error.
"""
function ShardedPauliSum(O::PauliSum{N,T0}, A::RankMap{N};
                         T::Type{<:Number}=Float64,
                         nthreads::Int=1,
                         capacity_factor::Real=4.0,
                         append_factor::Real=1.0,
                         min_capacity::Int=16,
                         debug::Bool=false,
                         imag_tol::Real=1e-10) where {N,T0}
    nthreads >= 1 || throw(ArgumentError("nthreads must be >= 1"))
    capacity_factor >= 1 || throw(ArgumentError("capacity_factor must be >= 1"))
    W = _word_type(N)
    nsh = nbins(A)

    counts = zeros(Int, nsh)
    for p in keys(O)
        counts[bin_index(A, p) + 1] += 1
    end
    heaviest = max(maximum(counts; init=0), cld(length(O), nsh), 1)
    # min_capacity floors the per-shard buffers: essential when evolving a
    # small observable whose population will grow far beyond length(O)
    live_cap = max(min_capacity, 16, ceil(Int, capacity_factor * heaviest))
    seg_size = max(16, cld(ceil(Int, append_factor * live_cap), nthreads))

    row_z = W[r.z % W for r in A.rows]
    row_x = W[r.x % W for r in A.rows]
    owner = Int32[fld((j - 1) * nthreads, nsh) + 1 for j in 1:nsh]
    shards = Vector{Shard{W,T}}(undef, nsh)
    cur = Vector{Vector{Int}}(undef, nthreads)
    if nthreads > 1 && Threads.nthreads() > 1
        # first-touch initialization: each shard's buffers (and each cursor
        # row) are allocated AND first written inside the owning thread's
        # task, so their pages land on the owner's socket
        @sync for t in 1:nthreads
            Threads.@spawn begin
                for j in 1:nsh
                    owner[j] == $t || continue
                    shards[j] = _alloc_shard(W, T, live_cap, seg_size, nthreads)
                end
                cur[$t] = zeros(Int, nsh)
            end
        end
    else
        for j in 1:nsh
            shards[j] = _alloc_shard(W, T, live_cap, seg_size, nthreads)
        end
        for t in 1:nthreads
            cur[t] = zeros(Int, nsh)
        end
    end
    for t in 1:nthreads, j in 1:nsh
        cur[t][j] = shards[j].seg_lo[t]
    end

    S = ShardedPauliSum{N,W,T}(A, row_z, row_x, shards, owner, cur,
                               nthreads, 0, ShardedConfig(live_cap, seg_size, debug))

    tol = Float64(imag_tol)
    for (p, c) in O
        sh = shards[bin_index(A, p) + 1]
        sh.n += 1
        z, x = _pack(W, p)
        sh.z[sh.n] = z
        sh.x[sh.n] = x
        sh.c[sh.n] = _engine_coeff(T, c, tol)
    end
    for sh in shards
        _sort_live!(sh)
    end
    return S
end

function _sort_live!(sh::Shard{W,T}) where {W,T}
    for i in 1:sh.n
        sh.ws[i] = (sh.z[i], sh.x[i], sh.c[i])
    end
    _sort_ws!(sh.ws, 1, sh.n)
    for i in 1:sh.n
        sh.z[i], sh.x[i], sh.c[i] = sh.ws[i]
    end
    return sh
end

"""
    PauliSum(S::ShardedPauliSum)

Gather back into a `Dict`-based `PauliSum` (testing / small runs). Only
valid on merged state (no pending appends), which is what every driver
leaves behind.
"""
function PauliSum(S::ShardedPauliSum{N,W,T}) where {N,W,T}
    out = PauliSum(N, T)
    sizehint!(out, length(S))
    for (j, sh) in enumerate(S.shards)
        for t in 1:S.nthreads
            S.cur[t][j] == sh.seg_lo[t] ||
                error("shard $j has pending appends; gather only merged state")
        end
        for i in 1:sh.n
            out[_unpack(PauliBasis{N}, sh.z[i], sh.x[i])] = sh.c[i]
        end
    end
    return out
end

Base.length(S::ShardedPauliSum) = sum(sh.n for sh in S.shards; init=0)
Base.isempty(S::ShardedPauliSum) = length(S) == 0

"""
    check_sharding(S::ShardedPauliSum)

Verify the structural invariants: every live term sits in the shard given
by its bits and `A` (invariant 1), and every shard's live keys are strictly
increasing — sorted, duplicate-free (invariant 2). Errors on violation.
"""
function check_sharding(S::ShardedPauliSum{N,W,T}) where {N,W,T}
    for (j, sh) in enumerate(S.shards)
        sh.n <= length(sh.z) || error("shard $j: count $(sh.n) exceeds capacity")
        for i in 1:sh.n
            b = _shard_index(S.row_z, S.row_x, sh.z[i], sh.x[i])
            b == j - 1 || error("shard $j: term $i belongs in shard $(b+1)")
            if i > 1
                _key_lt((sh.z[i-1], sh.x[i-1]), (sh.z[i], sh.x[i])) ||
                    error("shard $j: keys not strictly increasing at $i")
            end
        end
    end
    return true
end

function LinearAlgebra.norm(S::ShardedPauliSum{N,W,T}, p::Real=2) where {N,W,T}
    if p == 2
        s = 0.0
        for sh in S.shards, i in 1:sh.n
            s += abs2(sh.c[i])
        end
        return sqrt(s)
    elseif p == Inf
        s = 0.0
        for sh in S.shards, i in 1:sh.n
            a = abs(sh.c[i])
            a > s && (s = a)
        end
        return s
    else
        s = 0.0
        for sh in S.shards, i in 1:sh.n
            s += abs(sh.c[i])^p
        end
        return p == 1 ? s : s^(1/p)
    end
end

function LinearAlgebra.tr(S::ShardedPauliSum{N,W,T}) where {N,W,T}
    sh = S.shards[1]   # identity always lands in shard 0
    (sh.n >= 1 && sh.z[1] == zero(W) && sh.x[1] == zero(W)) || return zero(T) * 2^N
    return sh.c[1] * 2^N
end

function Base.show(io::IO, S::ShardedPauliSum{N,W,T}) where {N,W,T}
    occ = count(sh -> sh.n > 0, S.shards)
    print(io, "ShardedPauliSum{$N,$W,$T}: $(length(S)) terms in $occ/$(nshards(S)) shards, ",
          "$(S.nthreads) threads, live_cap=$(S.cfg.live_cap)/shard")
end
