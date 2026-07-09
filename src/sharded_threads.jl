# ============================================================
# Threading layer for the sharded engine: sense-reversing spin barrier,
# padded per-thread state, and the long-lived worker loop.
#
# Concurrency protocol (single-writer everywhere, no atomics in hot loops):
# - Thread t owns the shards with owner[k] == t: only t sweeps them and only
#   t merges them.
# - Every append buffer is segmented by SOURCE thread; thread t writes only
#   segment t of any shard, at cursor cur[t][j]. Within one rotation each
#   destination shard receives from exactly one source shard (k ↦ k ⊻ s_G is
#   a bijection), so segment writes are single-writer by construction.
# - cur[t] rows are written only by thread t and allocated by it (first
#   touch): no false sharing on cursor traffic.
# - Sweep bounds (sweep_hi) are snapshotted from the cursors by each shard's
#   owner during the precheck phase, when a barrier guarantees no thread is
#   rotating. Sweeps read append entries strictly below the snapshot; new
#   appends land at or above it. Live cursors moving on other threads
#   mid-rotation therefore never change what a sweep visits.
# ============================================================

"""
    SpinBarrier(n)

Sense-reversing spin barrier for `n` long-lived workers. The wait loop
calls `GC.safepoint()` — without it, a thread that triggers GC anywhere
would deadlock against spinning waiters — and `jl_cpu_pause` to be polite
to hyperthread siblings. `abort` lets a failing worker release the others
(they throw instead of spinning forever).
"""
mutable struct SpinBarrier
    @atomic count::Int
    @atomic sense::Bool
    @atomic abort::Bool
    const n::Int
end
SpinBarrier(n::Int) = SpinBarrier(0, false, false, n)

@inline function _wait!(b::SpinBarrier, local_sense::Bool)
    s = !local_sense
    if (@atomic b.count += 1) == b.n
        @atomic b.count = 0
        @atomic b.sense = s
    else
        while (@atomic b.sense) != s
            (@atomic b.abort) && error("sharded worker pool aborted")
            GC.safepoint()
            ccall(:jl_cpu_pause, Cvoid, ())
        end
    end
    (@atomic b.abort) && error("sharded worker pool aborted")
    return s
end

_abort!(b::SpinBarrier) = (@atomic b.abort = true)

"""
Per-thread mutable state: counter accumulators (flushed into
`WindowCounters` by thread 1 at window boundaries) and the precheck flag.
Padded past a cache line so neighboring states never falsely share.
"""
mutable struct ThreadState
    created::Int
    cross_appends::Int
    merge_in::Int
    merge_out::Int
    ok::Bool
    _pad::NTuple{8,UInt64}
end
ThreadState() = ThreadState(0, 0, 0, 0, true, ntuple(_ -> UInt64(0), 8))

# ------------------------------------------------------------
# Owner-local building blocks (used by both serial and threaded drivers)
# ------------------------------------------------------------

# Merge every shard owned by tid. Returns (merge_in, merge_out, maxpop).
function _merge_owned!(S::ShardedPauliSum{N,W,T}, tid::Int, f::MergeFilter) where {N,W,T}
    skip_clean = (f == NOFILTER)
    tin = 0
    tout = 0
    maxpop = 0
    @inbounds for j in 1:nshards(S)
        S.owner[j] == tid || continue
        sh = S.shards[j]
        m = _gather_append!(sh, S.cur, j, S.nthreads)
        if !(skip_clean && m == 0)
            # chunked growth is permitted here (window boundary), never mid-rotation
            sh.n + m > length(sh.z) && _grow_live!(sh, sh.n + m)
            _sort_ws!(sh.ws, 1, m)
            n_in, n_out = _merge_shard!(sh, m, f)
            tin += n_in
            tout += n_out
        end
        sh.n > maxpop && (maxpop = sh.n)
    end
    return tin, tout, maxpop
end

function _reset_cursor_row!(S::ShardedPauliSum, tid::Int)
    ck = S.cur[tid]
    @inbounds for j in 1:nshards(S)
        ck[j] = S.shards[j].seg_lo[tid]
    end
    return S
end

# Rotate every shard owned by tid. Returns (created, overflowed).
function _rotate_owned!(S::ShardedPauliSum{N,W,T}, tid::Int, s_G::Int,
                        gz::W, gx::W, n_g::Int, cosθ::Float64, sinθ::Float64,
                        f::MergeFilter) where {N,W,T}
    created = 0
    overflowed = false
    @inbounds for k in 1:nshards(S)
        S.owner[k] == tid || continue
        cr, ov = _rotate_shard!(S, k, tid, s_G, gz, gx, n_g, cosθ, sinθ, f)
        created += cr
        overflowed |= ov
    end
    return created, overflowed
end

# Snapshot sweep bounds for every owned shard and check capacities. Only
# safe while no thread is rotating (i.e. in the barrier-protected precheck
# phase, or right after a merge).
function _snapshot_and_precheck_owned!(S::ShardedPauliSum, tid::Int, s_G::Int)
    ok = true
    @inbounds for k in 1:nshards(S)
        S.owner[k] == tid || continue
        ok &= _snapshot_and_precheck!(S, k, tid, s_G)
    end
    return ok
end

# Post-merge append growth: each thread grows the append segments of the
# shards it OWNS whose (unique) source shard for this shift cannot fit.
# Only valid right after a merge (appends empty); cursor rows are reset by
# every thread afterwards.
function _grow_appends_owned!(S::ShardedPauliSum, tid::Int, s_G::Int)
    @inbounds for j in 1:nshards(S)
        S.owner[j] == tid || continue
        k = ((j - 1) ⊻ s_G) + 1
        need = S.shards[k].n
        sh = S.shards[j]
        seg = sh.seg_lo[2] - sh.seg_lo[1]
        need > seg && _grow_append!(sh, S.nthreads, need)
    end
    return S
end

# Greedy LPT reassignment of shards to threads from post-merge populations.
# Deterministic; runs on thread 1 between barriers with preallocated scratch.
function _lpt_rebalance!(S::ShardedPauliSum, pairs::Vector{Tuple{Int,Int}},
                         loads::Vector{Int})
    nsh = nshards(S)
    @inbounds for j in 1:nsh
        pairs[j] = (S.shards[j].n, j)
    end
    _sort_ws!(pairs, 1, nsh)              # ascending by population
    fill!(loads, 0)
    @inbounds for i in nsh:-1:1           # heaviest first
        pop, j = pairs[i]
        t = 1
        for u in 2:S.nthreads
            loads[u] < loads[t] && (t = u)
        end
        S.owner[j] = Int32(t)
        loads[t] += pop
    end
    return S
end

# ------------------------------------------------------------
# Threaded windowed driver
# ------------------------------------------------------------

function _worker!(S::ShardedPauliSum{N,W,T}, tid::Int, bar::SpinBarrier,
                  tls::Vector{ThreadState}, shifts::Vector{Int},
                  gz::Vector{W}, gx::Vector{W}, ng::Vector{Int},
                  cosv::Vector{Float64}, sinv::Vector{Float64},
                  window::Int, fstrict::MergeFilter, flocal::MergeFilter,
                  counters::Union{Nothing,WindowCounters},
                  rebalance_threshold::Float64,
                  pairs::Vector{Tuple{Int,Int}}, loads::Vector{Int}) where {N,W,T}
    st = tls[tid]
    nt = S.nthreads
    L = length(shifts)
    ls = false
    gcbase = Base.gc_num()
    t0 = UInt64(0)
    try
        ls = _wait!(bar, ls)                       # all workers running
        tid == 1 && (gcbase = Base.gc_num())
        for i in 1:L
            w = cld(i, window)
            s_G = shifts[i]

            st.ok = _snapshot_and_precheck_owned!(S, tid, s_G)
            ls = _wait!(bar, ls)                   # prechecks + snapshots visible
            allok = true
            for u in 1:nt
                allok &= tls[u].ok
            end
            if !allok                              # capacity-forced early merge
                tin, tout, _ = _merge_owned!(S, tid, fstrict)
                st.merge_in += tin
                st.merge_out += tout
                ls = _wait!(bar, ls)               # merges done
                _reset_cursor_row!(S, tid)
                ls = _wait!(bar, ls)               # cursors reset
                _grow_appends_owned!(S, tid, s_G)
                if tid == 1 && counters !== nothing
                    counters.early_merges[w] += 1
                end
                ls = _wait!(bar, ls)               # seg_lo stable again
                _reset_cursor_row!(S, tid)
                ls = _wait!(bar, ls)               # cursors at new seg_lo
                # re-snapshot sweep bounds (stale after the merge/growth)
                _snapshot_and_precheck_owned!(S, tid, s_G)
                ls = _wait!(bar, ls)
            end

            tid == 1 && (t0 = time_ns())
            created, ovf = _rotate_owned!(S, tid, s_G, gz[i], gx[i], ng[i],
                                          cosv[i], sinv[i], flocal)
            if ovf
                _abort!(bar)
                error("append segment overflow despite precheck (rotation $i) — this is a bug")
            end
            st.created += created
            s_G == 0 || (st.cross_appends += created)
            ls = _wait!(bar, ls)                   # rotation done, cursors quiescent
            tid == 1 && counters !== nothing &&
                (counters.t_rotate[w] += (time_ns() - t0) / 1e9)

            if i % window == 0 || i == L
                tid == 1 && (t0 = time_ns())
                tin, tout, _ = _merge_owned!(S, tid, fstrict)
                st.merge_in += tin
                st.merge_out += tout
                ls = _wait!(bar, ls)               # merges done
                _reset_cursor_row!(S, tid)
                if tid == 1
                    if counters !== nothing
                        counters.t_merge[w] += (time_ns() - t0) / 1e9
                        maxpop = 0
                        for j in 1:nshards(S)
                            S.shards[j].n > maxpop && (maxpop = S.shards[j].n)
                        end
                        counters.max_shard_pop[w] = max(counters.max_shard_pop[w], maxpop)
                        for u in 1:nt
                            counters.terms_created[w] += tls[u].created
                            counters.cross_appends[w] += tls[u].cross_appends
                            counters.merge_in[w] += tls[u].merge_in
                            counters.merge_out[w] += tls[u].merge_out
                            tls[u].created = 0
                            tls[u].cross_appends = 0
                            tls[u].merge_in = 0
                            tls[u].merge_out = 0
                        end
                        gcnow = Base.gc_num()
                        counters.allocd[w] = Base.GC_Diff(gcnow, gcbase).allocd
                        gcbase = gcnow
                    end
                    if isfinite(rebalance_threshold)
                        total = 0
                        maxload = 0
                        fill!(loads, 0)
                        for j in 1:nshards(S)
                            loads[S.owner[j]] += S.shards[j].n
                        end
                        for u in 1:nt
                            total += loads[u]
                            loads[u] > maxload && (maxload = loads[u])
                        end
                        if total > 0 && maxload * nt > rebalance_threshold * total
                            _lpt_rebalance!(S, pairs, loads)
                        end
                    end
                end
                ls = _wait!(bar, ls)               # counters read, owner stable
            end
        end
    catch
        _abort!(bar)
        rethrow()
    end
    return S
end

function _evolve_threaded!(S::ShardedPauliSum{N,W,T}, circ::CompiledCircuit{N},
                           fstrict::MergeFilter, flocal::MergeFilter,
                           counters::Union{Nothing,WindowCounters},
                           rebalance_threshold::Float64) where {N,W,T}
    nt = S.nthreads
    nt <= Threads.nthreads() ||
        error("engine was built for $nt threads but Julia has only " *
              "$(Threads.nthreads()) (start Julia with --threads=$nt or more)")
    L = length(circ)
    gz = Vector{W}(undef, L)
    gx = Vector{W}(undef, L)
    ng = Vector{Int}(undef, L)
    cosv = Vector{Float64}(undef, L)
    sinv = Vector{Float64}(undef, L)
    for i in 1:L
        g = circ.generators[i]
        gz[i], gx[i] = _pack(W, g)
        ng[i] = count_ones(gz[i] & gx[i])
        cosv[i] = cos(circ.angles[i])
        sinv[i] = sin(circ.angles[i])
    end
    bar = SpinBarrier(nt)
    tls = [ThreadState() for _ in 1:nt]
    pairs = Vector{Tuple{Int,Int}}(undef, nshards(S))
    loads = zeros(Int, nt)
    tasks = Vector{Task}(undef, nt - 1)
    for tid in 2:nt
        tasks[tid-1] = Threads.@spawn _worker!(S, $tid, bar, tls, circ.shifts,
                                               gz, gx, ng, cosv, sinv, circ.window,
                                               fstrict, flocal, counters,
                                               rebalance_threshold, pairs, loads)
    end
    err = nothing
    try
        _worker!(S, 1, bar, tls, circ.shifts, gz, gx, ng, cosv, sinv, circ.window,
                 fstrict, flocal, counters, rebalance_threshold, pairs, loads)
    catch e
        err = e                    # keep the primary failure, not the abort echoes
    end
    for t in tasks
        try
            wait(t)
        catch e
            err === nothing && (err = e)
        end
    end
    err === nothing || throw(err)
    return S
end

"""
    pin_engine!(S::ShardedPauliSum)

Pin Julia's threads to cores (compact within sockets) for stable NUMA
placement. Requires ThreadPinning.jl to be loaded (`using ThreadPinning`);
without it this is a no-op with an informational message.
"""
function pin_engine!(S::ShardedPauliSum)
    ext = Base.get_extension(@__MODULE__, :PauliOperatorsThreadPinningExt)
    if ext === nothing
        @info "ThreadPinning.jl not loaded — thread pinning skipped " *
              "(add `using ThreadPinning` before calling pin_engine!)"
    else
        ext._pin!(S)
    end
    return S
end
