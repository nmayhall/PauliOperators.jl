"""
Multinode (across-node) Pauli-operator evolution.

A `PauliSum` is a `Dict{PauliBasis,coeff}`. For Sparse Pauli Dynamics on large
lattices (e.g. 10x10x10 = 1000 qubits) the sum can exceed one node's memory, so
we shard it by hash of the `PauliBasis` across distributed workers: each worker
owns the terms that hash to it and stores an ordinary local `PauliSum`.

A Heisenberg-picture rotation `evolve!(O, G, θ)` splits every term that
anticommutes with the generator `G`:

    O(θ) = cos(θ)·O - i·sin(θ)·G·O

The cos part scales terms in place (no term moves). The sin part creates new terms
`G·p`, each with a definite `PauliBasis` that hashes to a definite owner — so a
distributed rotation is: (1) each worker cos-scales its terms and buckets the new
sin terms by destination owner; (2) buckets move worker→worker (peer to peer, not
via the master) and are merged; (3) staging is cleared. Truncation
(`coeff_clip!`) is then purely local with the global threshold.

This mirrors the hash-sharded `DistributedTPSCIstate` design in TPSChem.jl.
"""

# Worker-local storage: shard id -> local PauliSum, and outgoing bucket staging.
const _DPS_STORE = Dict{Symbol,Any}()
const _DPS_STAGE = Dict{Symbol,Any}()

"""
    DistributedPauliSum{N,T}

Metadata handle for a `PauliSum{N,T}` whose terms are sharded across `workers`.
The master holds only the id and worker list; the coefficients live on the workers.
"""
mutable struct DistributedPauliSum{N,T}
    id::Symbol
    workers::Vector{Int}
end

# Which worker owns a given Pauli basis (stable hash partition).
@inline function _pauli_owner(pb, workers)
    return workers[Int(mod(hash(pb), UInt(length(workers)))) + 1]
end

_dps_worker_ids(workers) = (pids = collect(workers); isempty(pids) &&
    error("No distributed workers available; start Julia with workers (addprocs)."); pids)

"""
    ensure_pauli_workers!(; workers=workers())

Activate the project and load `PauliOperators` on each worker. Returns the pids.
"""
function ensure_pauli_workers!(; workers=Distributed.workers())
    pids = _dps_worker_ids(workers)
    project_root = dirname(@__DIR__)
    @sync for pid in pids
        pid == Distributed.myid() && continue
        @async Distributed.remotecall_fetch(
            Core.eval, pid, Main,
            :(begin
                  import Pkg
                  Pkg.activate($project_root; io=devnull)
                  using PauliOperators
              end))
    end
    return pids
end

# ---- worker-local primitives ----

function _dps_store!(id::Symbol, local_ps)
    _DPS_STORE[id] = local_ps
    return length(local_ps)
end
_dps_get(id::Symbol) = (haskey(_DPS_STORE, id) || error("No sharded PauliSum $id on worker $(Distributed.myid())"); _DPS_STORE[id])
_dps_delete!(id::Symbol) = (delete!(_DPS_STORE, id); delete!(_DPS_STAGE, id); true)
_dps_length(id::Symbol) = length(_dps_get(id))
_dps_local_copy(id::Symbol) = copy(_dps_get(id))

# ---- distribute / gather ----

"""
    distribute(O::PauliSum{N,T}; workers=workers(), id=nothing) -> DistributedPauliSum

Shard a local `PauliSum` across `workers` by hash of the `PauliBasis`.
"""
function distribute(O::PauliSum{N,T}; workers=Distributed.workers(), id=nothing) where {N,T}
    pids = ensure_pauli_workers!(workers=workers)
    sid = id === nothing ? gensym(:dps) : Symbol(id)
    buckets = Dict(pid => PauliSum(N, T) for pid in pids)
    for (pb, c) in O
        buckets[_pauli_owner(pb, pids)][pb] = c
    end
    @sync for pid in pids
        @async begin
            if pid == Distributed.myid()
                _dps_store!(sid, buckets[pid])
            else
                Distributed.remotecall_fetch(_dps_store!, pid, sid, buckets[pid])
            end
        end
    end
    return DistributedPauliSum{N,T}(sid, pids)
end

"""
    DistributedPauliSum(N, T; workers=workers()) -> empty sharded sum
"""
function DistributedPauliSum(N::Integer, ::Type{T}=ComplexF64; workers=Distributed.workers(), id=nothing) where {T}
    return distribute(PauliSum(N, T); workers=workers, id=id)
end

Base.length(dO::DistributedPauliSum) = sum(Distributed.remotecall_fetch(_dps_length, pid, dO.id) for pid in dO.workers)

"""
    collect_paulisum(dO) -> PauliSum

Gather a sharded sum back to one local `PauliSum` (debug/analysis; do not use on
sums larger than node memory).
"""
function collect_paulisum(dO::DistributedPauliSum{N,T}) where {N,T}
    out = PauliSum(N, T)
    for pid in dO.workers
        merge!(out, Distributed.remotecall_fetch(_dps_local_copy, pid, dO.id))
    end
    return out
end

function destroy!(dO::DistributedPauliSum)
    @sync for pid in dO.workers
        @async Distributed.remotecall_fetch(_dps_delete!, pid, dO.id)
    end
    return dO
end

# ---- distributed evolution ----

# Phase 1: cos-scale anticommuting terms; bucket sin terms by owner.
# On-node this is threaded: the terms are chunked across Julia threads, each chunk
# fills its OWN per-destination staging buckets (no shared-Dict races) and records
# which terms to cos-scale; the cheap cos-scaling and the bucket merge run after.
function _dps_rotate_local!(id::Symbol, G, θ::Real, workers, threaded::Bool=true)
    return _dps_rotate_local_typed!(id, _dps_get(id), G, θ, workers, threaded)
end

# contiguous chunk ranges of 1:n split into k parts (some may be empty if k>n)
function _dps_chunk_ranges(n::Int, k::Int)
    sz = cld(n, k)
    return [((c-1)*sz + 1):min(c*sz, n) for c in 1:k]
end

# read-only over O; fill this chunk's sin buckets and record its anticommuting keys.
function _rotate_chunk!(O::PauliSum{N,T}, ks, range, G, _sin, workers, stage, coskeys) where {N,T}
    for idx in range
        p = ks[idx]
        commute(p, G) && continue
        c = O[p]
        tmp = c * _sin * G * p          # sin-branch Pauli  i·sin·c·(G p)
        qpb = PauliBasis(tmp)
        dest = stage[_pauli_owner(qpb, workers)]
        dest[qpb] = get(dest, qpb, zero(T)) + coeff(tmp)
        push!(coskeys, p)
    end
    return nothing
end

function _dps_rotate_local_typed!(id::Symbol, O::PauliSum{N,T}, G, θ::Real, workers,
                                  threaded::Bool) where {N,T}
    _cos = cos(θ)
    _sin = 1im*sin(θ)
    ks = collect(keys(O))
    nt = (threaded && Threads.nthreads() > 1 && length(ks) > 1) ? Threads.nthreads() : 1

    tstage = [Dict(pid => PauliSum(N, T) for pid in workers) for _ in 1:nt]
    tcos   = [Vector{eltype(ks)}() for _ in 1:nt]

    if nt == 1
        _rotate_chunk!(O, ks, eachindex(ks), G, _sin, workers, tstage[1], tcos[1])
    else
        ranges = _dps_chunk_ranges(length(ks), nt)
        @sync for c in 1:nt
            Threads.@spawn _rotate_chunk!(O, ks, ranges[c], G, _sin, workers, tstage[c], tcos[c])
        end
    end

    # cos-scale anticommuting terms in place (serial: no concurrent Dict writes)
    for c in 1:nt, p in tcos[c]
        O[p] = O[p] * _cos
    end

    # merge the per-thread staging into one per-destination stage
    if nt == 1
        _DPS_STAGE[id] = tstage[1]
    else
        stage = Dict(pid => PauliSum(N, T) for pid in workers)
        for c in 1:nt, pid in workers
            _dps_merge_bucket!(stage[pid], tstage[c][pid])
        end
        _DPS_STAGE[id] = stage
    end
    return nothing
end

# Peer take: worker holding `id`'s staging returns the bucket destined for `dest`.
function _dps_take_bucket(id::Symbol, dest::Int)
    stage = _DPS_STAGE[id]
    return stage[dest]
end

# Phase 2: each worker pulls the buckets addressed to it from every worker, merges.
function _dps_merge_incoming!(id::Symbol, workers)
    me = Distributed.myid()
    O = _dps_get(id)
    for qid in workers
        bucket = qid == me ? _DPS_STAGE[id][me] :
                 Distributed.remotecall_fetch(_dps_take_bucket, qid, id, me)
        _dps_merge_bucket!(O, bucket)
    end
    return nothing
end
function _dps_merge_bucket!(O::PauliSum{N,T}, bucket) where {N,T}
    for (pb, c) in bucket
        O[pb] = get(O, pb, zero(T)) + c
    end
    return O
end

_dps_clear_stage!(id::Symbol) = (delete!(_DPS_STAGE, id); nothing)

"""
    evolve!(dO::DistributedPauliSum, G::PauliBasis, θ::Real)

One distributed Heisenberg-picture rotation, in place. `G` is broadcast to all
workers; new sin-branch terms are routed to their owners and merged. `threaded`
enables Julia-thread parallelism of each worker's local rotation (start the
workers with `--threads=N`).
"""
function evolve!(dO::DistributedPauliSum{N,T}, G::PauliBasis{N}, θ::Real;
                 threaded::Bool=true) where {N,T}
    ws = dO.workers
    @sync for pid in ws
        @async Distributed.remotecall_fetch(_dps_rotate_local!, pid, dO.id, G, θ, ws, threaded)
    end
    @sync for pid in ws
        @async Distributed.remotecall_fetch(_dps_merge_incoming!, pid, dO.id, ws)
    end
    @sync for pid in ws
        @async Distributed.remotecall_fetch(_dps_clear_stage!, pid, dO.id)
    end
    return dO
end

"""
    coeff_clip!(dO::DistributedPauliSum, thresh::Real)

Drop terms with `|coeff| <= thresh` on every worker (global threshold, local work).
"""
function coeff_clip!(dO::DistributedPauliSum, thresh::Real)
    @sync for pid in dO.workers
        @async Distributed.remotecall_fetch(_dps_coeff_clip!, pid, dO.id, thresh)
    end
    return dO
end
_dps_coeff_clip!(id::Symbol, thresh::Real) = (coeff_clip!(_dps_get(id), thresh); nothing)

"""
    evolve(dO, generators, angles; truncation_thresh=0.0) -> dO

Apply a sequence of rotations (e.g. from `trotterize`) to a sharded sum, clipping
after each rotation when `truncation_thresh > 0`. Mutates and returns `dO`.
"""
function evolve!(dO::DistributedPauliSum{N,T}, generators::Vector{<:PauliBasis{N}},
                 angles::Vector{<:Real}; truncation_thresh::Real=0.0,
                 threaded::Bool=true) where {N,T}
    length(generators) == length(angles) || throw(DimensionMismatch("generators and angles must match"))
    for (G, θ) in zip(generators, angles)
        evolve!(dO, G, θ; threaded=threaded)
        truncation_thresh > 0 && coeff_clip!(dO, truncation_thresh)
    end
    return dO
end

# ---- reductions ----

"""
    opnorm2(dO) -> Float64

Frobenius-like 2-norm of the coefficient vector, sqrt(Σ|c|²), via a global reduction.
"""
function opnorm2(dO::DistributedPauliSum)
    s = sum(Distributed.remotecall_fetch(_dps_local_normsq, pid, dO.id) for pid in dO.workers)
    return sqrt(s)
end
_dps_local_normsq(id::Symbol) = sum(abs2, values(_dps_get(id)); init=0.0)

"""
    sharded_summary(dO) -> Vector of (worker, n_terms)
"""
sharded_summary(dO::DistributedPauliSum) =
    [(pid, Distributed.remotecall_fetch(_dps_length, pid, dO.id)) for pid in dO.workers]
