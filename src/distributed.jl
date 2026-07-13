"""
Multinode (across-node) Pauli-operator evolution.

For Sparse Pauli Dynamics on large lattices (e.g. 10x10x10 = 1000 qubits) the
operator can exceed one node's memory, so we shard it by hash of the
`PauliBasis` across distributed workers: each worker owns the terms that hash to
it and stores either a local `PauliSum` (`Dict`) or a local `SparsePauliVector`.

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

# Worker-local storage: shard id -> local Pauli container, and outgoing bucket staging.
const _DPS_STORE = Dict{Symbol,Any}()
const _DPS_STAGE = Dict{Symbol,Any}()
const _DPS_PENDING = Dict{Symbol,Any}()

"""
    DistributedPauliSum{N,T}

Metadata handle for an `N`-qubit Pauli sum whose terms are sharded across
`workers`. The master holds only the id, worker list, and local storage kind; the
coefficients live on the workers.
"""
mutable struct DistributedPauliSum{N,T}
    id::Symbol
    workers::Vector{Int}
    storage::Symbol
end

DistributedPauliSum{N,T}(id::Symbol, workers::Vector{Int}) where {N,T} =
    DistributedPauliSum{N,T}(id, workers, :dict)

# Which worker owns a given Pauli basis (stable hash partition).
@inline function _pauli_owner(pb, workers)
    return workers[Int(mod(hash(pb), UInt(length(workers)))) + 1]
end

_dps_worker_ids(workers) = (pids = collect(workers); isempty(pids) &&
    error("No distributed workers available; start Julia with workers (addprocs)."); pids)

const PAULI_STORAGE_ENV = "PAULI_STORAGE"

function _dps_storage_alias(storage)
    s = Symbol(lowercase(strip(string(storage))))
    s in (:dict, :paulisum, :dictionary) && return :dict
    s in (:spv, :sparse, :sparsepaulivector, :sparse_pauli_vector) && return :spv
    throw(ArgumentError("storage must be :dict or :spv, got $storage"))
end

"""
    pauli_storage([storage]; default=:dict, env=PAULI_STORAGE_ENV) -> Symbol

Resolve the Pauli storage representation for scripts and distributed jobs.
Explicit `storage` choices take precedence. If `storage` is omitted, the
environment variable `PAULI_STORAGE` is used when present, otherwise `default`
is used. Accepted aliases are `:dict`/`:paulisum`/`:dictionary` and
`:spv`/`:sparse`/`:sparsepaulivector`.

Examples:

    PAULI_STORAGE=spv julia --project job.jl
    dO = distribute(O; workers=workers())       # uses SPV shards

    dO = distribute(O; workers=workers(), storage=:dict)  # explicit override
"""
function pauli_storage(storage=nothing; default=:dict,
                       env::Union{AbstractString,Nothing}=PAULI_STORAGE_ENV)
    choice = storage
    if choice === nothing
        raw = env === nothing ? "" : strip(get(ENV, env, ""))
        choice = isempty(raw) ? default : raw
    end
    return _dps_storage_alias(choice)
end

_dps_storage(storage) = _dps_storage_alias(storage)
_dps_default_storage(::PauliSum) = :dict
_dps_default_storage(::SparsePauliVector) = :spv

function _dps_materialize_bucket(bucket::PauliSum{N,T}, storage::Symbol;
                                 capacity_factor::Real=2.0,
                                 append_factor::Real=1.0,
                                 min_capacity::Int=16) where {N,T}
    storage == :dict && return bucket
    return SparsePauliVector(bucket; T=T, capacity_factor=capacity_factor,
                             append_factor=append_factor, min_capacity=min_capacity)
end

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
                  $project_root in LOAD_PATH || pushfirst!(LOAD_PATH, $project_root)
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
_dps_get(id::Symbol) = (haskey(_DPS_STORE, id) || error("No sharded Pauli operator $id on worker $(Distributed.myid())"); _DPS_STORE[id])
_dps_delete!(id::Symbol) =
    (delete!(_DPS_STORE, id); delete!(_DPS_STAGE, id); delete!(_DPS_PENDING, id); true)
_dps_length(id::Symbol) = length(_dps_get(id))
_dps_local_copy(id::Symbol) = copy(_dps_get(id))
_dps_fetch(f, pid::Int, args...) =
    pid == Distributed.myid() ? f(args...) : Distributed.remotecall_fetch(f, pid, args...)

# ---- distribute / gather ----

"""
    distribute(O; workers=workers(), id=nothing, storage=nothing) -> DistributedPauliSum

Shard a local `PauliSum` or `SparsePauliVector` across `workers` by hash of the
`PauliBasis`. `storage=:dict` stores each shard as a `PauliSum`; `storage=:spv`
stores each shard as a `SparsePauliVector`. When `storage` is omitted, the
`PAULI_STORAGE` environment variable may select `dict` or `spv`; otherwise
PauliSum inputs keep Dict shards and SparsePauliVector inputs keep SPV shards.
"""
function distribute(O::AnyPauliSum{N,T}; workers=Distributed.workers(), id=nothing,
                    storage=nothing, capacity_factor::Real=2.0,
                    append_factor::Real=1.0, min_capacity::Int=16) where {N,T}
    pids = ensure_pauli_workers!(workers=workers)
    store = pauli_storage(storage; default=_dps_default_storage(O))
    sid = id === nothing ? gensym(:dps) : Symbol(id)
    buckets = Dict(pid => PauliSum(N, T) for pid in pids)
    for (pb, c) in O
        buckets[_pauli_owner(pb, pids)][pb] = c
    end
    @sync for pid in pids
        @async begin
            local_bucket = _dps_materialize_bucket(buckets[pid], store;
                                                   capacity_factor=capacity_factor,
                                                   append_factor=append_factor,
                                                   min_capacity=min_capacity)
            if pid == Distributed.myid()
                _dps_store!(sid, local_bucket)
            else
                Distributed.remotecall_fetch(_dps_store!, pid, sid, local_bucket)
            end
        end
    end
    return DistributedPauliSum{N,T}(sid, pids, store)
end

"""
    DistributedPauliSum(N, T; workers=workers(), storage=nothing) -> empty sharded sum
"""
function DistributedPauliSum(N::Integer, ::Type{T}=ComplexF64; workers=Distributed.workers(),
                             id=nothing, storage=nothing, kwargs...) where {T}
    return distribute(PauliSum(N, T); workers=workers, id=id, storage=storage, kwargs...)
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
        local_terms = Distributed.remotecall_fetch(_dps_local_copy, pid, dO.id)
        for (pb, c) in local_terms
            out[pb] = get(out, pb, zero(T)) + c
        end
    end
    return out
end

"""
    collect_sparsepaulivector(dO; kwargs...) -> SparsePauliVector

Gather a sharded operator back to one local `SparsePauliVector` (debug/analysis;
do not use on sums larger than node memory).
"""
function collect_sparsepaulivector(dO::DistributedPauliSum{N,T}; kwargs...) where {N,T}
    return SparsePauliVector(collect_paulisum(dO); T=T, kwargs...)
end

function destroy!(dO::DistributedPauliSum)
    @sync for pid in dO.workers
        @async _dps_fetch(_dps_delete!, pid, dO.id)
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

# ---- Vector-staging rotation (opt-in: evolve!(...; staging=:vector)) ----
# Same math as the Dict-staging rotation, but each thread appends the sin terms
# to a per-owner Vector instead of inserting into a per-owner Dict. Within one
# rotation the sin bases G*p are distinct (p -> G*p is injective), so no dedup is
# needed and a plain push! (no hashing / probing / rehashing of a staging Dict)
# suffices. The merge/take/clear phases are container-agnostic (they iterate
# (pb,c) pairs), so only the local rotation changes.
function _rotate_chunk_vec!(O::AnyPauliSum{N,T}, ks, range, G, _sin, workers, stagevecs, coskeys) where {N,T}
    single = length(workers) == 1
    w1 = workers[1]
    for idx in range
        p = ks[idx]
        commute(p, G) && continue
        c = O[p]
        tmp = c * _sin * G * p
        qpb = PauliBasis(tmp)
        owner = single ? w1 : _pauli_owner(qpb, workers)
        push!(stagevecs[owner], (qpb, coeff(tmp)))
        push!(coskeys, p)
    end
    return nothing
end

function _dps_rotate_local_typed_vec!(id::Symbol, O::AnyPauliSum{N,T}, G, θ::Real, workers,
                                      threaded::Bool) where {N,T}
    _cos = cos(θ)
    _sin = 1im*sin(θ)
    ks = collect(keys(O))
    nt = (threaded && Threads.nthreads() > 1 && length(ks) > 1) ? Threads.nthreads() : 1
    VT = Tuple{PauliBasis{N},T}
    tstage = [Dict(pid => VT[] for pid in workers) for _ in 1:nt]
    tcos   = [Vector{eltype(ks)}() for _ in 1:nt]

    if nt == 1
        _rotate_chunk_vec!(O, ks, eachindex(ks), G, _sin, workers, tstage[1], tcos[1])
    else
        ranges = _dps_chunk_ranges(length(ks), nt)
        @sync for c in 1:nt
            Threads.@spawn _rotate_chunk_vec!(O, ks, ranges[c], G, _sin, workers, tstage[c], tcos[c])
        end
    end

    for c in 1:nt, p in tcos[c]
        O[p] = O[p] * _cos
    end

    if nt == 1
        _DPS_STAGE[id] = tstage[1]
    else
        stage = Dict(pid => VT[] for pid in workers)
        for c in 1:nt, pid in workers
            append!(stage[pid], tstage[c][pid])
        end
        _DPS_STAGE[id] = stage
    end
    return nothing
end
_dps_rotate_local_vec!(id::Symbol, G, θ::Real, workers, threaded::Bool=true) =
    _dps_rotate_local_typed_vec!(id, _dps_get(id), G, θ, workers, threaded)

# ---- Packed SparsePauliVector rotation ----
# SPV shards should not go through the generic AnyPauliSum interface: collecting
# keys and then using getindex/setindex turns the flat sorted arrays back into a
# binary-search workload. This path mirrors the local SPV rotation kernel on each
# shard, but routes the generated sin branches to their hash owners.
function _rotate_chunk_spv!(O::SparsePauliVector{N,W,T}, range,
                            gz::W, gx::W, ng::Int,
                            cosθ::Float64, sinθ::Float64,
                            workers,
                            stage::Dict{Int,Vector{Tuple{W,W,T}}}) where {N,W,T}
    single = length(workers) == 1
    w1 = workers[1]
    @inbounds for i in range
        zi = O.z[i]
        xi = O.x[i]
        m1 = count_ones(gx & zi)
        m2 = count_ones(gz & xi)
        iseven(m1 - m2) && continue

        zp = gz ⊻ zi
        xp = gx ⊻ xi
        k = (count_ones(zp & xp) - ng - count_ones(zi & xi) + 2 * m1) & 3
        cnew = (T(k - 2) * sinθ) * O.c[i]
        O.c[i] *= cosθ

        owner = single ? w1 : _pauli_owner(PauliBasis{N,W}(zp, xp), workers)
        push!(stage[owner], (zp, xp, cnew))
    end
    return nothing
end

function _dps_rotate_local_spv_typed!(id::Symbol, O::SparsePauliVector{N,W,T},
                                      G::PauliBasis{N}, θ::Real, workers,
                                      threaded::Bool) where {N,W,T}
    gz, gx = _pack(W, G)
    ng = count_ones(gz & gx)
    cosθ = cos(θ)
    sinθ = sin(θ)
    nt = (threaded && Threads.nthreads() > 1 &&
          O.n >= _SPV_THREADED_ROTATE_MIN) ? Threads.nthreads() : 1
    VT = Tuple{W,W,T}
    tstage = [Dict(pid => VT[] for pid in workers) for _ in 1:nt]

    if nt == 1
        _rotate_chunk_spv!(O, 1:O.n, gz, gx, ng, cosθ, sinθ, workers, tstage[1])
    else
        ranges = _dps_chunk_ranges(O.n, nt)
        @sync for c in 1:nt
            Threads.@spawn _rotate_chunk_spv!(O, ranges[c], gz, gx, ng, cosθ, sinθ,
                                              workers, tstage[c])
        end
    end

    if nt == 1
        _DPS_STAGE[id] = tstage[1]
    else
        stage = Dict(pid => VT[] for pid in workers)
        for c in 1:nt, pid in workers
            append!(stage[pid], tstage[c][pid])
        end
        _DPS_STAGE[id] = stage
    end
    return nothing
end

function _dps_rotate_local_spv!(id::Symbol, G::PauliBasis{N}, θ::Real, workers,
                                threaded::Bool=true) where {N}
    return _dps_rotate_local_spv_typed!(id, _dps_get(id), G, θ, workers, threaded)
end

function _dps_evolve_local_spv!(id::Symbol, G::PauliBasis{N}, θ::Real) where {N}
    evolve!(_dps_get(id), G, θ)
    return nothing
end

function _dps_evolve_local_spv_sequence!(id::Symbol,
                                         generators::Vector{<:PauliBasis{N}},
                                         angles::Vector{<:Real},
                                         truncation_thresh::Real,
                                         threaded::Bool,
                                         window::Int) where {N}
    O = _dps_get(id)
    gens = PauliBasis{N}[generators...]
    if truncation_thresh > 0
        evolve!(O, gens, angles; window=window,
                truncation=CoeffTruncation(truncation_thresh),
                threaded=threaded)
    else
        evolve!(O, gens, angles; window=window, threaded=threaded)
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

function _dps_pending_dict!(id::Symbol, ::Type{PauliBasis{N}}, ::Type{T}) where {N,T}
    return get!(_DPS_PENDING, id) do
        Tuple{PauliBasis{N},T}[]
    end
end

function _dps_rotate_pending_dict_typed!(id::Symbol, O::PauliSum{N,T},
                                         G::PauliBasis{N}, θ::Real) where {N,T}
    pending = _dps_pending_dict!(id, PauliBasis{N}, T)
    _cos = cos(θ)
    _sin = 1im * sin(θ)
    hi = length(pending)

    for (p, c) in O
        commute(p, G) && continue
        tmp = c * _sin * G * p
        push!(pending, (PauliBasis(tmp), convert(T, coeff(tmp))))
        O[p] = c * _cos
    end

    for i in 1:hi
        p, c = pending[i]
        commute(p, G) && continue
        tmp = c * _sin * G * p
        push!(pending, (PauliBasis(tmp), convert(T, coeff(tmp))))
        pending[i] = (p, c * _cos)
    end
    return nothing
end
_dps_rotate_pending_dict!(id::Symbol, G::PauliBasis{N}, θ::Real) where {N} =
    _dps_rotate_pending_dict_typed!(id, _dps_get(id), G, θ)

function _dps_route_pending_dict_typed!(id::Symbol, O::PauliSum{N,T},
                                        workers) where {N,T}
    pending = _dps_pending_dict!(id, PauliBasis{N}, T)
    VT = Tuple{PauliBasis{N},T}
    stage = Dict(pid => VT[] for pid in workers)
    for (pb, c) in pending
        push!(stage[_pauli_owner(pb, workers)], (pb, c))
    end
    empty!(pending)
    _DPS_STAGE[id] = stage
    return nothing
end
_dps_route_pending_dict!(id::Symbol, workers) =
    _dps_route_pending_dict_typed!(id, _dps_get(id), workers)

function _dps_merge_bucket!(O::SparsePauliVector{N,W,T}, bucket) where {N,W,T}
    m = length(bucket)
    m == 0 && return O
    length(O.ws) < m && resize!(O.ws, m)
    i = 0
    @inbounds for (pb, c) in bucket
        i += 1
        z, x = _pack(W, pb)
        O.ws[i] = (z, x, convert(T, c))
    end
    _sort_ws!(O.ws, 1, i)
    _merge_spv!(O, i, NOFILTER)
    return O
end

function _dps_merge_bucket!(O::SparsePauliVector{N,W,T},
                            bucket::Vector{Tuple{W,W,T}}) where {N,W,T}
    m = length(bucket)
    m == 0 && return O
    length(O.ws) < m && resize!(O.ws, m)
    @inbounds for i in 1:m
        O.ws[i] = bucket[i]
    end
    _sort_ws!(O.ws, 1, m)
    _merge_spv!(O, m, NOFILTER)
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
    if dO.storage == :spv && length(ws) == 1
        _dps_fetch(_dps_evolve_local_spv!, ws[1], dO.id, G, θ)
        return dO
    end
    rotate! = dO.storage == :spv ? _dps_rotate_local_spv! : _dps_rotate_local_vec!
    @sync for pid in ws
        @async _dps_fetch(rotate!, pid, dO.id, G, θ, ws, threaded)
    end
    @sync for pid in ws
        @async _dps_fetch(_dps_merge_incoming!, pid, dO.id, ws)
    end
    @sync for pid in ws
        @async _dps_fetch(_dps_clear_stage!, pid, dO.id)
    end
    return dO
end

"""
    evolve_vec!(dO::DistributedPauliSum, G::PauliBasis, θ::Real; threaded=true)

Same result as [`evolve!`](@ref) using the Vector-staging local rotation
(`_dps_rotate_local_vec!`): each thread appends the sin terms to a per-owner
`Vector` instead of a per-owner `Dict`, avoiding hash/probe/rehash in the hot
loop. This is also the default Dict-backed distributed path; the explicit entry
point is kept for callers that want to request it directly.
"""
function evolve_vec!(dO::DistributedPauliSum{N,T}, G::PauliBasis{N}, θ::Real;
                     threaded::Bool=true) where {N,T}
    ws = dO.workers
    @sync for pid in ws
        @async _dps_fetch(_dps_rotate_local_vec!, pid, dO.id, G, θ, ws, threaded)
    end
    @sync for pid in ws
        @async _dps_fetch(_dps_merge_incoming!, pid, dO.id, ws)
    end
    @sync for pid in ws
        @async _dps_fetch(_dps_clear_stage!, pid, dO.id)
    end
    return dO
end

"""
    coeff_clip!(dO::DistributedPauliSum, thresh::Real)

Drop terms with `|coeff| <= thresh` on every worker (global threshold, local work).
"""
function coeff_clip!(dO::DistributedPauliSum, thresh::Real)
    @sync for pid in dO.workers
        @async _dps_fetch(_dps_coeff_clip!, pid, dO.id, thresh)
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
                 threaded::Bool=true, window::Int=1) where {N,T}
    length(generators) == length(angles) || throw(DimensionMismatch("generators and angles must match"))
    window >= 1 || throw(ArgumentError("window must be >= 1"))
    if dO.storage == :spv && length(dO.workers) == 1
        _dps_fetch(_dps_evolve_local_spv_sequence!, dO.workers[1],
                   dO.id, generators, angles, truncation_thresh, threaded, window)
        return dO
    end
    if dO.storage == :dict && window > 1
        ws = dO.workers
        for (i, (G, θ)) in enumerate(zip(generators, angles))
            @sync for pid in ws
                @async _dps_fetch(_dps_rotate_pending_dict!, pid, dO.id, G, θ)
            end
            if i % window == 0 || i == length(generators)
                @sync for pid in ws
                    @async _dps_fetch(_dps_route_pending_dict!, pid, dO.id, ws)
                end
                @sync for pid in ws
                    @async _dps_fetch(_dps_merge_incoming!, pid, dO.id, ws)
                end
                @sync for pid in ws
                    @async _dps_fetch(_dps_clear_stage!, pid, dO.id)
                end
                truncation_thresh > 0 && coeff_clip!(dO, truncation_thresh)
            end
        end
        return dO
    end
    window == 1 || throw(ArgumentError("window > 1 is currently supported only for Dict storage or single-worker SPV"))
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
