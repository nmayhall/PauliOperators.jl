"""
    PauliRecord{T}

Flat isbits wire format for shipping Pauli terms between ranks: the two
symplectic bitstrings plus the coefficient (product phases are already folded
into the coefficient by the rotation kernel before serialization).
"""
struct PauliRecord{T}
    z::Int128
    x::Int128
    c::T
end

# Records travel as UInt64 words so we depend only on the standard
# MPI_UINT64_T datatype, never on derived-datatype support for Int128.
_words_per_record(::Type{PauliRecord{T}}) where T = sizeof(PauliRecord{T}) ÷ 8

function _to_words(records::Vector{PauliRecord{T}}) where T
    isbitstype(PauliRecord{T}) || error("PauliRecord{$T} must be isbits for MPI transport")
    sizeof(PauliRecord{T}) % 8 == 0 || error("sizeof(PauliRecord{$T}) must be a multiple of 8")
    words = Vector{UInt64}(undef, _words_per_record(PauliRecord{T}) * length(records))
    copyto!(words, reinterpret(UInt64, records))
    return words
end

function _from_words(::Type{PauliRecord{T}}, words::Vector{UInt64}) where T
    wpr = _words_per_record(PauliRecord{T})
    length(words) % wpr == 0 || error("word buffer length is not a whole number of records")
    records = Vector{PauliRecord{T}}(undef, length(words) ÷ wpr)
    copyto!(reinterpret(UInt64, records), words)
    return records
end

"""
    DistributedPauliSum{N,T}

A `PauliSum` sharded across the ranks of an MPI communicator by a GF(2)
`RankMap`. Each rank holds a full-length `BinnedPauliSum`: the bins it owns
plus staging bins for terms created locally that belong to other ranks'
bins. The rank map, ownership table, and version are replicated state —
identical on every rank.

Requires `MPI.Init()` to have been called, and a power-of-two number of
ranks dividing `nbins`.
"""
struct DistributedPauliSum{N,T}
    comm::MPI.Comm
    localpart::BinnedPauliSum{N,T}
end

function DistributedPauliSum(O::PauliSum{N,T}, A::RankMap{N}, comm::MPI.Comm) where {N,T}
    MPI.Initialized() ||
        error("MPI has not been initialized: call MPI.Init() before constructing a DistributedPauliSum")
    nranks = MPI.Comm_size(comm)
    myrank = MPI.Comm_rank(comm)
    ispow2(nranks) ||
        error("DistributedPauliSum requires a power-of-two number of ranks, got $nranks")
    B = BinnedPauliSum(O, A; nranks)
    for b in 0:nbins(A)-1
        B.bin_owner[b+1] == myrank || empty!(B.bins[b+1])
    end
    return DistributedPauliSum{N,T}(comm, B)
end

myrank(D::DistributedPauliSum) = MPI.Comm_rank(D.comm)
nranks(D::DistributedPauliSum) = MPI.Comm_size(D.comm)
owned_bins(D::DistributedPauliSum) = owned_bins(D.localpart, myrank(D))

"""
    check_ownership(D::DistributedPauliSum) -> Bool

Invariant check for merge boundaries: every nonempty local bin is owned by
this rank (no staging terms awaiting shipment).
"""
function check_ownership(D::DistributedPauliSum)
    me = myrank(D)
    B = D.localpart
    for b in nonempty_bins(B)
        B.bin_owner[b+1] == me || return false
    end
    return true
end

# ============================================================
# Paired exchange (design-doc primitive 1)
# ============================================================

# Which rank pairs (me, me ⊻ ρ) can traffic flow between, computed from
# replicated state alone so both sides of every Sendrecv! agree without
# negotiating. `dests[b+1]` is where bin b's local content should go (its
# current owner for merges, its new owner during a reassignment).
# For merges the possible sources of staged content in bin b⊻σ are owned
# bins b (σ from the window's shift subgroup); a pair is active if any such
# (b, b⊻σ) straddles it. For reassignments the old→new owner map decides.
function _active_partners(B::BinnedPauliSum, subgroup, np::Int, me::Int)
    active = falses(np - 1)
    if subgroup === nothing
        active .= true
        return active
    end
    for σ in subgroup
        σ == 0 && continue
        for b in 0:length(B.bins)-1
            o1, o2 = Int(B.bin_owner[b+1]), Int(B.bin_owner[(b ⊻ σ)+1])
            if o1 == me && o2 != me
                active[(me ⊻ o2)] = true
            end
        end
    end
    return active
end

# Deadlock-free exchange core: walk ρ = 1..np-1 (the XOR schedule — identical
# order on every rank), and for each active pair Sendrecv! a version-tagged
# header then the payload. Every local nonempty bin whose current owner is
# another rank is shipped to that owner; received records are re-binned from
# their bits (invariant 1) and merged.
function _exchange!(D::DistributedPauliSum{N,T}, active; counters=nothing) where {N,T}
    B = D.localpart
    me = myrank(D)
    np = nranks(D)

    shipped = 0
    bytes = 0
    for ρ in 1:np-1
        active[ρ] || continue
        partner = me ⊻ ρ
        records = PauliRecord{T}[]
        for b in nonempty_bins(B)
            B.bin_owner[b+1] == partner || continue
            bin = B.bins[b+1]
            for (p, c) in bin
                push!(records, PauliRecord{T}(p.z, p.x, c))
            end
            empty!(bin)
        end
        sendwords = _to_words(records)

        sendheader = Int64[B.version, B.table_version, length(sendwords)]
        recvheader = Int64[0, 0, 0]
        MPI.Sendrecv!(sendheader, recvheader, D.comm;
                      dest=partner, sendtag=0, source=partner, recvtag=0)
        (recvheader[1] == B.version && recvheader[2] == B.table_version) ||
            error("_exchange!: version mismatch with rank $partner " *
                  "(($(recvheader[1]),$(recvheader[2])) vs ($(B.version),$(B.table_version))) " *
                  "— replicated state out of sync")

        recvwords = Vector{UInt64}(undef, recvheader[3])
        MPI.Sendrecv!(sendwords, recvwords, D.comm;
                      dest=partner, sendtag=1, source=partner, recvtag=1)

        for rec in _from_words(PauliRecord{T}, recvwords)
            p = PauliBasis{N}(rec.z, rec.x)
            b = bin_index(B.A, p)
            B.bin_owner[b+1] == me ||
                error("_exchange!: received a term for bin $b owned by rank $(B.bin_owner[b+1])")
            dest = B.bins[b+1]
            dest[p] = get(dest, p, zero(T)) + rec.c
        end

        shipped += length(records)
        bytes += length(sendwords) * 8
    end

    check_ownership(D) ||
        error("_exchange!: staging bins remain after exchange; " *
              "the predicted partner set did not cover all displacements")

    _record_merge!(counters, shipped, bytes)
    return D
end

"""
    pairwise_exchange!(D::DistributedPauliSum, subgroup; counters=nothing)

Ship every nonempty non-owned (staging) bin to its owning rank and merge the
received terms into their true bins. `subgroup` is the GF(2) span of bin
displacements that can be populated (from the `CompiledCircuit`), or
`nothing` for "any displacement possible". The partner set is derived from
the subgroup and the replicated ownership table (works for arbitrary tables,
not just the contiguous default), and pairs exchange in XOR-schedule order —
deterministic and deadlock free. Headers carry both replicated-state
versions, which must match on both sides.
"""
function pairwise_exchange!(D::DistributedPauliSum, subgroup; counters=nothing)
    active = _active_partners(D.localpart, subgroup, nranks(D), myrank(D))
    return _exchange!(D, active; counters)
end

"""
    rebalance_bins!(D::DistributedPauliSum; hist=bin_histogram(D), counters=nothing)

Reassign whole bins to ranks (LPT greedy: heaviest bin to the least-loaded
rank, ties keeping the current owner) and ship the moved bins. Pure table
edit — the rank map and all compiled shifts are untouched, so no recompile
is needed. Call at merge boundaries only. Returns `true` if the table
changed. All ranks must call collectively; the decision is deterministic
from the allreduced histogram, so every rank computes the same new table.
"""
function rebalance_bins!(D::DistributedPauliSum; hist::Vector{Int}=bin_histogram(D),
                         counters=nothing)
    B = D.localpart
    np = nranks(D)
    nb = length(B.bins)
    order = sort(1:nb, by = b -> (-hist[b], b))
    loads = zeros(Int, np)
    newowner = similar(B.bin_owner)
    for b in order
        least = minimum(loads)
        cur = Int(B.bin_owner[b]) + 1
        r = loads[cur] == least ? cur : argmin(loads)   # sticky tie-break
        newowner[b] = r - 1
        loads[r] += hist[b]
    end
    newowner == B.bin_owner && return false

    old = copy(B.bin_owner)
    me = myrank(D)
    active = falses(np - 1)
    for b in 1:nb
        o, n = Int(old[b]), Int(newowner[b])
        if o != n && (o == me || n == me)
            active[me ⊻ (o == me ? n : o)] = true
        end
    end
    B.bin_owner = newowner
    B.table_version += 1
    _exchange!(D, active; counters)
    return true
end

"""
    swap_row!(D::DistributedPauliSum{N}, i::Int, newrow::RankRow)

Replace row `i` of the rank map across the distributed sum (collective).
Terms move only between bin pairs differing in rank bit `i`, so the follow-up
shipping is one pairwise displacement. Bumps the map version: compiled
circuits must be recompiled. Call at merge boundaries only.
"""
function swap_row!(D::DistributedPauliSum{N}, i::Int, newrow::RankRow) where N
    swap_row!(D.localpart, i, newrow)
    pairwise_exchange!(D, [0, 1 << (i - 1)])
    return D
end

# ============================================================
# Small allreduce reductions (design-doc primitive 2)
# ============================================================
# Valid on merged data (at merge boundaries): mid-window, a Pauli's
# coefficient may be split across ranks and nonlinear reductions of it
# would be wrong.

function LinearAlgebra.norm(D::DistributedPauliSum{N,T}, p::Real=2) where {N,T}
    if p == 2
        local_sq = sum(bin -> norm(bin, 2)^2, D.localpart.bins)
        return sqrt(MPI.Allreduce(local_sq, +, D.comm))
    elseif p == Inf
        return MPI.Allreduce(norm(D.localpart, Inf), max, D.comm)
    else
        local_p = sum(bin -> norm(bin, p)^p, D.localpart.bins)
        return MPI.Allreduce(local_p, +, D.comm)^(1/p)
    end
end

function inner_product(D1::DistributedPauliSum{N,T}, D2::DistributedPauliSum{N,T}) where {N,T}
    (D1.localpart.version == D2.localpart.version &&
     D1.localpart.table_version == D2.localpart.table_version) ||
        error("inner_product requires DistributedPauliSums with the same RankMap " *
              "version and ownership table")
    return MPI.Allreduce(inner_product(D1.localpart, D2.localpart), +, D1.comm)
end

function expectation_value(D::DistributedPauliSum{N}, ψ::Ket{N}) where N
    return MPI.Allreduce(expectation_value(D.localpart, ψ), +, D.comm)
end

Base.length(D::DistributedPauliSum) = MPI.Allreduce(length(D.localpart), +, D.comm)

LinearAlgebra.tr(D::DistributedPauliSum) = MPI.Allreduce(tr(D.localpart), +, D.comm)

"""
    bin_histogram(D::DistributedPauliSum)

Global per-bin term counts (allreduce of the local histograms) — the balance
monitoring signal.
"""
bin_histogram(D::DistributedPauliSum) = MPI.Allreduce(bin_histogram(D.localpart), +, D.comm)

"""
    gather(D::DistributedPauliSum) -> Union{PauliSum, Nothing}

Gather the full sum onto rank 0 (returns `nothing` elsewhere). For testing
and debugging only: production code should never gather the full sum.
"""
function gather(D::DistributedPauliSum{N,T}) where {N,T}
    records = PauliRecord{T}[]
    for b in nonempty_bins(D.localpart)
        for (p, c) in D.localpart.bins[b+1]
            push!(records, PauliRecord{T}(p.z, p.x, c))
        end
    end
    words = _to_words(records)
    counts = MPI.Allgather(Int32(length(words)), D.comm)
    if myrank(D) == 0
        recvbuf = MPI.VBuffer(Vector{UInt64}(undef, sum(counts)), counts)
        MPI.Gatherv!(words, recvbuf, D.comm; root=0)
        out = PauliSum(N, T)
        for rec in _from_words(PauliRecord{T}, recvbuf.data)
            p = PauliBasis{N}(rec.z, rec.x)
            out[p] = get(out, p, zero(T)) + rec.c
        end
        return out
    else
        MPI.Gatherv!(words, nothing, D.comm; root=0)
        return nothing
    end
end

# ============================================================
# Distributed truncation
# ============================================================

# Element-local strategies decide keep/drop from one term alone and apply
# independently per rank (including staging bins), with no communication.
function _apply!(D::DistributedPauliSum, s::TruncationStrategy)
    _apply!(D.localpart, s)
    return D
end

# Global threshold selection via an allreduced histogram of |c| (approximate
# order statistic: guaranteed to keep at most max_terms, may clip slightly
# below the exact k-th coefficient). Valid at merge boundaries.
function _apply!(D::DistributedPauliSum{N,T}, s::AdaptiveTruncation) where {N,T}
    total = length(D)
    if total > s.max_terms
        # log10(|c|) histogram over [-16, 16), 64 buckets of half a decade
        nbuckets = 64
        lo, hi = -16.0, 16.0
        counts = zeros(Int, nbuckets + 2)   # [below lo; buckets; at/above hi]
        for bin in D.localpart.bins
            for (_, c) in bin
                l = log10(abs(c))
                idx = l < lo ? 1 : l >= hi ? nbuckets + 2 :
                      2 + floor(Int, (l - lo) / (hi - lo) * nbuckets)
                counts[idx] += 1
            end
        end
        counts = MPI.Allreduce(counts, +, D.comm)
        # walk from the top: keep whole buckets while they fit in max_terms
        kept = counts[nbuckets + 2]
        cut = nbuckets + 1     # index of first (highest) bucket NOT kept
        while cut >= 2 && kept + counts[cut] <= s.max_terms
            kept += counts[cut]
            cut -= 1
        end
        thresh = cut == 1 ? 0.0 : 10.0^(lo + (cut - 1) * (hi - lo) / nbuckets)
        for bin in D.localpart.bins
            coeff_clip!(bin, thresh)
        end
    else
        for bin in D.localpart.bins
            coeff_clip!(bin, s.min_thresh)
        end
    end
    return D
end

function _apply!(D::DistributedPauliSum, ::StochasticSamplingTruncation)
    error("StochasticSamplingTruncation on a DistributedPauliSum is not supported yet: " *
          "it requires distributed weighted sampling without replacement.")
end

function _apply!(D::DistributedPauliSum, s::CompositeTruncation)
    _apply_tup!(D, s.strategies)
    return D
end

_measure(::DistributedPauliSum, ::NoCorrection) = nothing

function _measure(D::DistributedPauliSum{N}, corr::EnergyCorrection{N}) where N
    return (energy = real(expectation_value(D, corr.ψ)),)
end

function _measure(::DistributedPauliSum, ::EnergyVarianceCorrection)
    error("EnergyVarianceCorrection on a DistributedPauliSum is not supported yet: " *
          "variance needs ⟨O²⟩, whose cross terms span ranks. Use EnergyCorrection, " *
          "or accumulate variance corrections on a gathered copy.")
end

function truncate!(D::DistributedPauliSum, strategy::TruncationStrategy,
                   corr::CorrectionAccumulator=NoCorrection())
    before = _measure(D, corr)
    _apply!(D, strategy)
    after = _measure(D, corr)
    _accumulate!(corr, before, after)
    return D
end

# ============================================================
# Distributed greedy bisection
# ============================================================

"""
    greedy_bisection_rankmap(D::DistributedPauliSum, r; protected, ncandidates, rng)

Distributed greedy row selection: rank 0 draws each candidate batch
(broadcast so all ranks agree), every rank counts how each candidate splits
its local terms, and one allreduce per batch produces the global split
counts. All ranks deterministically pick the same winner. Call at merge
boundaries (mid-window staging duplicates would be double-counted). Returns
the new `RankMap`; re-sharding onto it is up to the caller.
"""
function greedy_bisection_rankmap(D::DistributedPauliSum{N,T}, r::Int;
                                  protected::Vector{PauliBasis{N}}=PauliBasis{N}[],
                                  ncandidates::Int=32,
                                  rng::Random.AbstractRNG=Random.default_rng()) where {N,T}
    basis = protected_row_basis(protected)
    length(basis) >= r ||
        error("cannot build $r independent rows: the constraint nullspace has " *
              "dimension $(length(basis))")
    terms = PauliBasis{N}[]
    for b in nonempty_bins(D.localpart), (p, _) in D.localpart.bins[b+1]
        push!(terms, p)
    end
    rows = RankRow[]
    chosen = RankRow[]
    partial = zeros(Int, length(terms))
    for i in 1:r
        cands = myrank(D) == 0 ? _draw_candidates(basis, chosen, ncandidates, rng, N) :
                                 RankRow[]
        cands = MPI.bcast(cands, D.comm; root=0)
        counts = MPI.Allreduce(_split_counts(terms, partial, cands, i), +, D.comm)
        best = argmin(j -> _balance_score(view(counts, :, j)), 1:length(cands))
        row = cands[best]
        push!(chosen, _gf2_reduce(row, chosen, N))
        push!(rows, row)
        for (t, p) in enumerate(terms)
            partial[t] |= _row_parity(row, p) << (i - 1)
        end
    end
    return RankMap{N}(rows)
end
