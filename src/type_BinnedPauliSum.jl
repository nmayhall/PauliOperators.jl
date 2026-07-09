"""
    BinnedPauliSum{N,T}

A `PauliSum` partitioned into `2^r` bins by a GF(2) `RankMap`. Every term is
stored under its true bin index (`bin_index(A, p)`), so duplicates always
co-locate and merging/coefficient truncation never require cross-bin lookups.

On a single node all bins are local. In a distributed run each MPI rank holds
this same structure: the bins it owns (per `bin_owner`) plus *staging* bins —
bins owned by other ranks that have accumulated locally-created terms awaiting
shipment at the next merge boundary.

`A`, `bin_owner`, and `version` are replicated state: `version` is bumped on
any change to the map or the ownership table, and guards against routing with
stale maps.
"""
mutable struct BinnedPauliSum{N,T}
    bins::Vector{PauliSum{N,T}}   # length nbins(A); bins[b+1] holds bin b (0-based ids)
    A::RankMap{N}
    bin_owner::Vector{Int32}      # bin b -> owning rank; all zero on a single node
    version::Int                  # bumped when A changes (invalidates compiled shifts)
    table_version::Int            # bumped when bin_owner changes (routing only)
end

"""
    default_bin_owner(nbins::Int, nranks::Int)

Contiguous block assignment of bins to ranks: `owner(b) = b ÷ (nbins ÷ nranks)`.
Keying ownership on the *high* rank bits keeps bins that differ in the
frequently-shifting low parity bits on the same rank. `nranks` must divide
`nbins`.
"""
function default_bin_owner(nbins::Int, nranks::Int)
    nbins % nranks == 0 || error("nranks=$nranks must divide nbins=$nbins")
    per = nbins ÷ nranks
    return Int32[div(b, per) for b in 0:nbins-1]
end

function BinnedPauliSum(O::PauliSum{N,T}, A::RankMap{N}; nranks::Int=1) where {N,T}
    bins = [PauliSum(N, T) for _ in 1:nbins(A)]
    for (p, c) in O
        bins[bin_index(A, p) + 1][p] = c
    end
    return BinnedPauliSum{N,T}(bins, A, default_bin_owner(nbins(A), nranks), 0, 0)
end

"""
    PauliSum(B::BinnedPauliSum{N,T})

Flatten back to a single `PauliSum`. Coefficients of duplicate Paulis across
bins (possible mid-window in a distributed run) are summed.
"""
function PauliSum(B::BinnedPauliSum{N,T}) where {N,T}
    out = PauliSum(N, T)
    for bin in B.bins
        sum!(out, bin)
    end
    return out
end

"""
    nonempty_bins(B::BinnedPauliSum)

Iterator over the 0-based indices of nonempty bins.
"""
nonempty_bins(B::BinnedPauliSum) = (b for b in 0:length(B.bins)-1 if !isempty(B.bins[b+1]))

"""
    owned_bins(B::BinnedPauliSum, myrank::Integer=0)

Iterator over the 0-based indices of bins owned by `myrank`.
"""
owned_bins(B::BinnedPauliSum, myrank::Integer=0) =
    (b for b in 0:length(B.bins)-1 if B.bin_owner[b+1] == myrank)

"""
    rebin!(B::BinnedPauliSum)

Re-derive every term's bin from its bits. A no-op when the binning invariant
holds; used after replacing `A` (which bumps `version`).
"""
function rebin!(B::BinnedPauliSum{N,T}) where {N,T}
    old = B.bins
    B.bins = [PauliSum(N, T) for _ in 1:nbins(B.A)]
    for bin in old
        for (p, c) in bin
            dest = B.bins[bin_index(B.A, p) + 1]
            dest[p] = get(dest, p, zero(T)) + c
        end
    end
    return B
end

"""
    check_binning(B::BinnedPauliSum) -> Bool

Invariant check: every stored term sits in the bin its bits map to.
"""
function check_binning(B::BinnedPauliSum)
    for b in 0:length(B.bins)-1
        for (p, _) in B.bins[b+1]
            bin_index(B.A, p) == b || return false
        end
    end
    return true
end

Base.length(B::BinnedPauliSum) = sum(length, B.bins)
Base.isempty(B::BinnedPauliSum) = all(isempty, B.bins)

"""
    bin_histogram(B::BinnedPauliSum)

Per-bin term counts (index `b+1` for bin `b`) — the balance-monitoring signal.
"""
bin_histogram(B::BinnedPauliSum) = [length(bin) for bin in B.bins]

function LinearAlgebra.norm(B::BinnedPauliSum{N,T}, p::Real=2) where {N,T}
    if p == 2
        return sqrt(sum(bin -> norm(bin, 2)^2, B.bins))
    elseif p == Inf
        return maximum(bin -> norm(bin, Inf), B.bins)
    else
        return sum(bin -> norm(bin, p)^p, B.bins)^(1/p)
    end
end

function LinearAlgebra.tr(B::BinnedPauliSum)
    # The identity Pauli always lives in bin 0
    return tr(B.bins[1])
end

function inner_product(B1::BinnedPauliSum{N,T}, B2::BinnedPauliSum{N,T}) where {N,T}
    B1.version == B2.version ||
        error("inner_product requires BinnedPauliSums with the same RankMap version")
    out = T(0)
    for (bin1, bin2) in zip(B1.bins, B2.bins)
        out += inner_product(bin1, bin2)
    end
    return out
end

function expectation_value(B::BinnedPauliSum{N}, ψ::Ket{N}) where N
    return sum(bin -> expectation_value(bin, ψ), B.bins)
end

# ============================================================
# Truncation on binned sums
# ============================================================

# Element-local strategies (everything that decides keep/drop from one term
# alone) apply independently per bin.
function _apply!(B::BinnedPauliSum, s::TruncationStrategy)
    for bin in B.bins
        _apply!(bin, s)
    end
    return B
end

# Global strategies need the coefficient population across all bins.
function _apply!(B::BinnedPauliSum{N,T}, s::AdaptiveTruncation) where {N,T}
    if length(B) > s.max_terms
        coeffs = sort!(abs.(reduce(vcat, [collect(values(bin)) for bin in B.bins])))
        thresh = coeffs[end - s.max_terms]
        for bin in B.bins
            coeff_clip!(bin, thresh)
        end
    else
        for bin in B.bins
            coeff_clip!(bin, s.min_thresh)
        end
    end
    return B
end

function _apply!(B::BinnedPauliSum, s::StochasticSamplingTruncation)
    error("StochasticSamplingTruncation on a BinnedPauliSum is not supported yet: " *
          "it requires globally-coordinated weighted sampling.")
end

function _apply!(B::BinnedPauliSum, s::CompositeTruncation)
    _apply_tup!(B, s.strategies)
    return B
end

function _measure(B::BinnedPauliSum, corr::CorrectionAccumulator)
    return _measure(PauliSum(B), corr)
end
_measure(::BinnedPauliSum, ::NoCorrection) = nothing
function _measure(B::BinnedPauliSum{N}, corr::EnergyCorrection{N}) where N
    return (energy = real(expectation_value(B, corr.ψ)),)
end

function truncate!(B::BinnedPauliSum, strategy::TruncationStrategy,
                   corr::CorrectionAccumulator=NoCorrection())
    before = _measure(B, corr)
    _apply!(B, strategy)
    after = _measure(B, corr)
    _accumulate!(corr, before, after)
    return B
end

function greedy_bisection_rankmap(B::BinnedPauliSum{N}, r::Int; kw...) where N
    terms = PauliBasis{N}[]
    for bin in B.bins, (p, _) in bin
        push!(terms, p)
    end
    return greedy_bisection_rankmap(terms, r; kw...)
end

"""
    swap_row!(B::BinnedPauliSum{N}, i::Int, newrow::RankRow)

Replace row `i` of the rank map (bumping the map version — compiled circuits
must be recompiled) and rebin locally. Terms move only between bin pairs
differing in rank bit `i`, so on a distributed sum the follow-up exchange is
a single pairwise displacement (see the `DistributedPauliSum` method).
"""
function swap_row!(B::BinnedPauliSum{N,T}, i::Int, newrow::RankRow) where {N,T}
    1 <= i <= nbits(B.A) || error("row index $i outside 1:$(nbits(B.A))")
    rows = copy(B.A.rows)
    rows[i] = newrow
    B.A = RankMap{N}(rows)
    B.version += 1
    rebin!(B)
    return B
end
