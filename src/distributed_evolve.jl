compile(D::DistributedPauliSum{N}, generators::Vector{PauliBasis{N}}, angles::Vector{<:Real};
        window::Int=1) where N =
    compile(D.localpart.A, generators, angles; window, version=D.localpart.version)

"""
    evolve!(D::DistributedPauliSum{N,T}, G::PauliBasis{N}, θ::Real; s, counters)

One Pauli rotation on the local part. Purely local — rotations never block
on communication: sin-branch terms whose bin is owned by another rank
accumulate in local staging bins until the next merge boundary.
"""
function evolve!(D::DistributedPauliSum{N,T}, G::PauliBasis{N}, θ::Real;
                 s::Int=bin_shift(D.localpart.A, G), counters=nothing) where {N,T}
    evolve!(D.localpart, G, θ; s, counters)
    return D
end

"""
    merge_bins!(D::DistributedPauliSum, circ::CompiledCircuit, w::Int; counters)

Merge boundary for window `w`: ship all staging bins to their owners via the
paired-exchange schedule predicted by the window's shift subgroup, and merge
received terms into their true bins. Afterwards every term is on its owning
rank and duplicates are combined, so strict coefficient truncation and global
reductions are valid.
"""
function merge_bins!(D::DistributedPauliSum, circ::CompiledCircuit, w::Int; counters=nothing)
    circ.version == D.localpart.version ||
        error("CompiledCircuit version $(circ.version) does not match replicated state " *
              "version $(D.localpart.version). Recompile with `compile`.")
    pairwise_exchange!(D, circ.window_subgroups[w]; counters)
    return D
end

"""
    evolve!(D::DistributedPauliSum, circ::CompiledCircuit;
            truncation, local_truncation, correction, counters,
            rebalance_threshold=Inf)

Windowed distributed sequence evolution. Identical control flow to the
single-node binned driver: per rotation, a purely local rotation plus the
loose `local_truncation` (mid-window only); every `circ.window` rotations
(and at the end), a structured merge followed by the strict `truncation`
(whose corrections and any global thresholds ride the same merge boundary).

With a finite `rebalance_threshold`, each merge boundary also checks the
per-rank load from the allreduced bin histogram and, when
`max_load > threshold × mean_load`, reassigns whole bins to ranks
(`rebalance_bins!`) — a pure table edit that needs no recompile. All ranks
must call this collectively with identical arguments.
"""
function evolve!(D::DistributedPauliSum{N,T}, circ::CompiledCircuit{N};
                 truncation::TruncationStrategy=NoTruncation(),
                 local_truncation::TruncationStrategy=NoTruncation(),
                 correction::CorrectionAccumulator=NoCorrection(),
                 counters=nothing,
                 rebalance_threshold::Real=Inf) where {N,T}
    circ.version == D.localpart.version ||
        error("CompiledCircuit was compiled against RankMap version $(circ.version), " *
              "but the DistributedPauliSum is at version $(D.localpart.version). " *
              "Recompile with `compile`.")
    np = nranks(D)
    L = length(circ)
    for i in 1:L
        evolve!(D, circ.generators[i], circ.angles[i]; s=circ.shifts[i], counters)
        if i % circ.window == 0 || i == L
            merge_bins!(D, circ, cld(i, circ.window); counters)
            truncate!(D, truncation, correction)
            if isfinite(rebalance_threshold) && np > 1
                hist = bin_histogram(D)
                loads = zeros(Int, np)
                for b in eachindex(hist)
                    loads[D.localpart.bin_owner[b]+1] += hist[b]
                end
                total = sum(loads)
                if total > 0 && maximum(loads) > rebalance_threshold * total / np
                    rebalance_bins!(D; hist, counters)
                end
            end
        else
            # loose clip only mid-window (see the binned driver); applied to
            # the local part alone so no collectives run between merges
            local_truncation isa NoTruncation || truncate!(D.localpart, local_truncation)
        end
    end
    return D
end
