"""
    PropagationCounters()

Measurement counters for binned/distributed propagation (design invariant:
everything measurable). `moved_per_rotation[i]` counts sin-branch terms that
changed bin during rotation `i` — exactly zero for protected generators.
`shipped_per_merge`/`bytes_per_merge` count cross-rank traffic at each merge
boundary (always zero on a single node).
"""
mutable struct PropagationCounters
    rotations::Int
    merges::Int
    terms_created::Int
    moved_per_rotation::Vector{Int}
    shipped_per_merge::Vector{Int}
    bytes_per_merge::Vector{Int}
end
PropagationCounters() = PropagationCounters(0, 0, 0, Int[], Int[], Int[])

function _record_rotation!(c::PropagationCounters, moved::Int, created::Int)
    c.rotations += 1
    c.terms_created += created
    push!(c.moved_per_rotation, moved)
    return c
end
_record_rotation!(::Nothing, moved::Int, created::Int) = nothing

function _record_merge!(c::PropagationCounters, shipped::Int, bytes::Int)
    c.merges += 1
    push!(c.shipped_per_merge, shipped)
    push!(c.bytes_per_merge, bytes)
    return c
end
_record_merge!(::Nothing, shipped::Int, bytes::Int) = nothing


"""
    gf2_span(shifts::Vector{Int}; cap::Int=12)

The GF(2) span (xor-closure) of a set of bin shifts, always including 0.
Returns the sorted span, or `nothing` if the xor-basis has more than `cap`
independent elements (span too large to be a useful communication schedule —
treat as "all bins possible").
"""
function gf2_span(shifts::Vector{Int}; cap::Int=12)
    basis = Int[]
    for s in shifts
        v = s
        for b in basis
            v = min(v, v ⊻ b)
        end
        v != 0 && push!(basis, v)
    end
    length(basis) > cap && return nothing
    span = [0]
    for b in basis
        append!(span, [x ⊻ b for x in span])
    end
    return sort!(span)
end

"""
    CompiledCircuit{N}

Precomputed shift bookkeeping for a fixed rotation sequence under a fixed
`RankMap` (design invariant: the shift bookkeeping is precomputable). Holds
the per-generator bin shifts `s_G` and, per merge window, the GF(2) span of
the nonzero shifts in that window — the set of bin displacements (and hence
exchange partners) that can be populated at the window's merge boundary.

Stamped with the `version` of the map it was compiled against; using it with
a `BinnedPauliSum` at a different version is an error.
"""
struct CompiledCircuit{N}
    generators::Vector{PauliBasis{N}}
    angles::Vector{Float64}
    shifts::Vector{Int}
    window::Int
    window_subgroups::Vector{Union{Nothing,Vector{Int}}}
    version::Int
end

Base.length(circ::CompiledCircuit) = length(circ.generators)
nwindows(circ::CompiledCircuit) = cld(length(circ), circ.window)

"""
    compile(A::RankMap{N}, generators, angles; window=1, version=0)
    compile(B::BinnedPauliSum{N}, generators, angles; window=1)

Precompute the `CompiledCircuit` for a rotation sequence (e.g. from
`trotterize`): per-generator shifts and per-window shift subgroups. `window`
is the merge cadence M (`window=1` = eager: merge after every rotation).
Recompile whenever the rank map changes.
"""
function compile(A::RankMap{N}, generators::Vector{PauliBasis{N}}, angles::Vector{<:Real};
                 window::Int=1, version::Int=0) where N
    length(generators) == length(angles) ||
        throw(DimensionMismatch("generators and angles must have same length"))
    window >= 1 || throw(ArgumentError("window must be >= 1"))

    shifts = [bin_shift(A, G) for G in generators]
    if nbits(A) > 0 && !isempty(shifts) && all(==(0), shifts)
        @warn "All generator shifts are zero: the population will never leave its " *
              "initial bins. Add rows sensitive to the generators (e.g. z-pair slots " *
              "for diagonal-heavy models) or un-protect a generator family."
    end

    L = length(generators)
    subgroups = Vector{Union{Nothing,Vector{Int}}}(undef, cld(L, window))
    for w in 1:cld(L, window)
        lo, hi = (w - 1) * window + 1, min(w * window, L)
        subgroups[w] = gf2_span(shifts[lo:hi])
    end

    return CompiledCircuit{N}(generators, Float64.(angles), shifts, window, subgroups, version)
end

compile(B::BinnedPauliSum{N}, generators::Vector{PauliBasis{N}}, angles::Vector{<:Real};
        window::Int=1) where N =
    compile(B.A, generators, angles; window, version=B.version)


# ============================================================
# Binned rotation kernel
# ============================================================

# The sin-branch splitting loop of the serial kernel (evolve.jl), with the
# sin branch routed to an outbox instead of merged back into the same sum.
function _evolve_bin_split!(bin::PauliSum{N,T}, G::PauliBasis{N}, θ::Real,
                            outbox::PauliSum{N,T}) where {N,T}
    _cos = cos(θ)
    _sin = 1im * sin(θ)
    for (p, c) in bin
        commute(p, G) && continue
        tmp = c * _sin * G * p
        pb = PauliBasis(tmp)
        outbox[pb] = get(outbox, pb, zero(T)) + coeff(tmp)
        bin[p] *= _cos
    end
    return bin
end

"""
    evolve!(B::BinnedPauliSum{N,T}, G::PauliBasis{N}, θ::Real; s, counters)

One Pauli rotation on a binned sum. If the generator's shift `s` is zero the
rotation is bin-preserving and the unchanged serial kernel runs on each
nonempty bin. Otherwise the rotation is two-phase: (1) every nonempty bin's
sin branch accumulates into an outbox for bin `b ⊻ s`; (2) outboxes are
delivered. The two phases prevent a sin branch landing in a
not-yet-processed bin from being rotated twice.

Every term ends in its true bin (`bin_index(A, p)`), including terms whose
bin is owned by another rank in a distributed run (staging bins) — shipping
happens at merge boundaries, never here.
"""
function evolve!(B::BinnedPauliSum{N,T}, G::PauliBasis{N}, θ::Real;
                 s::Int=bin_shift(B.A, G), counters=nothing) where {N,T}
    before = counters === nothing ? 0 : length(B)
    moved = 0
    if s == 0
        for b in nonempty_bins(B)
            evolve!(B.bins[b+1], G, θ)
        end
    else
        outboxes = Dict{Int,PauliSum{N,T}}()
        for b in nonempty_bins(B)
            ob = get!(() -> PauliSum(N, T), outboxes, b ⊻ s)
            _evolve_bin_split!(B.bins[b+1], G, θ, ob)
        end
        for (j, ob) in outboxes
            moved += length(ob)
            sum!(B.bins[j+1], ob)
        end
    end
    if counters !== nothing
        _record_rotation!(counters, moved, length(B) - before)
    end
    return B
end

"""
    merge_bins!(B::BinnedPauliSum, circ::CompiledCircuit, w::Int; counters)

Merge-boundary hook. On a single node every term is already in its true bin,
so this only updates counters; the distributed method ships staging bins to
their owners here.
"""
function merge_bins!(B::BinnedPauliSum, circ::CompiledCircuit, w::Int; counters=nothing)
    _record_merge!(counters, 0, 0)
    return B
end

"""
    evolve!(B::BinnedPauliSum, circ::CompiledCircuit;
            truncation, local_truncation, correction, counters)

Windowed sequence evolution (the binned analogue of
`evolve(O, generators, angles; ...)`). Every `circ.window` rotations (and at
the end): merge, then apply the strict `truncation` with `correction`
accumulation on merged data. Between merges, each rotation is followed by
the loose `local_truncation`, which controls memory growth mid-window. With
`window=1` (eager) the loose truncation never runs and the result matches
the serial per-rotation-truncated evolution exactly.
"""
function evolve!(B::BinnedPauliSum{N,T}, circ::CompiledCircuit{N};
                 truncation::TruncationStrategy=NoTruncation(),
                 local_truncation::TruncationStrategy=NoTruncation(),
                 correction::CorrectionAccumulator=NoCorrection(),
                 counters=nothing) where {N,T}
    circ.version == B.version ||
        error("CompiledCircuit was compiled against RankMap version $(circ.version), " *
              "but the BinnedPauliSum is at version $(B.version). Recompile with `compile`.")
    L = length(circ)
    for i in 1:L
        evolve!(B, circ.generators[i], circ.angles[i]; s=circ.shifts[i], counters)
        if i % circ.window == 0 || i == L
            merge_bins!(B, circ, cld(i, circ.window); counters)
            truncate!(B, truncation, correction)
        else
            # loose clip only mid-window: it controls memory between merges,
            # and clipping possibly-split coefficients right before a merge
            # would combine them is pointless and lossy
            local_truncation isa NoTruncation || truncate!(B, local_truncation)
        end
    end
    return B
end
