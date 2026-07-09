"""
    RankRow(z::Int128, x::Int128)

One row of a GF(2) rank map: a 2N-bit parity check on the symplectic
representation of a Pauli, split into a z-half and an x-half with the same
bit layout as `PauliBasis`. The row's answer for a Pauli `p` is

    parity(count_ones(p.z & row.z)) ⊻ parity(count_ones(p.x & row.x))
"""
struct RankRow
    z::Int128
    x::Int128
end

RankRow(z::Integer, x::Integer) = RankRow(Int128(z), Int128(x))

"""
    RankRow(N::Integer; z=[], x=[])

Build a row from lists of watched qubit indices (1-based), mirroring the
`Pauli(N; X=, Z=)` convention. Geometric reading: a row watching only the
`x` slots of a set of sites is a spatial cut — a hopping generator gets a
nonzero shift on it exactly when the hop crosses the cut boundary, while all
diagonal (Z-only) generators are automatically communication free. Rows
watching `z` slots in same-site pairs split the diagonal sector while
keeping on-site ZZ generators free (the diagonal-trap fix).
"""
function RankRow(N::Integer; z=Int[], x=Int[])
    zmask = Int128(0)
    xmask = Int128(0)
    for i in z
        1 <= i <= N || throw(ArgumentError("z index $i outside 1:$N"))
        zmask |= Int128(1) << (i - 1)
    end
    for i in x
        1 <= i <= N || throw(ArgumentError("x index $i outside 1:$N"))
        xmask |= Int128(1) << (i - 1)
    end
    return RankRow(zmask, xmask)
end

"""
    RankMap{N}(rows::Vector{RankRow})

A linear map over GF(2) from N-qubit Paulis to bin indices. Each of the `r`
rows contributes one bit of the (0-based) bin index, so a `RankMap` with `r`
rows partitions Pauli space into `2^r` bins.

Because Pauli multiplication XORs the z and x bitstrings, `bin_index` is
linear: `bin_index(A, PauliBasis(G*p)) == bin_index(A, p) ⊻ bin_shift(A, G)`.
This is the identity that makes distributed routing structured — see
`bin_shift`.
"""
struct RankMap{N}
    rows::Vector{RankRow}

    function RankMap{N}(rows::Vector{RankRow}) where N
        N isa Integer || error("RankMap{N}: N must be an integer")
        1 <= N <= 127 || error("RankMap{N}: N must be in 1:127")
        length(rows) <= 20 || error("RankMap: more than 2^20 bins is not supported")
        mask = N == 127 ? typemax(Int128) : (Int128(1) << N) - 1
        for row in rows
            (row.z & ~mask == 0 && row.x & ~mask == 0) ||
                error("RankRow watches bits outside the N=$N qubit register")
        end
        return new{N}(rows)
    end
end

"""
    nbits(A::RankMap)

Number of parity bits (rows) in the rank map.
"""
nbits(A::RankMap) = length(A.rows)

"""
    nbins(A::RankMap)

Number of bins the rank map partitions Pauli space into: `2^nbits(A)`.
"""
nbins(A::RankMap) = 1 << nbits(A)

"""
    bin_index(A::RankMap{N}, p::PauliBasis{N})

The (0-based) bin index of `p`: row `i` of `A` contributes bit `i-1`, the
parity of the overlap between `p`'s symplectic bits and the row's watched
slots.
"""
# parity of the 2N-bit GF(2) dot product between a row and a Pauli's
# symplectic vector
@inline _row_parity(row::RankRow, p::PauliBasis) =
    (count_ones(p.z & row.z) ⊻ count_ones(p.x & row.x)) & 1

@inline function bin_index(A::RankMap{N}, p::PauliBasis{N}) where N
    b = 0
    @inbounds for i in eachindex(A.rows)
        b |= _row_parity(A.rows[i], p) << (i - 1)
    end
    return b
end

"""
    bin_shift(A::RankMap{N}, G::PauliBasis{N})

The fixed bin shift `s_G` induced by multiplication with `G`: every term `p`
satisfies `bin_index(A, PauliBasis(G*p)) == bin_index(A, p) ⊻ bin_shift(A, G)`.
A generator with `bin_shift == 0` is *protected* — rotations by it never move
terms between bins (communication free).
"""
bin_shift(A::RankMap{N}, G::PauliBasis{N}) where N = bin_index(A, G)

"""
    rand(rng::AbstractRNG, ::Type{RankMap{N}}, r::Int)

Draw a rank map with `r` dense random rows, uniform over all 2N-bit vectors
(restricted to the N-qubit register). Random dense rows give near-perfect
statistical load balance for any population: two distinct Paulis land in the
same bin with probability exactly 2^-r.
"""
function Base.rand(rng::Random.AbstractRNG, ::Type{RankMap{N}}, r::Int) where N
    mask = N == 127 ? typemax(Int128) : (Int128(1) << N) - 1
    rows = [RankRow(rand(rng, Int128) & mask, rand(rng, Int128) & mask) for _ in 1:r]
    return RankMap{N}(rows)
end
Base.rand(::Type{RankMap{N}}, r::Int) where N = rand(Random.default_rng(), RankMap{N}, r)

function Base.show(io::IO, A::RankMap{N}) where N
    print(io, "RankMap{$N}($(nbits(A)) rows, $(nbins(A)) bins)")
end


# ============================================================
# Constrained construction: protected (communication-free) generators
# ============================================================
#
# A generator G is protected by a rank map iff every row has even overlap
# with G's symplectic mask, i.e. every row is orthogonal (plain GF(2) dot
# product over the 2N concatenated bits) to (G.z, G.x). Valid rows form the
# nullspace of the constraint system whose rows are the protected masks.
#
# 2N-bit vectors are RankRows; bit i is z-half bit i for i < N, x-half bit
# i-N otherwise.

@inline _getbit(v::RankRow, i::Int, N::Int) =
    i < N ? Int((v.z >> i) & 1) : Int((v.x >> (i - N)) & 1)
@inline _flipbit(v::RankRow, i::Int, N::Int) =
    i < N ? RankRow(v.z ⊻ (Int128(1) << i), v.x) : RankRow(v.z, v.x ⊻ (Int128(1) << (i - N)))
@inline _xor(v1::RankRow, v2::RankRow) = RankRow(v1.z ⊻ v2.z, v1.x ⊻ v2.x)
_iszero(v::RankRow) = v.z == 0 && v.x == 0

_leadbit(v::RankRow, N::Int) =
    v.z != 0 ? trailing_zeros(v.z) : v.x != 0 ? N + trailing_zeros(v.x) : -1

"""
    protected_row_basis(protected::Vector{PauliBasis{N}}) -> Vector{RankRow}

GF(2) nullspace basis of the evenness constraints imposed by the protected
generators: every returned basis vector (and hence every xor-combination of
them) has even overlap with every protected mask, so any rank map built from
them gives `bin_shift(A, G) == 0` for all protected `G`. Duplicate or
linearly dependent masks cost nothing. The basis has `2N - rank(constraints)`
elements.
"""
function protected_row_basis(protected::Vector{PauliBasis{N}}) where N
    # reduced row echelon form of the constraint matrix, over Int128 bit ops
    pivots = Int[]
    echelon = RankRow[]
    for G in protected
        v = RankRow(G.z, G.x)
        for (p, e) in zip(pivots, echelon)
            _getbit(v, p, N) == 1 && (v = _xor(v, e))
        end
        i = _leadbit(v, N)
        i == -1 && continue        # dependent (or duplicate) mask
        push!(pivots, i)
        push!(echelon, v)
    end
    for j in eachindex(echelon), k in eachindex(echelon)
        if k != j && _getbit(echelon[k], pivots[j], N) == 1
            echelon[k] = _xor(echelon[k], echelon[j])
        end
    end
    # one nullspace basis vector per free bit
    pivset = Set(pivots)
    basis = RankRow[]
    for f in 0:2N-1
        f in pivset && continue
        v = _flipbit(RankRow(Int128(0), Int128(0)), f, N)
        for (p, e) in zip(pivots, echelon)
            _getbit(e, f, N) == 1 && (v = _flipbit(v, p, N))
        end
        push!(basis, v)
    end
    return basis
end

"""
    rand_valid_row(basis::Vector{RankRow}; rng=Random.default_rng())

A uniformly random nonzero element of the span of `basis` (a random xor of a
random subset, redrawn if zero).
"""
function rand_valid_row(basis::Vector{RankRow}; rng::Random.AbstractRNG=Random.default_rng())
    isempty(basis) &&
        error("the constraint nullspace is empty: every row is forced to zero " *
              "(too many independent protected generators)")
    for _ in 1:1000
        v = RankRow(Int128(0), Int128(0))
        for b in basis
            rand(rng, Bool) && (v = _xor(v, b))
        end
        _iszero(v) || return v
    end
    error("failed to draw a nonzero row from the constraint nullspace")
end

# Reduce v against an xor-basis (elements with distinct lead bits); returns
# the reduced vector (zero iff v is in the span). Processing in increasing
# lead-bit order makes a single pass sufficient: an element's support lies at
# or above its own lead, so it can never re-set an already-cleared position.
function _gf2_reduce(v::RankRow, basis::Vector{RankRow}, N::Int)
    for b in sort(basis, by = b -> _leadbit(b, N))
        i = _leadbit(b, N)
        _getbit(v, i, N) == 1 && (v = _xor(v, b))
    end
    return v
end

"""
    RankMap{N}(r::Int; protected=PauliBasis{N}[], rng=Random.default_rng())

A rank map with `r` random rows drawn from the nullspace of the protected
generators' evenness constraints: every protected generator gets
`bin_shift == 0` (its rotations are communication free), while random rows
within the constrained family retain the statistical load-balance
guarantees. Rows are drawn mutually independent over GF(2), so all `2^r`
bins are structurally reachable.
"""
function RankMap{N}(r::Int; protected::Vector{PauliBasis{N}}=PauliBasis{N}[],
                    rng::Random.AbstractRNG=Random.default_rng()) where N
    basis = protected_row_basis(protected)
    length(basis) >= r ||
        error("cannot draw $r independent rows: the constraint nullspace has " *
              "dimension $(length(basis))")
    rows = RankRow[]
    chosen = RankRow[]              # xor-basis of accepted rows
    while length(rows) < r
        v = rand_valid_row(basis; rng)
        red = _gf2_reduce(v, chosen, N)
        _iszero(red) && continue    # dependent on already-chosen rows
        push!(rows, v)
        push!(chosen, red)
    end
    return RankMap{N}(rows)
end


# ============================================================
# Greedy bisection: pick rows by how well they split a real population
# ============================================================

function _draw_candidates(basis::Vector{RankRow}, chosen::Vector{RankRow},
                          ncandidates::Int, rng::Random.AbstractRNG, N::Int)
    cands = RankRow[]
    tries = 0
    while length(cands) < ncandidates
        (tries += 1) > 100 * ncandidates &&
            error("could not draw $ncandidates candidate rows independent of the " *
                  "already-chosen rows")
        v = rand_valid_row(basis; rng)
        _iszero(_gf2_reduce(v, chosen, N)) && continue
        push!(cands, v)
    end
    return cands
end

# counts[b+1, j]: how many terms land in bin b if candidate j becomes row i
# (bins formed by the i-1 already-chosen rows plus the candidate)
function _split_counts(terms::Vector{PauliBasis{N}}, partial::Vector{Int},
                       cands::Vector{RankRow}, i::Int) where N
    counts = zeros(Int, 1 << i, length(cands))
    for (j, c) in enumerate(cands)
        for (t, p) in enumerate(terms)
            counts[(partial[t] | (_row_parity(c, p) << (i - 1))) + 1, j] += 1
        end
    end
    return counts
end

_balance_score(counts::AbstractVector{Int}) = sum(abs2, counts)

"""
    greedy_bisection_rankmap(pop, r::Int; protected=PauliBasis{N}[],
                             ncandidates=32, rng=Random.default_rng())

Build a rank map row by row against an actual population (`pop` may be a
`Vector{PauliBasis{N}}`, a `PauliSum`, or a `BinnedPauliSum`): at each step,
sample `ncandidates` rows from the protected-constraint nullspace
(independent of the rows chosen so far), count how each splits the current
bins, and keep the one minimizing `Σ n_b²`. Cheaper than perfect balance
needs to be, and beats random draws when the population is clustered.
"""
function greedy_bisection_rankmap(terms::Vector{PauliBasis{N}}, r::Int;
                                  protected::Vector{PauliBasis{N}}=PauliBasis{N}[],
                                  ncandidates::Int=32,
                                  rng::Random.AbstractRNG=Random.default_rng()) where N
    basis = protected_row_basis(protected)
    length(basis) >= r ||
        error("cannot build $r independent rows: the constraint nullspace has " *
              "dimension $(length(basis))")
    rows = RankRow[]
    chosen = RankRow[]
    partial = zeros(Int, length(terms))
    for i in 1:r
        cands = _draw_candidates(basis, chosen, ncandidates, rng, N)
        counts = _split_counts(terms, partial, cands, i)
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

greedy_bisection_rankmap(O::PauliSum{N}, r::Int; kw...) where N =
    greedy_bisection_rankmap(collect(keys(O)), r; kw...)
