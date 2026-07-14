"""
    get_on_bits(x::Integer)

Return the (1-based) positions of the set bits of `x`, in increasing order.
Used to translate `z`/`x` bitstrings into site indices, e.g. for display.
"""
function get_on_bits(x::T) where T<:Integer
    N = count_ones(x)
    inds = Vector{Int}(undef, N)
    if N == 0
        return inds
    end

    count = 1
    for i in 1:8*sizeof(x)
        if x >> (i-1) & 1 == 1
            inds[count] = i
            count += 1
        end
        count <= N || break
    end
    return inds
end

# Branchless-suffix-parity Majorana weight on packed words; the word-level
# kernel behind `majorana_weight` (see clip.jl for the derivation). The
# shift cascade covers 8*sizeof(W) bits, so it is correct at every word
# width including the BitIntegers.jl types.
@inline function _majorana_weight_bits(z::W, x::W) where {W<:Unsigned}
    zonly = z & ~x
    S = x
    shift = 1
    while shift < 8 * sizeof(W)
        S ⊻= S >> shift
        shift <<= 1
    end
    ctrl = ~(S ⊻ x)
    return count_ones(x) + 2 * count_ones(zonly & ctrl) +
           2 * count_ones(~(z | x) & ~ctrl)
end

