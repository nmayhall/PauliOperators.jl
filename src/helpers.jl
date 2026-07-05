"""
    uinttype(N::Integer)

Smallest unsigned integer type able to hold the `N`-bit symplectic `z`/`x`
bitstrings of an `N`-qubit Pauli. Native `UInt8..UInt128` up to 128 bits, then
`BitIntegers` fixed-width unsigned integers (`UInt256`, `UInt512`, `UInt1024`, ...)
beyond — so Paulis on more than 128 qubits are supported (e.g. a 10x10x10 lattice
uses `UInt1024`). Widths above 1024 are defined on demand.
"""
function uinttype(N::Integer)
    N >= 0 || throw(DomainError(N, "N must be non-negative"))
    N <= 8 && return UInt8
    bits = nextpow(2, N)
    bits <= 128 && return getfield(Base, Symbol("UInt", bits))
    bits > 1024 && _define_wide_uint(bits)
    return getfield(BitIntegers, Symbol("UInt", bits))
end

function _define_wide_uint(bits::Integer)
    @eval BitIntegers begin
        BitIntegers.@define_integers $bits
    end
    return nothing
end

function get_on_bits(x::T) where T<:Integer
    N = count_ones(x)
    inds = Vector{Int}(undef, N)
    if N == 0
        return inds
    end

    count = 1
    for i in 1:length(bitstring(x))
        if x >> (i-1) & 1 == 1
            inds[count] = i
            count += 1
        end
        count <= N || break
    end
    return inds
end

