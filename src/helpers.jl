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

# ---- shared threading helpers (used by threaded PauliSum reductions/rotations) ----

# Split 1:n into k contiguous ranges (trailing ranges may be empty if k > n).
function chunk_ranges(n::Integer, k::Integer)
    sz = cld(n, k)
    return [((c-1)*sz + 1):min(c*sz, n) for c in 1:k]
end

# Number of threads to use for a term-wise reduction/loop over `n` Pauli terms.
# Returns 1 (serial) unless there are enough terms to amortize the thread overhead.
function reduction_nthreads(n::Integer; min_per_thread::Integer=4096)
    nt = Threads.nthreads()
    (nt > 1 && n >= 2*min_per_thread) || return 1
    return min(nt, max(1, cld(n, min_per_thread)))
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

