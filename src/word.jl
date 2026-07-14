# ============================================================
# Storage-word selection for bitstring-carrying types.
#
# Every Pauli/Ket bitstring is stored in a single unsigned machine word W,
# sized to the qubit count: UInt64/UInt128 natively, then BitIntegers.jl
# primitive wide types up to 1024 qubits. All kernels are generic over
# W<:Unsigned (⊻, &, <<, count_ones work natively at every width), so hot
# paths never consult `word_type` — W is carried in the type parameters and
# inferred from arguments by dispatch. `word_type` is only called when
# constructing from scratch (strings, rand, empty sums, ...).
# ============================================================

"""
    word_type(N)

The canonical storage word for an `N`-qubit register: the narrowest
supported `Unsigned` type with at least `N` bits. `UInt64` and `UInt128`
natively, `UInt256`/`UInt512`/`UInt1024` via BitIntegers.jl. Throws an
`ArgumentError` beyond 1024 qubits.
"""
function word_type(N::Integer)
    N > 0     || throw(ArgumentError("N must be positive, got $N"))
    N <= 64   && return UInt64
    N <= 128  && return UInt128
    N <= 256  && return UInt256
    N <= 512  && return UInt512
    N <= 1024 && return UInt1024
    throw(ArgumentError("N = $N exceeds the 1024-qubit limit (extend word_type)"))
end

# Historical name used by the SparsePauliVector internals and tests.
const _word_type = word_type

# n-bit all-ones mask; exact at n == 0 and n == 8*sizeof(W).
@inline _nbit_mask(::Type{W}, n::Integer) where {W<:Unsigned} =
    typemax(W) >> (8 * sizeof(W) - n)

# Reinterpret any Integer into W (two's complement, so Int128(-1) means
# "all bits on") and mask to the low N bits.
@inline _to_word(::Type{W}, N::Integer, v::Integer) where {W<:Unsigned} =
    (v % W) & _nbit_mask(W, N)
