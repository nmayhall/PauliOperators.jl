"""
An occupation number vector. The bitstring is stored in an unsigned word `W`
sized to `N` by `word_type(N)` (up to 1024 qubits); `Ket{N}(v)` infers it.
"""
struct Ket{N, W<:Unsigned}
    v::W

    function Ket{N,W}(v::W) where {N, W<:Unsigned}
        8 * sizeof(W) >= N || throw(ArgumentError("$W is too narrow for N=$N"))
        return new{N,W}(v)
    end
end

struct Bra{N, W<:Unsigned}
    v::W

    function Bra{N,W}(v::W) where {N, W<:Unsigned}
        8 * sizeof(W) >= N || throw(ArgumentError("$W is too narrow for N=$N"))
        return new{N,W}(v)
    end
end

# Hot paths: W inferred from the argument, no masking (caller invariant).
Ket{N}(v::W) where {N, W<:Unsigned} = Ket{N,W}(v)
Bra{N}(v::W) where {N, W<:Unsigned} = Bra{N,W}(v)

# Convenience paths: any Integer, masked to the low N bits, canonical W.
Ket{N}(v::Integer) where {N} = (W = word_type(N); Ket{N,W}(_to_word(W, N, v)))
Bra{N}(v::Integer) where {N} = (W = word_type(N); Bra{N,W}(_to_word(W, N, v)))
Ket{N,W}(v::Integer) where {N, W<:Unsigned} = Ket{N,W}(_to_word(W, N, v))
Bra{N,W}(v::Integer) where {N, W<:Unsigned} = Bra{N,W}(_to_word(W, N, v))

"""
    Ket(vec::Vector{T}) where T<:Union{Bool, Integer}

Create a `Ket` from a vector of 0s and 1s representing qubit occupations.
"""
function Ket(vec::Vector{T}) where T<:Union{Bool, Integer}
    N = length(vec)
    W = word_type(N)
    v = zero(W)
    for i in 1:N
        if vec[i] == 1
            v |= one(W) << (i-1)
        end
    end
    return Ket{N}(v)
end
function Bra(vec::Vector{T}) where T<:Union{Bool, Integer}
    N = length(vec)
    W = word_type(N)
    v = zero(W)
    for i in 1:N
        if vec[i] == 1
            v |= one(W) << (i-1)
        end
    end
    return Bra{N}(v)
end

"""
    Ket(N::Integer, v::Integer)

Create an `N`-qubit `Ket` from the integer `v` (bits beyond `N` are masked off).
The storage word is always the canonical `word_type(N)`, regardless of the
type of `v` — so `Ket(4, 0b0011)` is a `Ket{4,UInt64}`, not `Ket{4,UInt8}`.
"""
Ket(N::Integer, v::Integer) = (M = Int(N); W = word_type(M); Ket{M,W}(_to_word(W, M, v)))
Bra(N::Integer, v::Integer) = (M = Int(N); W = word_type(M); Bra{M,W}(_to_word(W, M, v)))


function Base.size(d::Ket{N}) where N
    return (BigInt(2)^N, 1)
end

function Base.size(d::Bra{N}) where N
    return (1, BigInt(2)^N)
end

Base.adjoint(d::Ket{N}) where N = Bra{N}(d.v)
Base.adjoint(d::Bra{N}) where N = Ket{N}(d.v)


Base.rand(::Type{Ket{N}}) where N =
    (W = word_type(N); Ket{N,W}(rand(W) & _nbit_mask(W, N)))
Base.rand(::Type{Bra{N}}) where N =
    (W = word_type(N); Bra{N,W}(rand(W) & _nbit_mask(W, N)))
Base.rand(::Type{Ket{N,W}}) where {N, W<:Unsigned} = Ket{N,W}(rand(W) & _nbit_mask(W, N))
Base.rand(::Type{Bra{N,W}}) where {N, W<:Unsigned} = Bra{N,W}(rand(W) & _nbit_mask(W, N))


@inline coeff(d::Ket) = 1 




"""
    Base.show(io::IO, P::Union{Ket, Bra})

Print the ket/bra string representation (e.g., `|010>` or `<010|`).
"""
function Base.show(io::IO, P::Union{Ket,Bra})
    print(io, string(P))
end

function Base.string(p::Ket{N}) where N
    out = [0 for i in 1:8*sizeof(p.v)]
    for i in get_on_bits(p.v)
        out[i] = 1
    end
    return "|"*join(out[1:N])*">"
end
function Base.string(p::Bra{N}) where N
    out = [0 for i in 1:8*sizeof(p.v)]
    for i in get_on_bits(p.v)
        out[i] = 1
    end
    return "<"*join(out[1:N])*"|"
end


"""
    Base.:+(p::Ket{N}, q::Ket{N}) where N

Add two `Ket`'s together, return a `KetSum`
"""
function Base.:+(p::Ket{N,W}, q::Ket{N,W}) where {N,W}
    if p == q
        return KetSum{N, W, ComplexF64}(p => 2)
    else
        return KetSum{N, W, ComplexF64}(p => 1, q => 1)
    end
end
"""
    Base.:+(p::Bra{N}, q::Bra{N}) where N

Add two `Ket`'s together, return a `KetSum`
"""
function Base.:+(p::Bra{N,W}, q::Bra{N,W}) where {N,W}
    if p == q
        return KetSum{N, W, ComplexF64}(p' => 2)'
    else
        return KetSum{N, W, ComplexF64}(p' => 1, q' => 1)'
    end
end



function index(k::Union{Ket{N}, Bra{N}}) where N
    return k.v+1
end

"""
    Base.Vector(k::Union{Ket{N}, Bra{N}}; T=Int64) where N

Create dense vector representation in standard basis 
"""
function Base.Vector(k::Union{Ket{N}, Bra{N}}; T=Int64) where N
    vec = zeros(T,Int128(2)^N)
    vec[index(k)] = T(1) 
    return vec 
end
function Base.Matrix(k::Union{Ket{N}, Bra{N}}; T=Int64) where N
    vec = zeros(T,Int128(2)^N,1)
    vec[index(k),1] = T(1) 
    return vec 
end


function Base.iterate(::Type{Ket{N}}, state = 1) where N
    state > 4^N && return
    return Ket{N}(state-1), state+1 
end

function otimes(k1::Ket{N}, k2::Ket{M}) where {N,M}
    W = word_type(N + M)
    Ket{N+M,W}(W(k1.v) | W(k2.v) << N)
end
function otimes(k1::Bra{N}, k2::Bra{M}) where {N,M}
    W = word_type(N + M)
    Bra{N+M,W}(W(k1.v) | W(k2.v) << N)
end