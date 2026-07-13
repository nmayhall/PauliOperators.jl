"""
An occupation-number vector on `N` qubits. The bitstring `v` is stored as unsigned
integer type `T` (default `uinttype(N)`), so more than 128 qubits are supported.
"""
struct Ket{N,T<:Unsigned}
    v::T
    Ket{N,T}(v::T) where {N,T<:Unsigned} = new{N,T}(v)
end

struct Bra{N,T<:Unsigned}
    v::T
    Bra{N,T}(v::T) where {N,T<:Unsigned} = new{N,T}(v)
end

Ket{N,T}(v::Integer) where {N,T<:Unsigned} = Ket{N,T}(T(v))
Ket{N}(v::Integer) where {N} = Ket{N,uinttype(N)}(v)
Bra{N,T}(v::Integer) where {N,T<:Unsigned} = Bra{N,T}(T(v))
Bra{N}(v::Integer) where {N} = Bra{N,uinttype(N)}(v)

"""
    Ket(vec::Vector{T}) where T<:Union{Bool, Integer}

Create a `Ket` from a vector of 0s and 1s representing qubit occupations.
"""
function Ket(vec::Vector{S}) where S<:Union{Bool, Integer}
    N = length(vec)
    T = uinttype(N)
    v = zero(T)
    for i in 1:N
        if vec[i] == 1
            v |= T(2)^(i-1)
        end
    end
    return Ket{N,T}(v)
end
function Bra(vec::Vector{S}) where S<:Union{Bool, Integer}
    N = length(vec)
    T = uinttype(N)
    v = zero(T)
    for i in 1:N
        if vec[i] == 1
            v |= T(2)^(i-1)
        end
    end
    return Bra{N,T}(v)
end

"""
    Ket(N::Integer, v::Integer)

Create an `N`-qubit `Ket` from the integer `v` (bits beyond `N` are masked off).
"""
function Ket(N::Integer, v::Integer)
    T = uinttype(N)
    return Ket{N,T}(T(v) & _bitmask(T, N))
end
function Bra(N::Integer, v::Integer)
    T = uinttype(N)
    return Bra{N,T}(T(v) & _bitmask(T, N))
end


function Base.size(d::Ket{N}) where N
    return (BigInt(2)^N, 1)
end

function Base.size(d::Bra{N}) where N
    return (1, BigInt(2)^N)
end

Base.adjoint(d::Ket{N,T}) where {N,T} = Bra{N,T}(d.v)
Base.adjoint(d::Bra{N,T}) where {N,T} = Ket{N,T}(d.v)


Base.rand(::Type{Ket{N}}) where N = rand(Ket{N,uinttype(N)})
Base.rand(::Type{Bra{N}}) where N = rand(Bra{N,uinttype(N)})
Base.rand(::Type{Ket{N,T}}) where {N,T<:Unsigned} = Ket{N,T}(rand(T) & _bitmask(T,N))
Base.rand(::Type{Bra{N,T}}) where {N,T<:Unsigned} = Bra{N,T}(rand(T) & _bitmask(T,N))


@inline coeff(d::Ket) = 1 




"""
    Base.show(io::IO, P::Union{Ket, Bra})

Print the ket/bra string representation (e.g., `|010>` or `<010|`).
"""
function Base.show(io::IO, P::Union{Ket,Bra})
    print(io, string(P))
end

function Base.string(p::Ket{N}) where N
    out = [0 for i in 1:N]
    for i in get_on_bits(p.v)
        out[i] = 1
    end
    return "|"*join(out[1:N])*">"
end
function Base.string(p::Bra{N}) where N
    out = [0 for i in 1:N]
    for i in get_on_bits(p.v)
        out[i] = 1
    end
    return "<"*join(out[1:N])*"|"
end


"""
    Base.:+(p::Ket{N}, q::Ket{N}) where N

Add two `Ket`'s together, return a `KetSum`
"""
function Base.:+(p::Ket{N}, q::Ket{N}) where N
    if p == q
        return KetSum{N, ComplexF64}(p => 2) 
    else 
        return KetSum{N, ComplexF64}(p => 1, q => 1)
    end
end
"""
    Base.:+(p::Bra{N}, q::Bra{N}) where N

Add two `Ket`'s together, return a `KetSum`
"""
function Base.:+(p::Bra{N}, q::Bra{N}) where N
    if p == q
        return KetSum{N, ComplexF64}(p' => 2)'
    else 
        return KetSum{N, ComplexF64}(p' => 1, q' => 1)'
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
    T = uinttype(N+M)
    Ket{N+M,T}(T(k1.v) | (T(k2.v) << N))
end
function otimes(k1::Bra{N}, k2::Bra{M}) where {N,M}
    T = uinttype(N+M)
    Bra{N+M,T}(T(k1.v) | (T(k2.v) << N))
end