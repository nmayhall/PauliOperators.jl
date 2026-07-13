
"""
    z::T
    x::T

A positive, Hermitian Pauli, used as a basis for more general `Pauli`'s (which can have a complex phase).
These are primarily used to provide a basis for linear combinations of Paulis, e.g., `PauliSum`'s.

    PauliBasis{N}(z,x)  =  i^θs ⋅ z₁...|x₁...
                            =  P₁⊗...⊗Pₙ

The bitstrings `z`, `x` are stored as an unsigned integer of type `T`. `T` defaults to
`uinttype(N)`, the smallest unsigned type able to hold `N` bits, which uses `BitIntegers`
fixed-width integers for `N > 128` (so e.g. a 10x10x10 lattice is `PauliBasis{1000,UInt1024}`).

Phase definitions:
- `symplectic_phase`: `θs` - phase needed to cancel the phase arising from the ZX factorized form: `θs = θ-θg`
"""
struct PauliBasis{N,T<:Unsigned}
    z::T
    x::T

    PauliBasis{N,T}(z::T, x::T) where {N,T<:Unsigned} = new{N,T}(z, x)
end

PauliBasis{N,T}(z::Integer, x::Integer) where {N,T<:Unsigned} = PauliBasis{N,T}(T(z), T(x))
PauliBasis{N}(z::Integer, x::Integer) where {N} = PauliBasis{N,uinttype(N)}(z, x)

LinearAlgebra.ishermitian(p::PauliBasis) = true
coeff(p::PauliBasis) = 1

@inline symplectic_phase(p::PauliBasis) = (4-count_ones(p.z & p.x)%4)%4

function PauliBasis(str::String)
    for i in str
        i in ['I', 'Z', 'X', 'Y'] || error("Bad string: ", str)
    end

    N = length(str)
    T = uinttype(N)
    x = zero(T)
    z = zero(T)
    two = T(2)
    one_ = T(1)

    for (i0, i) in enumerate(str)
        idx = T(i0)
        if i in ['X', 'Y']
            x |= two^(idx-one_)
        end
        if i in ['Z', 'Y']
            z |= two^(idx-one_)
        end
    end
    return PauliBasis{N,T}(z, x)
end



"""
    Base.Matrix(p::PauliBasis{N}) where N

Build dense matrix representation in standard basis
"""
function Base.Matrix(p::PauliBasis{N}) where N
    mat = ones(Int8,1,1)
    str = string(p)
    X = [0 1; 1 0]
    Y = [0 -1im; 1im 0]
    Z = [1 0; 0 -1]
    I = [1 0; 0 1]
    for i in 1:N
        if str[i] == 'X'
            mat = kron(X,mat)
        elseif str[i] == 'Y'
            mat = kron(Y,mat)
        elseif str[i] == 'Z'
            mat = kron(Z,mat)
        elseif str[i] == 'I'
            mat = kron(I,mat)
        else
            throw(ErrorException)
        end
    end

    return mat
end

PauliBasis(p::PauliBasis) = p

"""
    Base.string(p::PauliBasis{N}) where N

Return a string representation. Y sites are displayed as `Y`.
"""
function Base.string(p::PauliBasis{N}) where N
    yloc = get_on_bits(p.x & p.z)
    Xloc = get_on_bits(p.x & ~p.z)
    Zloc = get_on_bits(p.z & ~p.x)
    out = ["I" for i in 1:N]

    for i in Xloc
        out[i] = "X"
    end
    for i in yloc
        out[i] = "Y"
    end
    for i in Zloc
        out[i] = "Z"
    end
    return join(out)
end

function Base.rand(::Type{PauliBasis{N}}) where N
    return rand(PauliBasis{N,uinttype(N)})
end
function Base.rand(::Type{PauliBasis{N,T}}) where {N,T<:Unsigned}
    mask = _bitmask(T, N)
    return PauliBasis{N,T}(rand(T) & mask, rand(T) & mask)
end

# Lowest-N-bits mask for an unsigned type T (handles N == bitwidth(T)).
@inline function _bitmask(::Type{T}, N::Integer) where {T<:Unsigned}
    N >= sizeof(T) * 8 && return ~zero(T)
    return (one(T) << N) - one(T)
end


Base.show(io::IO, p::PauliBasis{N}) where N = print(io, string(p))

function otimes(p1::PauliBasis{N}, p2::PauliBasis{M}) where {N,M}
    T = uinttype(N+M)
    z = T(p1.z) | (T(p2.z) << N)
    x = T(p1.x) | (T(p2.x) << N)
    PauliBasis{N+M,T}(z, x)
end

Base.:*(p1::PauliBasis, p2::PauliBasis) = Pauli(p1) * Pauli(p2)
Base.:*(p1::PauliBasis{N}, a::Number) where N = Pauli(p1)*a
Base.:*(a::Number, p1::PauliBasis{N}) where N = Pauli(p1)*a

Base.adjoint(p::PauliBasis) = p

function Base.iterate(::Type{PauliBasis{N}}, state = 1) where N
    state > 4^N && return
    next = CartesianIndices((2^N,2^N))[state]
    return PauliBasis{N}(next[1]-1, next[2]-1), state+1
end

@inline commute(p1::PauliBasis, p2::PauliBasis) = iseven(count_ones(p1.x & p2.z) - count_ones(p1.z & p2.x))
