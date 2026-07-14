
"""
    z::W
    x::W

A positive, Hermitian Pauli, used as a basis for more general `Pauli`'s (which can have a complex phase).
These are primarily used to provide a basis for linear combinations of Paulis, e.g., `PauliSum`'s.

    PauliBasis{N}(z,x)  =  i^θs ⋅ z₁...|x₁...
                            =  P₁⊗...⊗Pₙ

The bitstrings are stored in an unsigned word `W` sized to `N` by
`word_type(N)` (`UInt64` up to 64 qubits, ..., `UInt1024` up to 1024).
`W` never needs to be written explicitly: `PauliBasis{N}(z, x)` infers it.

Phase definitions:
- `symplectic_phase`: `θs` - phase needed to cancel the phase arising from the ZX factorized form: `θs = θ-θg`
"""
struct PauliBasis{N, W<:Unsigned}
    z::W
    x::W

    function PauliBasis{N,W}(z::W, x::W) where {N, W<:Unsigned}
        8 * sizeof(W) >= N || throw(ArgumentError("$W is too narrow for N=$N"))
        return new{N,W}(z, x)
    end
end

# Hot path: W inferred from the arguments, no masking (caller invariant).
PauliBasis{N}(z::W, x::W) where {N, W<:Unsigned} = PauliBasis{N,W}(z, x)

# Convenience paths: any Integer (negative values are two's-complement
# reinterpreted), masked to the low N bits, canonical W.
PauliBasis{N}(z::Integer, x::Integer) where {N} =
    (W = word_type(N); PauliBasis{N,W}(_to_word(W, N, z), _to_word(W, N, x)))
PauliBasis{N,W}(z::Integer, x::Integer) where {N, W<:Unsigned} =
    PauliBasis{N,W}(_to_word(W, N, z), _to_word(W, N, x))

LinearAlgebra.ishermitian(p::PauliBasis) = true
coeff(p::PauliBasis) = 1

"""
    symplectic_phase(p::Union{Pauli{N}, PauliBasis{N}})

The power of `i` needed to recover the Hermitian Pauli string from the bare
ZX bitstring form: `P = i^θs ⋅ (z|x)`, with `θs = (-n_Y) mod 4` where `n_Y`
is the number of Y sites (`count_ones(z & x)`). Arises because each Y site
is stored as `ZX = iY`.
"""
@inline symplectic_phase(p::PauliBasis) = (4-count_ones(p.z & p.x)%4)%4

function PauliBasis(str::String)
    for i in str
        i in ['I', 'Z', 'X', 'Y'] || error("Bad string: ", str)
    end

    N = length(str)
    W = word_type(N)
    x = zero(W)
    z = zero(W)
    idx = 0

    for i in str
        if i in ['X', 'Y']
            x |= one(W) << idx
        end
        if i in ['Z', 'Y']
            z |= one(W) << idx
        end
        idx += 1
    end
    return PauliBasis{N}(z, x)
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
    W = word_type(N)
    m = _nbit_mask(W, N)
    return PauliBasis{N,W}(rand(W) & m, rand(W) & m)
end
Base.rand(::Type{PauliBasis{N,W}}) where {N, W<:Unsigned} =
    (m = _nbit_mask(W, N); PauliBasis{N,W}(rand(W) & m, rand(W) & m))


Base.show(io::IO, p::PauliBasis{N}) where N = print(io, string(p))

function otimes(p1::PauliBasis{N}, p2::PauliBasis{M}) where {N,M}
    W = word_type(N + M)
    PauliBasis{N+M,W}(W(p1.z) | W(p2.z) << N, W(p1.x) | W(p2.x) << N)
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
 
"""
    commute(p1::PauliBasis, p2::PauliBasis)

Return `true` if the two Pauli strings commute, via the symplectic parity
test `popcount(x₁ & z₂) ≡ popcount(z₁ & x₂) (mod 2)` — no product is formed.
"""
@inline commute(p1::PauliBasis, p2::PauliBasis) = iseven(count_ones(p1.x & p2.z) - count_ones(p1.z & p2.x))