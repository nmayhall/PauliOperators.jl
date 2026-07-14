"""
    Pauli{N} 

is our basic type for representing Pauli operators acting on `N`.
Assume we want to represent a Pauli string of the following form:

    σ1 ⊗ σ2 ⊗ σ3 ⊗ ⋯ ⊗ σN,

where, `σ ∈ {X, Y, Z, I}`.
To do this efficiently, we use the symplectic representation of the Pauli group, 
where we factor each Pauli into a product of X and Z operators: 
    
    σ = i^(3*(z+x)%2) Zᶻ Xˣ,
    
with z,x ∈ {0,1}. The phase factor comes from the fact that `Z*X = iY`.
In this representation, any tensor product of Pauli's is represented as two binary strings, one for x and one for z, along with the associated phase accumulated from each site.
The format is as follows: 
    
    i^θ   Z^z₁ ⋅ X^x₁ ⊗ Z^z₂ ⋅ X^x₂ ⊗ ⋯ ⊗ Z^zₙ ⋅ X^xₙ  
    
Products of operators simply concatonate the left and right strings separately. For example, 
To create a Y operator, bits in the same locations in `z` and `x` should be on. 
    
    XYZIy = 11001|01101     where y = iY

Since we get a factor of `i` each time we create a Y operator, we need to keep track of this to cancel the  phase `θs`, arising from the ZX factorization.
    
    P₁⊗...⊗Pₙ = i^θs ⋅ z₁...|x₁...  where Pᵢ ∈ {I,X,Y,Z}.

similarly, 

    z₁...|x₁... = i^-θs ⋅ P₁⊗...⊗Pₙ 

We use `θs` to denote the phase needed to make the Pauli operator Hermitian and positive, and we refer to this as the `symplectic_phase`, since it arises solely from the symplectic representation of the Pauli.
However, this is not the only phase we need to worry about. Since various phases accumulate during Pauli multiplication, we allow a given `Pauli` to have an arbitrary global phase, `θg`, so that the `Pauli` type can be closed under multiplication. As such, our `Pauli` phases are defined according to the following:

    Pauli{N}(s,z,x)  =  s ⋅ z₁...|x₁... 
                     =  s ⋅ i^-θs ⋅ P₁⊗...⊗Pₙ
                     =  coeff ⋅ P₁⊗...⊗Pₙ

    PauliBasis{N}(z,x)  =  i^θs ⋅ z₁...|x₁... 
                            =  P₁⊗...⊗Pₙ

Phase definitions:
- `symplectic_phase`: `θs` - phase needed to cancel the phase arising from the ZX factorized form: `θs = θ-θg`

Since we need to keep track of a phase for a Pauli, we might as well let it become a general scalar value for broader use. As such, `Pauli.s` is a arbitrary complex number.
"""
struct Pauli{N, W<:Unsigned}
    s::ComplexF64
    z::W
    x::W

    function Pauli{N,W}(s::Number, z::W, x::W) where {N, W<:Unsigned}
        8 * sizeof(W) >= N || throw(ArgumentError("$W is too narrow for N=$N"))
        return new{N,W}(ComplexF64(s), z, x)
    end
end

# Hot path: W inferred from the arguments, no masking (caller invariant).
Pauli{N}(s::Number, z::W, x::W) where {N, W<:Unsigned} = Pauli{N,W}(s, z, x)

# Convenience paths: any Integer, masked to the low N bits, canonical W.
Pauli{N}(s::Number, z::Integer, x::Integer) where {N} =
    (W = word_type(N); Pauli{N,W}(s, _to_word(W, N, z), _to_word(W, N, x)))
Pauli{N,W}(s::Number, z::Integer, x::Integer) where {N, W<:Unsigned} =
    Pauli{N,W}(s, _to_word(W, N, z), _to_word(W, N, x))

PauliTypes{N,W} = Union{Pauli{N,W}, PauliBasis{N,W}}

"""
    coeff(p::Pauli)

Return the coefficient from the product of the scalar times the inverse symplectic_phase
"""
@inline coeff(p::Pauli) = p.s * 1im^inv_symplectic_phase(p)
@inline inv_symplectic_phase(p::Pauli) = (4-symplectic_phase(p)%4)
@inline symplectic_phase(p::Pauli) = (4-count_ones(p.z & p.x)%4)%4

function Pauli(p::PauliBasis{N}) where N
    return Pauli{N}(1im^symplectic_phase(p), p.z, p.x)
end

"""
    Pauli(z::Integer, x::Integer, N)

Construct a `Pauli{N}` from integer bitstrings `z` and `x` with scalar `s=1`.
"""
function Pauli(z::I, x::I, N) where I<:Integer
    W = word_type(N)
    m = _nbit_mask(W, N)
    (z >= 0 && (z % W) & ~m == zero(W)) || throw(DimensionMismatch)
    (x >= 0 && (x % W) & ~m == zero(W)) || throw(DimensionMismatch)
    return Pauli{N}(1, z, x)
end


"""
    Pauli(str::String)

Create a `Pauli` from a string, e.g., 

    a = Pauli("XXYZIZ")

This is convenient for manual manipulations, but is not type-stable so will be slow.
"""
function Pauli(str::String)
    for i in str
        i in ['I', 'Z', 'X', 'Y'] || error("Bad string: ", str)
    end

    N = length(str)
    W = word_type(N)
    x = zero(W)
    z = zero(W)
    ny = 0
    idx = 0

    for i in str
        if i in ['X', 'Y']
            x |= one(W) << idx
            if i == 'Y'
                ny += 1
            end
        end
        if i in ['Z', 'Y']
            z |= one(W) << idx
        end
        idx += 1
    end
    θ = 4-ny%4
    return Pauli{N}(1im^θ, z, x)
end




"""
    Pauli(N::Integer; X=[], Y=[], Z=[])

constructor for creating PauliBoolVec by specifying the qubits where each X, Y, and Z gates exist 
"""
function Pauli(N::Integer; X=[], Y=[], Z=[])
    for i in X
        i ∉ Y || throw(DimensionMismatch)
        i ∉ Z || throw(DimensionMismatch)
    end
    for i in Y
        i ∉ Z || throw(DimensionMismatch)
    end
    
    str = ["I" for i in 1:N]
    for i in X
        str[i] = "X"
    end
    for i in Y
        str[i] = "Y"
    end
    for i in Z
        str[i] = "Z"
    end
   
    return Pauli(join(str))
end   




"""
    Base.string(p::Pauli{N}) where N

Display, y = iY
"""
function Base.string(p::Pauli{N}) where N
    yloc = get_on_bits(p.x & p.z)
    Xloc = get_on_bits(p.x & ~p.z)
    Zloc = get_on_bits(p.z & ~p.x)
    out = ["I" for i in 1:N]

    for i in Xloc
        out[i] = "X"
    end
    for i in yloc
        out[i] = "y"
    end
    for i in Zloc
        out[i] = "Z"
    end
    return join(out)
end

function Base.show(io::IO, p::Pauli{N}) where N
    @printf(io, "% .4f % .4fim | %s", real(p.s), imag(p.s), string(p))
end


"""
    rand(Pauli{N})

Generate a random `Pauli{N}` with random `z`, `x` bitstrings and a random complex scalar.
"""
function Base.rand(::Type{Pauli{N}}) where N
    W = word_type(N)
    m = _nbit_mask(W, N)
    return Pauli{N,W}(randn(ComplexF64), rand(W) & m, rand(W) & m)
end
Base.rand(::Type{Pauli{N,W}}) where {N, W<:Unsigned} =
    (m = _nbit_mask(W, N); Pauli{N,W}(randn(ComplexF64), rand(W) & m, rand(W) & m))


function nY(p::Pauli)
    return count_ones(p.x & p.z)
end

"""
    ishermitian(p::Pauli; thresh=1e-16)

Return `true` if the coefficient of `p` is real (within `thresh`).
"""
function LinearAlgebra.ishermitian(p::Pauli; thresh=1e-16)
    return abs(imag(coeff(p)))<thresh
end


"""
    Base.Matrix(p::Pauli{N}) where N

Build dense matrix representation in standard basis
"""
Base.Matrix(p::Pauli) = Matrix(PauliBasis(p)) * coeff(p)


"""
    Base.:-(p::Pauli{N}) where {N}

Negate the scalar of `p`.
"""
function Base.:-(p::Pauli{N}) where {N}
    return Pauli{N}(-p.s, p.z, p.x) 
end


"""
    Base.adjoint(p::Pauli)


    Pauli{N}(s,z,x)  =  s ⋅ z₁...|x₁... 
                     =  s ⋅ i^-θs ⋅ P₁⊗...⊗Pₙ
                     =  coeff ⋅ P₁⊗...⊗Pₙ

Since the PauliBasis is Hermitian, we have that
    Pauli' = coeff' ⋅ P₁⊗...⊗Pₙ
"""
Base.adjoint(p::Pauli{N}) where N = Pauli{N}(coeff(p)'*1im^symplectic_phase(p), p.z, p.x)

function LinearAlgebra.tr(p::Union{Pauli{N}, PauliBasis{N}}) where N
    return coeff(p) * ((p.z == 0) && (p.x == 0)) * 2^N
end

"""
    Base.:*(p1::Pauli{N}, p2::Pauli{N}) where {N}

Multiply two `Pauli`'s together
"""
function Base.:*(p1::Pauli{N}, p2::Pauli{N}) where {N}
    x = p1.x ⊻ p2.x
    z = p1.z ⊻ p2.z
    s = p1.s * p2.s * 1im^(2*count_ones(p1.x & p2.z) % 4)
    return Pauli{N}(s, z, x)
end

Base.:*(p::Pauli{N}, s::Number) where N = Pauli{N}(p.s * s, p.z, p.x)
Base.:*(s::Number, p::Pauli{N}) where N = p*s 

"""
    Base.:+(p::Pauli{N}, q::Pauli{N}) where N

Add two `Pauli`'s together, return a `PauliSum`
"""
function Base.:+(p::PauliTypes{N,W}, q::PauliTypes{N,W}) where {N,W}
    if PauliBasis(p) == PauliBasis(q)
        return PauliSum{N, W, ComplexF64}(PauliBasis(p) => coeff(p)+coeff(q))
    else
        return PauliSum{N, W, ComplexF64}(PauliBasis(p) => coeff(p), PauliBasis(q) => coeff(q))
    end
end

"""
    otimes(p1::Pauli{N}, p2::Pauli{M}) where {N,M}

Tensor product of two Paulis, returning a `Pauli{N+M}`.
"""
function otimes(p1::Pauli{N}, p2::Pauli{M}) where {N,M}
    W = word_type(N + M)
    Pauli{N+M,W}(p1.s * p2.s, W(p1.z) | W(p2.z) << N, W(p1.x) | W(p2.x) << N)
end

"""
    osum(p1::Pauli{N}, p2::Pauli{M}) where {N,M}

Returns the direct sum of two Paulis
"""
function osum(p1::Pauli{N}, p2::Pauli{M}) where {N,M}
    return p1 ⊗ Pauli(M) + Pauli(N) ⊗ p2 
end

function Base.iterate(::Type{Pauli{N}}, state = 1) where N
    state > 4^N && return
    next = CartesianIndices((2^N,2^N))[state]
    bp = PauliBasis{N}(next[1]-1, next[2]-1)
    return Pauli(bp), state+1 
end