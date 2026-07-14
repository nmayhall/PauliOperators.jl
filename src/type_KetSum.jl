
KetSum{N, W, T} = Dict{Ket{N,W}, T}

KetSum(N::Integer, T::Type) = Dict{Ket{Int(N), word_type(N)}, T}()
function KetSum(N; T=Float64)
    return KetSum(N, T)
end
function KetSum(k::Ket{N,W}; T=Float64) where {N,W}
    out = Dict{Ket{N,W}, T}()
    out[k] = 1
    return out
end
    

Base.adjoint(d::KetSum{N,W,T}) where {N,W,T} = Adjoint(d)
Base.parent(d::Adjoint{<:Any, <:KetSum}) = d.parent

"""
    Base.show(io::IO, v::KetSum{N,W,T}) where {N,W,T}

Print each ket and its coefficient.
"""
function Base.show(io::IO, v::KetSum{N,W,T}) where {N,W,T}
    for (ket,coeff) in v
        print(io, string(ket), coeff)
    end
end

"""
    LinearAlgebra.dot(v1::KetSum{N,W,T}, v2::KetSum{N,W,TT}) where {N,W,T,TT}

Compute the inner product `v1' * v2`, iterating over the shorter dictionary.
"""
function LinearAlgebra.dot(v1::KetSum{N,W,T}, v2::KetSum{N,W,TT}) where {N,W,T,TT}
    out = 0.0
    if length(v1) < length(v2)
        for (ket,coeff) in v1
            out += adjoint(coeff) * get(v2, ket, 0.0)
        end
    else
        for (ket,coeff) in v2
            out += coeff * adjoint(get(v1, ket, 0.0))
        end
    end
    return out
end

"""
    scale!(v1::KetSum{N,W,T}, a::Number) where {N,W,T}

Scale all coefficients in `v1` by `a` in-place.
"""
function scale!(v1::KetSum{N,W,T}, a::Number) where {N,W,T}
    map!(x->x*a, values(v1))
end
function Base.:*(v::KetSum, a::Number)
    out = deepcopy(v)
    scale!(out,a)
    return out
end
Base.:*(a::Number, v::KetSum) = v*a
Base.:/(v::KetSum, a::Number) = v*(1/a)



"""
    Base.Vector(k::KetSum{N,W,T}) where {N,W,T}

Create a dense vector representation of the `KetSum` in the standard computational basis.
"""
function Base.Vector(k::KetSum{N,W,T}) where {N,W,T}
    vec = zeros(T,Int128(2)^N)
    for (k,coeff) in k
        vec[index(k)] = T(coeff) 
    end
    return vec 
end


function Base.Vector(k::Adjoint{<:Any, KetSum{N,W,T}}) where {N,W,T}
    vec = zeros(T,Int128(2)^N)
    for (k,coeff) in k.parent
        vec[index(k)] = T(coeff') 
    end
    return vec 
end

"""
    otimes(p1::KetSum{N,W,T}, p2::KetSum{M,T}) where {N,M,T}

Tensor product of two `KetSum`s, returning a `KetSum{N+M}`.
"""
function otimes(p1::KetSum{N,W1,T}, p2::KetSum{M,W2,T}) where {N,M,W1,W2,T}
    out = KetSum(N+M, T)
    for (op1,coeff1) in p1
        for (op2,coeff2) in p2
            out[op1 ⊗ op2] = coeff1 * coeff2 
        end
    end
    return out 
end


function Base.rand(::Type{KetSum{N, W, T}}; n_terms=2) where {N,W,T}
    out = KetSum(N, T)
    for i in 1:n_terms
        p = rand(Ket{N})
        out[p] = coeff(p) * rand(T)
    end
    return out 
end
function Base.rand(::Type{KetSum{N}}; n_terms=2, T=ComplexF64) where {N}
    out = KetSum(N, T)
    for i in 1:n_terms
        p = rand(Ket{N})
        out[p] = coeff(p) * rand(T)
    end
    return out 
end