
KetSum{N, T} = Dict{Ket{N}, T}

KetSum(N::Integer, T::Type) = Dict{Ket{N}, T}()
function KetSum(N; T=Float64)
    return Dict{Ket{N}, T}()
end
    

Base.adjoint(d::KetSum{N,T}) where {N,T} = Adjoint(d)
Base.parent(d::Adjoint{<:Any, <:KetSum}) = d.parent

"""
    Base.show(io::IO, v::KetSum{N,T}) where {N,T}

TBW
"""
function Base.show(io::IO, v::KetSum{N,T}) where {N,T}
    for (ket,coeff) in v
        print(io, string(ket), coeff)
    end
end

"""
    LinearAlgebra.dot(v1::KetSum{N,T}, v2::KetSum{N,TT}) where {N,T,TT}

TBW
"""
function LinearAlgebra.dot(v1::KetSum{N,T}, v2::KetSum{N,TT}) where {N,T,TT}
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
    scale!(v1::KetSum{N,T}, a::Number) where {N,T}

TBW
"""
function scale!(v1::KetSum{N,T}, a::Number) where {N,T}
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
    Base.Vector(k::KetSum{N,T}) where {N,T}

TBW
"""
function Base.Vector(k::KetSum{N,T}) where {N,T}
    vec = zeros(T,Int128(2)^N)
    for (k,coeff) in k
        vec[index(k)] = T(coeff) 
    end
    return vec 
end


function Base.Vector(k::Adjoint{<:Any, KetSum{N,T}}) where {N,T}
    vec = zeros(T,Int128(2)^N)
    for (k,coeff) in k.parent
        vec[index(k)] = T(coeff') 
    end
    return vec 
end

"""
    otimes(p1::KetSum{N,T}, p2::KetSum{M,T}) where {N,M,T}

TBW
"""
function otimes(p1::KetSum{N,T}, p2::KetSum{M,T}) where {N,M,T}
    out = KetSum(N+M, T)
    for (op1,coeff1) in p1
        for (op2,coeff2) in p2
            out[op1 ⊗ op2] = coeff1 * coeff2 
        end
    end
    return out 
end


function Base.rand(::Type{KetSum{N, T}}; n_terms=2) where {N,T}
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