
DyadSum{N,W,T} = Dict{DyadBasis{N,W}, T}


DyadSum(N::Integer, T::Type) = Dict{DyadBasis{Int(N), word_type(N)}, T}()
DyadSum(N::Integer; T=ComplexF64) = DyadSum(N, T)
DyadSum(d::Dyad{N,W}; T=ComplexF64) where {N,W} = Dict{DyadBasis{N,W}, T}(DyadBasis(d)=>T(coeff(d)))
DyadSum(d::DyadBasis{N,W}; T=ComplexF64) where {N,W} = Dict{DyadBasis{N,W}, T}(DyadBasis(d)=>T(1))


Base.adjoint(d::DyadSum{N,W,T}) where {N,W,T} = Adjoint(d)
Base.parent(d::Adjoint{<:Any, <:DyadSum}) = d.parent

function Base.getindex(ds::DyadSum{N,W,T}, d::Dyad{N}) where {N,W,T} 
    return ds[DyadBasis(d)]
end
function Base.getindex(ds::Adjoint{<:Any,DyadSum{N,W,T}}, d::Dyad{N}) where {N,W,T} 
    return parent(ds)[DyadBasis(d)]'
end


function Base.show(io::IO, ::MIME"text/plain", ps::DyadSum{N,W,T}) where {N,W,T}
    for (key, val) in ps
        @printf(io, " %12.8f +%12.8fi %s\n", real(val), imag(val), key)
    end
end

function Base.show(io::IO, ::MIME"text/plain", ps::Adjoint{<:Any, DyadSum{N,W,T}}) where {N,W,T}
    for (key, val) in ps.parent
        @printf(io, " %12.8f +%12.8fi %s\n", real(val), -imag(val), key')
    end
end

"""
    coeff_clip!(ps::DyadSum{N,W,T}, thresh::Real) where {N,W,T}

Remove Dyad terms with |coefficient| <= `thresh`.
"""
function coeff_clip!(ps::DyadSum{N,W,T}, thresh::Real) where {N,W,T}
    filter!(p->abs(p.second) > thresh, ps)
end

"""
    clip!(ps::DyadSum; thresh=1e-16)

!!! warning "Deprecated"
    Use `coeff_clip!(ps, thresh)` instead.
"""
clip!(ps::DyadSum; thresh=1e-16) = coeff_clip!(ps, thresh)


function Base.Matrix(ds::DyadSum{N, W, T}) where {N,W,T}
    out = zeros(T, Int128(2)^N, Int128(2)^N)
    for (op, coeff) in ds
        out .+= Matrix(op) .* coeff 
    end
    return out
end
function Base.Matrix(ps::Adjoint{<:Any, DyadSum{N, W, T}}) where {N,W,T}
    out = zeros(T, Int128(2)^N, Int128(2)^N)
    for (op, coeff) in ps.parent
        out .+= Matrix(op') .* coeff'
    end
    return out
end


function Base.rand(::Type{DyadSum{N, W, T}}; n_terms=2) where {N,W,T}
    out = DyadSum(N, T)
    for i in 1:n_terms
        p = rand(Dyad{N})
        out[DyadBasis(p)] = coeff(p) * rand(T)
    end
    return out 
end
function Base.rand(::Type{DyadSum{N}}; n_terms=2, T=ComplexF64) where {N}
    out = DyadSum(N, T)
    for i in 1:n_terms
        p = rand(Dyad{N})
        out[DyadBasis(p)] = coeff(p) * rand(T)
    end
    return out 
end


"""
    Base.:*(d1::DyadSum{N,W,T}, d2::DyadSum{N,W,T}) where {N,W,T}

Multiply two `DyadSum`s.
"""
function Base.:*(d1::DyadSum{N,W,T}, d2::DyadSum{N,W,T}) where {N,W,T}
    d3 = DyadSum{N,W,T}()
    for (dyad1, coeff1) in d1
        for (dyad2, coeff2) in d2
            sdyad3 = dyad1*dyad2
            if haskey(d3, DyadBasis(sdyad3)) 
                d3[DyadBasis(sdyad3)] += coeff(sdyad3) * coeff1 * coeff2
            else
                d3[DyadBasis(sdyad3)] = coeff(sdyad3) * coeff1 * coeff2
            end
        end
    end
    return d3
end
function Base.:*(d1::Adjoint{<:Any, DyadSum{N,W,T}}, d2::DyadSum{N,W,T}) where {N,W,T}
    d3 = DyadSum{N,W,T}()
    for (dyad1, coeff1) in d1.parent
        for (dyad2, coeff2) in d2
            sdyad3 = dyad1'*dyad2
            if haskey(d3, DyadBasis(sdyad3)) 
                d3[DyadBasis(sdyad3)] += coeff(sdyad3) * coeff1' * coeff2
            else
                d3[DyadBasis(sdyad3)] = coeff(sdyad3) * coeff1' * coeff2
            end
        end
    end
    return d3
end
function Base.:*(d1::DyadSum{N,W,T}, d2::Adjoint{<:Any, DyadSum{N,W,T}}) where {N,W,T}
    d3 = DyadSum{N,W,T}()
    for (dyad1, coeff1) in d1
        for (dyad2, coeff2) in d2.parent
            sdyad3 = dyad1*dyad2'
            if haskey(d3, DyadBasis(sdyad3)) 
                d3[DyadBasis(sdyad3)] += coeff(sdyad3) * coeff1 * coeff2'
            else
                d3[DyadBasis(sdyad3)] = coeff(sdyad3) * coeff1 * coeff2'
            end
        end
    end
    return d3
end

function Base.:*(d1::Adjoint{<:Any, DyadSum{N,W,T}}, d2::Adjoint{<:Any, DyadSum{N,W,T}}) where {N,W,T}
    d3 = DyadSum{N,W,T}()
    for (dyad1, coeff1) in d1.parent
        for (dyad2, coeff2) in d2.parent
            sdyad3 = dyad1'*dyad2'
            if haskey(d3, DyadBasis(sdyad3)) 
                d3[DyadBasis(sdyad3)] += coeff(sdyad3) * coeff1' * coeff2'
            else
                d3[DyadBasis(sdyad3)] = coeff(sdyad3) * coeff1' * coeff2'
            end
        end
    end
    return d3
end

function Base.:*(ps1::DyadSum{N, W, T}, a::Number) where {N, W, T}
    out = deepcopy(ps1)
    mul!(out, a)
    return out
end
Base.:*(a::Number, ps1::DyadSum{N, W, T}) where {N, W, T} = ps1 * a

function Base.:*(ps1::Adjoint{<:Any, DyadSum{N, W, T}}, a::Number) where {N, W, T}
    return (ps1.parent * a)'
end
Base.:*(a::Number, ps1::Adjoint{<:Any, DyadSum{N, W, T}}) where {N, W, T} = ps1 * a
function Base.getindex(ps::Adjoint{<:Any, DyadSum{N,W,T}}, a::DyadBasis{N}) where {N,W,T} 
    return ps.parent[a]'
end

function LinearAlgebra.ishermitian(d::DyadSum{N, W, T}) where {N,W,T}
    isherm = true
    for (dyad,coeff) in d
        if dyad.ket.v == dyad.bra.v 
            if abs(imag(coeff)) > 1e-16 
                return false
            end
        else
            if haskey(d, dyad') == false
                return false
            else
                if abs(coeff - d[DyadBasis(dyad')]') > 1e-16
                    return false
                end 
            end
        end
    end
    return isherm
end

function Base.sum!(ps1::DyadSum{N}, ps2::DyadSum{N}) where {N}
    mergewith!(+, ps1, ps2)
end

function Base.sum!(ps1::DyadSum{N,W,T}, ps2::Adjoint{<:Any, DyadSum{N,W,T}}) where {N,W,T}
    for (dyad, coeff) in ps2.parent
        if haskey(ps1, dyad')
            ps1[dyad'] += coeff'
        else
            ps1[dyad'] = coeff'
        end
    end
    return ps1
end

"""
    Base.:+(p::DyadBasis{N}, q::DyadBasis{N}) where N

Add two `Dyad`'s together, return a `DyadSum`
"""
function Base.:+(p::Union{Dyad{N,W}, DyadBasis{N,W}}, q::Union{Dyad{N,W}, DyadBasis{N,W}}) where {N,W}
    if DyadBasis(p) == DyadBasis(q)
        return DyadSum{N, W, ComplexF64}(DyadBasis(p) => coeff(p)+coeff(q))
    else
        return DyadSum{N, W, ComplexF64}(DyadBasis(p) => coeff(p), DyadBasis(q) => coeff(q))
    end
end

"""
    Base.:+(ps1::DyadSum{N}, ps2::DyadSum{N}) where {N}

Add two `DyadSum`s.
"""
function Base.:+(ps1::DyadSum{N,W,T}, ps2::DyadSum{N,W,T}) where {N,W,T} 
    out = deepcopy(ps1)
    sum!(out, ps2)
    return out
end
function Base.:+(d1::DyadSum{N,W,T}, d2::Adjoint{<:Any, <:DyadSum{N,W,T}}) where {N,W,T}
    out = deepcopy(d1)
    sum!(out, d2)
    return out
end
Base.:+(d2::Adjoint{<:Any, <:DyadSum{N,W,T}}, d1::DyadSum{N,W,T}) where {N,W,T} = d1 + d2

function Base.:-(ps1::DyadSum)
    out = deepcopy(ps1)
    map!(x->-x, values(out))
    return out 
end

function LinearAlgebra.tr(p::DyadSum{N, W, T}) where {N,W,T}
    tmp = T(0)
    for (dyad, coeff) in p
        tmp += coeff * (dyad.ket.v == dyad.bra.v)
    end
    return tmp
end

function LinearAlgebra.mul!(ps::DyadSum, a::Number)
    map!(x->a*x, values(ps))
    return ps
end


"""
    otimes(p1::DyadSum{N,W,T}, p2::DyadSum{M,T}) where {N,M,T}

Tensor product of two `DyadSum`s, returning a `DyadSum{N+M}`.
"""
function otimes(p1::DyadSum{N,W1,T}, p2::DyadSum{M,W2,T}) where {N,M,W1,W2,T}
    out = DyadSum(N+M, T)
    for (op1,coeff1) in p1
        for (op2,coeff2) in p2
            out[op1 ⊗ op2] = coeff1 * coeff2 
        end
    end
    return out 
end