"""
    PauliSum{N, W, T} = Dict{PauliBasis{N,W},T}

A collection of `Pauli`s, joined by addition.
This uses a `Dict` to store them, however, the specific use cases should probably dictate the container type,
so this will probably be removed.

`W` is the unsigned storage word for the bitstrings (see [`word_type`](@ref));
value-level constructors (`PauliSum(N)`) pick it automatically.
"""
PauliSum{N, W, T} = Dict{PauliBasis{N,W},T}

PauliSum(N, T) = Dict{PauliBasis{Int(N), word_type(N)}, T}()
PauliSum(N::Integer) = PauliSum(N, ComplexF64)

function Base.rand(::Type{PauliSum{N, W, T}}; n_paulis=2) where {N,W,T}
    out = Dict{PauliBasis{N,W}, T}()
    for i in 1:n_paulis
        p = rand(Pauli{N,W})
        out[PauliBasis(p)] = coeff(p) * rand(T)
    end
    return out
end
function Base.rand(::Type{PauliSum{N}}; n_paulis=2, T=ComplexF64) where {N}
    out = PauliSum(N, T)
    for i in 1:n_paulis
        p = rand(Pauli{N})
        out[PauliBasis(p)] = coeff(p) * rand(T)
    end
    return out
end

function LinearAlgebra.ishermitian(p::PauliSum{N, W, T}) where {N,W,T}
    isherm = true
    for coeff in values(p)
        isherm = isherm && isapprox(imag(coeff), 0, atol=1e-16)
    end
    return isherm
end

function Base.show(io::IO, ::MIME"text/plain", ps::PauliSum{N,W,T}) where {N,W,T}
    for (key, val) in ps
        @printf(io, " %12.8f +%12.8fi %s\n", real(val), imag(val), key)
    end
end

function Base.show(io::IO, ::MIME"text/plain", ps::Adjoint{<:Any, PauliSum{N,W,T}}) where {N,W,T}
    for (key, val) in ps.parent
        @printf(io, " %12.8f +%12.8fi %s\n", real(val), -imag(val), key)
    end
end

"""
    Base.Matrix(ps::PauliSum{N}; T=ComplexF64) where N

Create a dense Matrix of type `T` in the standard basis
"""
function Base.Matrix(ps::PauliSum{N, W, T}) where {N,W,T}
    out = zeros(T, Int128(2)^N, Int128(2)^N)
    for (op, coeff) in ps
        out .+= Matrix(op) .* coeff
    end
    return out
end

function LinearAlgebra.tr(p::PauliSum{N, W, T}) where {N,W,T}
    return get(p, PauliBasis{N,W}(zero(W), zero(W)), 0)*2^N
end



"""
    Base.:-(ps1::PauliSum, ps2::PauliSum)

Subtract two `PauliSum`s.
"""
function Base.:-(ps1::PauliSum)
    out = deepcopy(ps1)
    map!(x->-x, values(out))
    return out
end

Base.adjoint(d::PauliSum{N,W,T}) where {N,W,T} = Adjoint(d)
Base.parent(d::Adjoint{<:Any, <:PauliSum}) = d.parent

function Base.Matrix(ps::Adjoint{<:Any, PauliSum{N, W, T}}) where {N,W,T}
    out = zeros(T, Int128(2)^N, Int128(2)^N)
    for (op, coeff) in ps.parent
        out .+= Matrix(op) .* adjoint(coeff)
    end
    return out
end

function Base.size(d::PauliSum{N}) where N
    return (BigInt(2)^N, BigInt(2)^N)
end


function LinearAlgebra.mul!(ps::PauliSum, a::Number)
    map!(x->a*x, values(ps))
    return ps
end


"""
    Base.:*(ps1::PauliSum{N}, ps2::PauliSum{N}) where {N}

Multiply two `PauliSum`s.
"""
function Base.:*(ps1::PauliSum{N, W, T}, ps2::PauliSum{N, W, T}) where {N, W, T}
    out = PauliSum(N, T)
    for (op1, coeff1) in ps1
        for (op2, coeff2) in ps2
            prod = Pauli(op1) * Pauli(op2)
            c = coeff(prod)
            prod = PauliBasis(prod)
            if haskey(out, prod)
                out[prod] += c * coeff1 * coeff2
            else
                out[prod] = c * coeff1 * coeff2
            end
        end
    end
    return out
end
"""
    Base.:*(ps1::Adjoint{<:Any, PauliSum{N, W, T}}, ps2::PauliSum{N, W, T}) where {N, W, T}

Multiply two `PauliSum`s.
"""
function Base.:*(ps1::Adjoint{<:Any, PauliSum{N, W, T}}, ps2::PauliSum{N, W, T}) where {N, W, T}
    out = PauliSum(N, T)
    for (op1, coeff1) in ps1.parent
        for (op2, coeff2) in ps2
            prod = Pauli(op1) * Pauli(op2)
            c = coeff(prod)
            prod = PauliBasis(prod)
            if haskey(out, prod)
                out[prod] += c * coeff1' * coeff2
            else
                out[prod] = c * coeff1' * coeff2
            end
        end
    end
    return out
end
"""
    Base.:*(ps1::PauliSum{N, W, T}, ps2::Adjoint{<:Any, PauliSum{N, W, T}}) where {N, W, T}

Multiply two `PauliSum`s.
"""
function Base.:*(ps1::PauliSum{N, W, T}, ps2::Adjoint{<:Any, PauliSum{N, W, T}}) where {N, W, T}
    out = PauliSum(N, T)
    for (op1, coeff1) in ps1
        for (op2, coeff2) in ps2.parent
            prod = Pauli(op1) * Pauli(op2)
            c = coeff(prod)
            prod = PauliBasis(prod)
            if haskey(out, prod)
                out[prod] += c * coeff1 * coeff2'
            else
                out[prod] = c * coeff1 * coeff2'
            end
        end
    end
    return out
end

"""
    Base.:*(ps1::Adjoint{<:Any, PauliSum{N, W, T}}, ps2::Adjoint{<:Any, PauliSum{N, W, T}}) where {N, W, T}

Multiply two `PauliSum`s.
"""
function Base.:*(ps1::Adjoint{<:Any, PauliSum{N, W, T}}, ps2::Adjoint{<:Any, PauliSum{N, W, T}}) where {N, W, T}
    out = PauliSum(N, T)
    for (op1, coeff1) in ps1.parent
        for (op2, coeff2) in ps2.parent
            prod = Pauli(op1) * Pauli(op2)
            c = coeff(prod)
            prod = PauliBasis(prod)
            if haskey(out, prod)
                out[prod] += c * coeff1' * coeff2'
            else
                out[prod] = c * coeff1' * coeff2'
            end
        end
    end
    return out
end

function Base.:*(ps1::PauliSum{N, W, T}, a::Number) where {N, W, T}
    out = deepcopy(ps1)
    mul!(out, a)
    return out
end
Base.:*(a::Number, ps1::PauliSum{N, W, T}) where {N, W, T} = ps1 * a

function Base.:*(ps1::Adjoint{<:Any, PauliSum{N, W, T}}, a::Number) where {N, W, T}
    out = deepcopy(ps1.parent)
    map!(x->adjoint(x), values(out))
    mul!(out, a)
    return out
end
Base.:*(a::Number, ps1::Adjoint{<:Any, PauliSum{N, W, T}}) where {N, W, T} = ps1 * a

Base.getindex(ps::PauliSum, s::String) = ps[PauliBasis(s)]
function Base.getindex(ps::Adjoint{<:Any, PauliSum{N,W,T}}, a::PauliBasis{N,W}) where {N,W,T}
    return ps.parent[a]'
end

Base.keys(ps::Adjoint{<:Any, PauliSum{N,W,T}}) where {N,W,T} = keys(ps.parent)


"""
    otimes(p1::PauliSum{N,W1,T}, p2::PauliSum{M,W2,T}) where {N,M,W1,W2,T}

Tensor product of two `PauliSum`s, returning a `PauliSum{N+M}`.
"""
function otimes(p1::PauliSum{N,W1,T}, p2::PauliSum{M,W2,T}) where {N,M,W1,W2,T}
    out = PauliSum(N+M, T)
    for (op1,coeff1) in p1
        for (op2,coeff2) in p2
            out[op1 ⊗ op2] = coeff1 * coeff2
        end
    end
    return out
end

"""
    osum(p1::PauliSum{N,W1,T}, p2::PauliSum{M,W2,T}) where {N,M,W1,W2,T}

Direct sum of two `PauliSum`s: `p1 ⊕ p2 = p1 ⊗ I_M + I_N ⊗ p2`,
returning a `PauliSum{N+M, T}`.
"""
function osum(p1::PauliSum{N,W1,T}, p2::PauliSum{M,W2,T}) where {N,M,W1,W2,T}
    I_N = PauliSum(N, T); I_N[PauliBasis{N}(0, 0)] = one(T)
    I_M = PauliSum(M, T); I_M[PauliBasis{M}(0, 0)] = one(T)
    return p1 ⊗ I_M + I_N ⊗ p2
end
