# ============================================================
# SparsePauliVector algebra and observables — PauliSum-parity operations
# on flat sorted storage. Hot reductions (norm, tr, inner_product,
# expectation_value, sum!, mul!, commutator!) are allocation-free; binary
# operators allocate only their result.
# ============================================================

# ------------------------------------------------------------
# Scalar operations
# ------------------------------------------------------------

function LinearAlgebra.mul!(v::SparsePauliVector, a::Number)
    @inbounds for i in 1:v.n
        v.c[i] *= a
    end
    return v
end

function Base.:-(v::SparsePauliVector)
    out = copy(v)
    mul!(out, -1)
    return out
end

function Base.:*(v::SparsePauliVector, a::Number)
    out = copy(v)
    mul!(out, a)
    return out
end
Base.:*(a::Number, v::SparsePauliVector) = v * a

function Base.:*(v::Adjoint{<:Any, <:SparsePauliVector{N,W,T}}, a::Number) where {N,W,T}
    out = copy(v.parent)
    @inbounds for i in 1:out.n
        out.c[i] = adjoint(out.c[i])
    end
    mul!(out, a)
    return out
end
Base.:*(a::Number, v::Adjoint{<:Any, <:SparsePauliVector}) = v * a

function LinearAlgebra.ishermitian(v::SparsePauliVector)
    isherm = true
    @inbounds for i in 1:v.n
        isherm = isherm && isapprox(imag(v.c[i]), 0, atol=1e-16)
    end
    return isherm
end

# Identity key (0, 0) is the minimum of the sorted order: if present it
# occupies slot 1. Allocation-free.
function LinearAlgebra.tr(v::SparsePauliVector{N,W,T}) where {N,W,T}
    (v.n >= 1 && v.z[1] == zero(W) && v.x[1] == zero(W)) || return zero(T) * 2^N
    return v.c[1] * 2^N
end

function LinearAlgebra.norm(v::SparsePauliVector{N,W,T}, p::Real=2) where {N,W,T}
    if p == 2
        s = zero(real(T))
        @inbounds for i in 1:v.n
            s += abs2(v.c[i])
        end
        return sqrt(s)
    elseif p == 1
        s = zero(real(T))
        @inbounds for i in 1:v.n
            s += abs(v.c[i])
        end
        return s
    elseif p == Inf
        s = zero(real(T))
        @inbounds for i in 1:v.n
            a = abs(v.c[i])
            a > s && (s = a)
        end
        return s
    else
        s = zero(real(T))
        @inbounds for i in 1:v.n
            s += abs(v.c[i])^p
        end
        return s^(1 / p)
    end
end

# Norm of the difference without materializing it (two-pointer walk), then
# the same tolerance formula as isapprox(::PauliSum, ::PauliSum).
function Base.isapprox(v1::SparsePauliVector{N}, v2::SparsePauliVector{N};
                       atol=1e-14, rtol=0) where {N}
    s = 0.0
    i = 1
    j = 1
    @inbounds while i <= v1.n || j <= v2.n
        if j > v2.n || (i <= v1.n && _key_lt((v1.z[i], v1.x[i]), (v2.z[j], v2.x[j])))
            s += abs2(v1.c[i])
            i += 1
        elseif i > v1.n || _key_lt((v2.z[j], v2.x[j]), (v1.z[i], v1.x[i]))
            s += abs2(v2.c[j])
            j += 1
        else
            s += abs2(v1.c[i] - v2.c[j])
            i += 1
            j += 1
        end
    end
    return sqrt(s) <= atol + rtol * max(norm(v1), norm(v2))
end

function Base.Matrix(v::SparsePauliVector{N,W,T}) where {N,W,T}
    out = zeros(T, Int128(2)^N, Int128(2)^N)
    for (op, c) in v
        out .+= Matrix(op) .* c
    end
    return out
end

function Base.Matrix(v::Adjoint{<:Any, <:SparsePauliVector{N,W,T}}) where {N,W,T}
    out = zeros(T, Int128(2)^N, Int128(2)^N)
    for (op, c) in v.parent
        out .+= Matrix(op) .* adjoint(c)
    end
    return out
end

# ------------------------------------------------------------
# Addition / subtraction (two-pointer merged walk of sorted inputs)
# ------------------------------------------------------------

# out = t1·v1 + t2·v2 with per-source transforms (sign flip / conjugation
# as plain Bools — no closures). Zero entries are kept (Dict parity).
function _add_spv(v1::SparsePauliVector{N,W,T}, conj1::Bool, neg1::Bool,
                  v2::SparsePauliVector{N,W,T}, conj2::Bool, neg2::Bool) where {N,W,T}
    out = _alloc_spv(N, W, T, max(16, v1.n + v2.n), 16)
    i = 1
    j = 1
    m = 0
    @inbounds while i <= v1.n || j <= v2.n
        take1 = j > v2.n || (i <= v1.n && !_key_lt((v2.z[j], v2.x[j]), (v1.z[i], v1.x[i])))
        take2 = i > v1.n || (j <= v2.n && !_key_lt((v1.z[i], v1.x[i]), (v2.z[j], v2.x[j])))
        local kz::W, kx::W
        acc = zero(T)
        if take1
            kz = v1.z[i]
            kx = v1.x[i]
            c = conj1 ? conj(v1.c[i]) : v1.c[i]
            acc += neg1 ? -c : c
            i += 1
        end
        if take2
            kz = v2.z[j]
            kx = v2.x[j]
            c = conj2 ? conj(v2.c[j]) : v2.c[j]
            acc += neg2 ? -c : c
            j += 1
        end
        m += 1
        out.z[m] = kz
        out.x[m] = kx
        out.c[m] = acc
    end
    out.n = m
    return out
end

Base.:+(v1::SparsePauliVector{N,W,T}, v2::SparsePauliVector{N,W,T}) where {N,W,T} =
    _add_spv(v1, false, false, v2, false, false)
Base.:-(v1::SparsePauliVector{N,W,T}, v2::SparsePauliVector{N,W,T}) where {N,W,T} =
    _add_spv(v1, false, false, v2, false, true)
Base.:+(v1::SparsePauliVector{N,W,T},
        v2::Adjoint{<:Any, <:SparsePauliVector{N,W,T}}) where {N,W,T} =
    _add_spv(v1, false, false, v2.parent, true, false)
Base.:+(v2::Adjoint{<:Any, <:SparsePauliVector{N,W,T}},
        v1::SparsePauliVector{N,W,T}) where {N,W,T} = v1 + v2

"""
    Base.sum!(v1::SparsePauliVector, v2::SparsePauliVector)

In-place accumulate `v1 += v2` — the flat-storage `mergewith!(+, ...)`.
`v2` is already sorted, so this is a straight merge (no sort);
allocation-free once `v1`'s buffers cover the union.
"""
function Base.sum!(v1::SparsePauliVector{N,W,T}, v2::SparsePauliVector{N,W,T}) where {N,W,T}
    v1.an == 0 || error("sum! on a SparsePauliVector with pending appends")
    length(v1.ws) < v2.n && resize!(v1.ws, v2.n)
    @inbounds for i in 1:v2.n
        v1.ws[i] = (v2.z[i], v2.x[i], v2.c[i])
    end
    _merge_spv!(v1, v2.n, NOFILTER)
    return v1
end

function Base.sum!(v1::SparsePauliVector{N,W,T},
                   v2::Adjoint{<:Any, <:SparsePauliVector{N,W,T}}) where {N,W,T}
    p = v2.parent
    v1.an == 0 || error("sum! on a SparsePauliVector with pending appends")
    length(v1.ws) < p.n && resize!(v1.ws, p.n)
    @inbounds for i in 1:p.n
        v1.ws[i] = (p.z[i], p.x[i], conj(p.c[i]))
    end
    _merge_spv!(v1, p.n, NOFILTER)
    return v1
end

# Single-term promotion (the SparsePauliVector mirror of
# addition.jl's Singles/Sums methods, without touching the Sums union).
function _single_spv(p::Union{Pauli{N}, PauliBasis{N}}, ::Type{W}, ::Type{T}) where {N,W,T}
    v = _alloc_spv(N, W, T, 16, 16)
    v[PauliBasis(p)] = convert(T, coeff(p))
    return v
end

function Base.sum!(v::SparsePauliVector{N,W,T}, p::Union{Pauli{N}, PauliBasis{N}}) where {N,W,T}
    b = PauliBasis(p)
    v[b] = get(v, b, zero(T)) + convert(T, coeff(p))
    return v
end
Base.sum!(p::Union{Pauli{N}, PauliBasis{N}}, v::SparsePauliVector{N}) where {N} = sum!(v, p)

Base.:+(p::Union{Pauli{N}, PauliBasis{N}}, v::SparsePauliVector{N,W,T}) where {N,W,T} =
    _single_spv(p, W, T) + v
Base.:+(v::SparsePauliVector{N,W,T}, p::Union{Pauli{N}, PauliBasis{N}}) where {N,W,T} =
    v + _single_spv(p, W, T)
Base.:-(p::Union{Pauli{N}, PauliBasis{N}}, v::SparsePauliVector{N,W,T}) where {N,W,T} =
    _single_spv(p, W, T) - v
Base.:-(v::SparsePauliVector{N,W,T}, p::Union{Pauli{N}, PauliBasis{N}}) where {N,W,T} =
    v - _single_spv(p, W, T)

# ------------------------------------------------------------
# Multiplication (bit-kernel products, sort + dedup scan)
# ------------------------------------------------------------

# Scan sorted ws[1:m] into out's live buffer, summing equal-key runs.
# Zero coefficients (cancellations) are kept — Dict parity.
function _triples_to_live!(out::SparsePauliVector{N,W,T}, m::Int) where {N,W,T}
    length(out.z) < m && _grow_live!(out, m)
    n = 0
    i = 1
    @inbounds while i <= m
        kz, kx, acc = out.ws[i]
        i += 1
        while i <= m && _key_eq(out.ws[i], (kz, kx))
            acc += out.ws[i][3]
            i += 1
        end
        n += 1
        out.z[n] = kz
        out.x[n] = kx
        out.c[n] = acc
    end
    out.n = n
    return out
end

# out = (conj1 ? A† : A) · (conj2 ? B† : B), key = XOR, phase i^k from the
# fused-phase identity (commutator.jl). Adjoint of a PauliBasis term is the
# same basis with conjugated coefficient, so only coefficients transform.
function _mul_spv(A::SparsePauliVector{N,W,T}, conj1::Bool,
                  B::SparsePauliVector{N,W,T}, conj2::Bool) where {N,W,T}
    nout = A.n * B.n
    out = _alloc_spv(N, W, T, max(16, nout), 16)
    length(out.ws) < nout && resize!(out.ws, nout)
    m = 0
    @inbounds for i in 1:A.n
        az = A.z[i]
        ax = A.x[i]
        n_a = count_ones(az & ax)
        ca = conj1 ? conj(A.c[i]) : A.c[i]
        for j in 1:B.n
            bz = B.z[j]
            bx = B.x[j]
            zp = az ⊻ bz
            xp = ax ⊻ bx
            k = mod(count_ones(zp & xp) - n_a - count_ones(bz & bx) +
                    2 * count_ones(ax & bz), 4)
            cb = conj2 ? conj(B.c[j]) : B.c[j]
            m += 1
            out.ws[m] = (zp, xp, convert(T, PHASE_TBL[k + 1] * ca * cb))
        end
    end
    _sort_ws!(out.ws, 1, m)
    return _triples_to_live!(out, m)
end

Base.:*(A::SparsePauliVector{N,W,T}, B::SparsePauliVector{N,W,T}) where {N,W,T} =
    _mul_spv(A, false, B, false)
Base.:*(A::Adjoint{<:Any, <:SparsePauliVector{N,W,T}},
        B::SparsePauliVector{N,W,T}) where {N,W,T} =
    _mul_spv(A.parent, true, B, false)
Base.:*(A::SparsePauliVector{N,W,T},
        B::Adjoint{<:Any, <:SparsePauliVector{N,W,T}}) where {N,W,T} =
    _mul_spv(A, false, B.parent, true)
Base.:*(A::Adjoint{<:Any, <:SparsePauliVector{N,W,T}},
        B::Adjoint{<:Any, <:SparsePauliVector{N,W,T}}) where {N,W,T} =
    _mul_spv(A.parent, true, B.parent, true)

# Single Pauli × sum: one XOR + phase per term; output re-sorted.
function _mul_single_spv(v::SparsePauliVector{N,W,T}, p::Union{Pauli{N}, PauliBasis{N}},
                         left::Bool) where {N,W,T}
    b = PauliBasis(p)
    gz, gx = _pack(W, b)
    n_g = count_ones(gz & gx)
    cp = convert(ComplexF64, coeff(p))
    out = _alloc_spv(N, W, T, max(16, v.n), 16)
    length(out.ws) < v.n && resize!(out.ws, v.n)
    @inbounds for i in 1:v.n
        vz = v.z[i]
        vx = v.x[i]
        zp = gz ⊻ vz
        xp = gx ⊻ vx
        n_v = count_ones(vz & vx)
        n_p = count_ones(zp & xp)
        # phase exponent for (left ? p·v : v·p)
        m1 = left ? count_ones(gx & vz) : count_ones(vx & gz)
        n_first = left ? n_g : n_v
        n_second = left ? n_v : n_g
        k = mod(n_p - n_first - n_second + 2 * m1, 4)
        out.ws[i] = (zp, xp, convert(T, PHASE_TBL[k + 1] * cp * v.c[i]))
    end
    _sort_ws!(out.ws, 1, v.n)
    return _triples_to_live!(out, v.n)
end

Base.:*(p::Union{Pauli{N}, PauliBasis{N}}, v::SparsePauliVector{N}) where {N} =
    _mul_single_spv(v, p, true)
Base.:*(v::SparsePauliVector{N}, p::Union{Pauli{N}, PauliBasis{N}}) where {N} =
    _mul_single_spv(v, p, false)

function Base.:*(O::SparsePauliVector{N,W,T}, k::Ket{N}) where {N,W,T}
    out = KetSum(N)
    for (p, c) in O
        c2, k2 = p * k
        tmp = get(out, k2, 0.0)
        out[k2] = tmp + c2 * c
    end
    return out
end

# ------------------------------------------------------------
# Tensor product / direct sum
# ------------------------------------------------------------

function otimes(v1::SparsePauliVector{N,W1,T}, v2::SparsePauliVector{M,W2,T}) where {N,M,W1,W2,T}
    NM = N + M
    Wo = _word_type(NM)
    nout = v1.n * v2.n
    out = _alloc_spv(NM, Wo, T, max(16, nout), 16)
    length(out.ws) < nout && resize!(out.ws, nout)
    m = 0
    for (op1, c1) in v1
        for (op2, c2) in v2
            m += 1
            z, x = _pack(Wo, op1 ⊗ op2)
            out.ws[m] = (z, x, c1 * c2)
        end
    end
    _sort_ws!(out.ws, 1, m)
    return _triples_to_live!(out, m)   # keys unique; scan is a plain copy
end

function osum(v1::SparsePauliVector{N,W1,T}, v2::SparsePauliVector{M,W2,T}) where {N,M,W1,W2,T}
    I_N = SparsePauliVector(N, T)
    I_N[PauliBasis{N}(0, 0)] = one(T)
    I_M = SparsePauliVector(M, T)
    I_M[PauliBasis{M}(0, 0)] = one(T)
    return v1 ⊗ I_M + I_N ⊗ v2
end

# ------------------------------------------------------------
# Inner product, commutators
# ------------------------------------------------------------

"""
    inner_product(v1::SparsePauliVector, v2::SparsePauliVector)

Liouville inner product tr(v1†·v2) over shared basis terms (two-pointer
coefficient dot product; both inputs sorted). Allocation-free.
"""
function inner_product(v1::SparsePauliVector{N,W,T}, v2::SparsePauliVector{N,W,T}) where {N,W,T}
    out = zero(T)
    i = 1
    j = 1
    @inbounds while i <= v1.n && j <= v2.n
        ka = (v1.z[i], v1.x[i])
        kb = (v2.z[j], v2.x[j])
        if _key_eq(ka, kb)
            out += conj(v1.c[i]) * v2.c[j]
            i += 1
            j += 1
        elseif _key_lt(ka, kb)
            i += 1
        else
            j += 1
        end
    end
    return out
end

# Fill ws[1:*] with (anti)commutator triples. Returns (m, overflowed);
# never writes past length(ws).
function _commutator_triples!(ws::Vector{Tuple{W,W,T}},
                              A::SparsePauliVector{N,W,T},
                              B::SparsePauliVector{N,W,T},
                              anti::Bool) where {N,W,T}
    m = 0
    cap = length(ws)
    @inbounds for i in 1:A.n
        az = A.z[i]
        ax = A.x[i]
        n_a = count_ones(az & ax)
        ca = A.c[i]
        for j in 1:B.n
            bz = B.z[j]
            bx = B.x[j]
            m1 = count_ones(ax & bz)
            if !anti
                iseven(m1 - count_ones(az & bx)) && continue
            end
            zp = az ⊻ bz
            xp = ax ⊻ bx
            k = mod(count_ones(zp & xp) - n_a - count_ones(bz & bx) + 2 * m1, 4)
            local c::T
            if anti
                isodd(k) && continue
                c = convert(T, (2 * (1 - k)) * ca * B.c[j])
            else
                c = convert(T, (2 * (2 - k)) * im * ca * B.c[j])
            end
            m += 1
            m <= cap || return m, true
            ws[m] = (zp, xp, c)
        end
    end
    return m, false
end

"""
    commutator!(out::SparsePauliVector, A::SparsePauliVector, B::SparsePauliVector)

Compute `[A, B] = AB - BA` into `out` (contents overwritten), using
`out.ws` as the pairing buffer. Allocation-free: errors if `out`'s
workspace or live capacity cannot hold the result — size `out` with
`SparsePauliVector(N, T; capacity=...)` or use `commutator(A, B)`, which
grows automatically. Requires complex `T` (commutator coefficients are
imaginary; store `-i[A,B]` yourself if you need a real-typed engine).
"""
function commutator!(out::SparsePauliVector{N,W,T}, A::SparsePauliVector{N,W,T},
                     B::SparsePauliVector{N,W,T}) where {N,W,T}
    T <: Real && error("commutator coefficients are imaginary; use a complex " *
                       "coefficient type (T=ComplexF64)")
    m, ovf = _commutator_triples!(out.ws, A, B, false)
    ovf && error("commutator! workspace overflow ($(m)+ pairs, capacity " *
                 "$(length(out.ws))); enlarge `out` or use commutator(A, B)")
    _sort_ws!(out.ws, 1, m)
    m <= length(out.z) || error("commutator! live-capacity overflow; enlarge `out` " *
                                "or use commutator(A, B)")
    out.an = 0
    return _triples_to_live!(out, m)
end

function anticommutator!(out::SparsePauliVector{N,W,T}, A::SparsePauliVector{N,W,T},
                         B::SparsePauliVector{N,W,T}) where {N,W,T}
    m, ovf = _commutator_triples!(out.ws, A, B, true)
    ovf && error("anticommutator! workspace overflow ($(m)+ pairs, capacity " *
                 "$(length(out.ws))); enlarge `out` or use anticommutator(A, B)")
    _sort_ws!(out.ws, 1, m)
    m <= length(out.z) || error("anticommutator! live-capacity overflow; enlarge " *
                                "`out` or use anticommutator(A, B)")
    out.an = 0
    return _triples_to_live!(out, m)
end

function commutator(A::SparsePauliVector{N,W,T}, B::SparsePauliVector{N,W,T}) where {N,W,T}
    nout = A.n * B.n
    out = _alloc_spv(N, W, T, max(16, nout), 16)
    length(out.ws) < nout && resize!(out.ws, nout)
    return commutator!(out, A, B)
end

function anticommutator(A::SparsePauliVector{N,W,T}, B::SparsePauliVector{N,W,T}) where {N,W,T}
    nout = A.n * B.n
    out = _alloc_spv(N, W, T, max(16, nout), 16)
    length(out.ws) < nout && resize!(out.ws, nout)
    return anticommutator!(out, A, B)
end

# ------------------------------------------------------------
# Expectation values against Dyad-type states (Ket kernel in spv_evolve.jl)
# ------------------------------------------------------------

function expectation_value(v::SparsePauliVector{N,W,T},
                           d::Union{Dyad{N}, DyadBasis{N}}) where {N,W,T}
    eval = zero(T)
    for (pi, ci) in v
        eval += expectation_value(pi, d) * ci
    end
    return eval
end

function expectation_value(v::SparsePauliVector{N,W,T}, d::DyadSum{N,T2}) where {N,W,T,T2}
    eval = zero(promote_type(T, T2))
    for (pi, ci) in v
        for (dj, cj) in d
            eval += expectation_value(pi, dj) * ci * cj
        end
    end
    return eval
end

# ------------------------------------------------------------
# Analysis helpers with Dict-specific bodies (the pair-iterating ones in
# analysis.jl are widened to AnyPauliSum instead)
# ------------------------------------------------------------

"""
    largest(v::SparsePauliVector)

Return the term with the largest absolute coefficient as a single-term
`SparsePauliVector` (parity with `largest(::PauliSum)`).
"""
function largest(v::SparsePauliVector{N,W,T}) where {N,W,T}
    v.n > 0 || throw(ArgumentError("collection must be non-empty"))
    best = -1.0
    bi = 1
    @inbounds for i in 1:v.n
        a = abs(v.c[i])
        if a > best
            best = a
            bi = i
        end
    end
    out = SparsePauliVector(N, T)
    out[_unpack(PauliBasis{N}, v.z[bi], v.x[bi])] = v.c[bi]
    return out
end

"""
    largest_diag(v::SparsePauliVector)

Return the `PauliBasis => coefficient` pair for the diagonal term (x == 0)
with the largest absolute coefficient (parity with `largest_diag(::PauliSum)`).
"""
function largest_diag(v::SparsePauliVector{N,W,T}) where {N,W,T}
    best = -1.0
    bi = 0
    @inbounds for i in 1:v.n
        v.x[i] == zero(W) || continue
        a = abs(v.c[i])
        if a > best
            best = a
            bi = i
        end
    end
    bi > 0 || throw(ArgumentError("collection must be non-empty"))
    return Pair(_unpack(PauliBasis{N}, v.z[bi], v.x[bi]), v.c[bi])
end

# ------------------------------------------------------------
# Diagonal filters (clip.jl parity)
# ------------------------------------------------------------

function offdiag(v::SparsePauliVector{N,W,T}) where {N,W,T}
    out = copy(v)
    n = 0
    @inbounds for i in 1:out.n
        out.x[i] == zero(W) && continue
        n += 1
        out.z[n] = out.z[i]
        out.x[n] = out.x[i]
        out.c[n] = out.c[i]
    end
    out.n = n
    return out
end

function LinearAlgebra.diag(v::SparsePauliVector{N,W,T}) where {N,W,T}
    out = copy(v)
    n = 0
    @inbounds for i in 1:out.n
        out.x[i] == zero(W) || continue
        n += 1
        out.z[n] = out.z[i]
        out.x[n] = out.x[i]
        out.c[n] = out.c[i]
    end
    out.n = n
    return out
end
