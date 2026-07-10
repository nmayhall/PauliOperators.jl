# ============================================================
# SparsePauliVector — flat, preallocated, structure-of-arrays storage for
# fast serial Pauli propagation. The Dict-free analogue of PauliSum.
#
# Design contract: the steady-state hot path (rotation sweeps, merges,
# clips, reductions) allocates zero bytes. All storage is preallocated at
# construction or grown at window boundaries, and reused. No Dict, no
# per-term heap objects, isbits data only.
# ============================================================

"""
    _word_type(N)

Packed bit-word type for an `N`-qubit register: `UInt64` when the bits fit
(halves bit bandwidth), `UInt128` up to 128 qubits.

Beyond 128 qubits the intended extension is BitIntegers.jl (`UInt256`,
`UInt512`, ... are primitive `Unsigned` types supporting `⊻`, `&`,
`count_ones`, `<` natively), so every kernel is written generically over
`W<:Unsigned` with standard operators — adding larger lattices later means
extending this one function (plus a wider `PauliBasis` for conversions).
"""
_word_type(N::Integer) = N <= 64 ? UInt64 :
                         N <= 128 ? UInt128 :
                         error("N > 128 requires a multi-word type " *
                               "(add BitIntegers.jl and extend _word_type)")

# All key packing/unpacking goes through these two functions. The double
# `%` zero-extends: `Int128 % W` alone would sign-extend for W wider than
# 128 bits, and N = 128 uses the Int128 sign bit.
@inline _pack(::Type{W}, p::PauliBasis) where {W<:Unsigned} =
    ((p.z % UInt128) % W, (p.x % UInt128) % W)
@inline _unpack(::Type{PauliBasis{N}}, z::W, x::W) where {N,W<:Unsigned} =
    PauliBasis{N}((z % UInt128) % Int128, (x % UInt128) % Int128)

const _HIST_BINS = 64

"""
    SparsePauliVector{N,W,T}

A sum of `N`-qubit Pauli terms stored as flat sorted parallel arrays — the
zero-allocation replacement for the `Dict`-backed `PauliSum{N,T}`. `W` is
the packed bit-word type (`UInt64` for `N ≤ 64`, chosen automatically by
the constructors), `T` the coefficient type (`Float64` suffices for
Hermitian operators and halves coefficient bandwidth).

Three parallel-array buffer sets:

- live `z/x/c[1:n]`: current terms, **sorted by strictly increasing
  `(z, x)` key**, duplicate-free.
- append `az/ax/ac[1:an]`: sin-branch terms created during an evolution
  window, unsorted. `an == 0` whenever any public API other than the
  evolve internals runs.
- scratch `sz/sx/sc`: merge output; swapped with live by field assignment
  (pointer swap, never a copy).

`ws` is the preallocated sort/merge workspace, `hist` a fixed-size |c|
exponent histogram for adaptive truncation thresholds.

The struct is deliberately `mutable`: `n`/`an` change constantly and the
merge pointer-swaps live↔scratch via field reassignment. This costs
nothing in hot loops — kernels take the raw `Vector`s as arguments.

Supports the full `PauliSum` API (arithmetic, `evolve!`, `truncate!`,
expectation values, clips, ...). Convert with `PauliSum(v)` /
`SparsePauliVector(O)`. Note `setindex!`/`delete!` are O(n) (sorted
insert); build from a `PauliSum` or with `sum!` for bulk construction.
"""
mutable struct SparsePauliVector{N, W<:Unsigned, T<:Number}
    z::Vector{W}
    x::Vector{W}
    c::Vector{T}
    n::Int
    az::Vector{W}
    ax::Vector{W}
    ac::Vector{T}
    an::Int
    sz::Vector{W}
    sx::Vector{W}
    sc::Vector{T}
    ws::Vector{Tuple{W,W,T}}
    hist::Vector{Int}
end

"""
    AnyPauliSum{N,T}

Union of the two Pauli-sum representations, for methods that only iterate
`(PauliBasis, coefficient)` pairs and work identically on both.
"""
AnyPauliSum{N,T} = Union{PauliSum{N,T}, SparsePauliVector{N,W,T} where W}

function _alloc_spv(N::Int, ::Type{W}, ::Type{T}, live_cap::Int, append_cap::Int) where {W,T}
    return SparsePauliVector{N,W,T}(
        zeros(W, live_cap), zeros(W, live_cap), zeros(T, live_cap), 0,
        zeros(W, append_cap), zeros(W, append_cap), zeros(T, append_cap), 0,
        zeros(W, live_cap), zeros(W, live_cap), zeros(T, live_cap),
        # ws must hold all appends at a merge AND the live terms during the
        # construction-time sort
        Vector{Tuple{W,W,T}}(undef, max(append_cap, live_cap)),
        zeros(Int, _HIST_BINS))
end

# Boundary growth (the only allocation points after construction).
function _grow_live!(v::SparsePauliVector, need::Int)
    newcap = max(2 * length(v.z), need)
    resize!(v.z, newcap)
    resize!(v.x, newcap)
    resize!(v.c, newcap)
    resize!(v.sz, newcap)
    resize!(v.sx, newcap)
    resize!(v.sc, newcap)
    length(v.ws) < newcap && resize!(v.ws, newcap)
    return v
end

function _grow_append!(v::SparsePauliVector, need::Int)
    newcap = max(2 * length(v.az), need)
    resize!(v.az, newcap)
    resize!(v.ax, newcap)
    resize!(v.ac, newcap)
    length(v.ws) < newcap && resize!(v.ws, newcap)
    return v
end

# ------------------------------------------------------------
# Constructors and conversion
# ------------------------------------------------------------

"""
    SparsePauliVector(N, T=ComplexF64; capacity=16)

Empty `N`-qubit sum with coefficient type `T` and initial term capacity
`capacity`. The packed word type is chosen automatically.
"""
function SparsePauliVector(N::Integer, T::Type{<:Number}=ComplexF64; capacity::Int=16)
    cap = max(capacity, 1)
    return _alloc_spv(Int(N), _word_type(N), T, cap, cap)
end

_engine_coeff(::Type{T}, c::Number, tol::Float64) where {T<:Complex} = convert(T, c)
function _engine_coeff(::Type{T}, c::Number, tol::Float64) where {T<:Real}
    abs(imag(c)) <= tol * max(1.0, abs(c)) ||
        error("coefficient $c has an imaginary part; use a complex coefficient " *
              "type (T=ComplexF64) or provide Hermitian (real-coefficient) input")
    return convert(T, real(c))
end

"""
    SparsePauliVector(O::PauliSum{N}; T, capacity_factor=2.0, append_factor=1.0,
                      min_capacity=16, imag_tol=1e-10)

Convert a `Dict`-backed `PauliSum` into flat sorted storage.

`capacity_factor` sizes the live buffer relative to `length(O)` (headroom
for population growth between truncations); `append_factor` sizes the
evolve-time append buffer relative to the live buffer. Exhaustion during
evolution triggers an early merge and, if the population genuinely needs
more room, chunked buffer doubling at the window boundary — never a
hot-loop reallocation.

Real `T` requires (numerically) real coefficients: terms with
`|imag(c)| > imag_tol·max(1,|c|)` are an error.
"""
function SparsePauliVector(O::PauliSum{N,T0};
                           T::Type{<:Number}=T0,
                           capacity_factor::Real=2.0,
                           append_factor::Real=1.0,
                           min_capacity::Int=16,
                           imag_tol::Real=1e-10) where {N,T0}
    capacity_factor >= 1 || throw(ArgumentError("capacity_factor must be >= 1"))
    W = _word_type(N)
    live_cap = max(min_capacity, ceil(Int, capacity_factor * max(length(O), 1)))
    append_cap = max(min_capacity, ceil(Int, append_factor * live_cap))
    v = _alloc_spv(Int(N), W, T, live_cap, append_cap)
    tol = Float64(imag_tol)
    for (p, c) in O
        v.n += 1
        z, x = _pack(W, p)
        v.z[v.n] = z
        v.x[v.n] = x
        v.c[v.n] = _engine_coeff(T, c, tol)
    end
    _sort_live!(v)
    return v
end

SparsePauliVector(p::PauliBasis{N}; T::Type{<:Number}=ComplexF64) where {N} =
    (v = SparsePauliVector(N, T); v[p] = one(T); v)
SparsePauliVector(p::Pauli{N}; T::Type{<:Number}=ComplexF64) where {N} =
    (v = SparsePauliVector(N, T); v[PauliBasis(p)] = T(coeff(p)); v)

function _sort_live!(v::SparsePauliVector)
    @inbounds for i in 1:v.n
        v.ws[i] = (v.z[i], v.x[i], v.c[i])
    end
    _sort_ws!(v.ws, 1, v.n)
    @inbounds for i in 1:v.n
        v.z[i], v.x[i], v.c[i] = v.ws[i]
    end
    return v
end

"""
    PauliSum(v::SparsePauliVector)

Gather back into a `Dict`-based `PauliSum`. Only valid on merged state (no
pending appends), which is what every public operation leaves behind.
"""
function PauliSum(v::SparsePauliVector{N,W,T}) where {N,W,T}
    v.an == 0 ||
        error("SparsePauliVector has pending appends; gather only merged state")
    out = PauliSum(N, T)
    sizehint!(out, v.n)
    @inbounds for i in 1:v.n
        out[_unpack(PauliBasis{N}, v.z[i], v.x[i])] = v.c[i]
    end
    return out
end

function Base.convert(::Type{PauliSum{N,T}}, v::SparsePauliVector{N}) where {N,T}
    out = PauliSum(N, T)
    sizehint!(out, v.n)
    for (p, c) in v
        out[p] = convert(T, c)
    end
    return out
end

function Base.convert(::Type{SparsePauliVector{N,W,T}}, O::PauliSum{N}) where {N,W<:Unsigned,T}
    W === _word_type(N) ||
        error("requested word type $W; constructors use $(_word_type(N)) for N=$N")
    return SparsePauliVector(O; T=T)
end

function Base.rand(::Type{SparsePauliVector{N,W,T}}; n_paulis=2) where {N,W,T}
    v = _alloc_spv(Int(N), W, T, max(16, 2 * n_paulis), max(16, 2 * n_paulis))
    for _ in 1:n_paulis
        p = rand(Pauli{N})
        v[PauliBasis(p)] = coeff(p) * rand(T)
    end
    return v
end
Base.rand(::Type{SparsePauliVector{N}}; n_paulis=2, T=ComplexF64) where {N} =
    rand(SparsePauliVector{N, _word_type(N), T}; n_paulis=n_paulis)

function Base.copy(v::SparsePauliVector{N,W,T}) where {N,W,T}
    return SparsePauliVector{N,W,T}(
        copy(v.z), copy(v.x), copy(v.c), v.n,
        copy(v.az), copy(v.ax), copy(v.ac), v.an,
        copy(v.sz), copy(v.sx), copy(v.sc), copy(v.ws), copy(v.hist))
end

"""
    check_spv(v::SparsePauliVector)

Verify the structural invariants: consistent buffer lengths, and strictly
increasing (sorted, duplicate-free) live keys. Errors on violation.
Testing / debugging utility.
"""
function check_spv(v::SparsePauliVector{N,W,T}) where {N,W,T}
    v.n <= length(v.z) || error("count $(v.n) exceeds live capacity $(length(v.z))")
    v.an <= length(v.az) || error("append count $(v.an) exceeds capacity $(length(v.az))")
    length(v.z) == length(v.x) == length(v.c) ==
        length(v.sz) == length(v.sx) == length(v.sc) ||
        error("live/scratch buffer lengths inconsistent")
    length(v.az) == length(v.ax) == length(v.ac) ||
        error("append buffer lengths inconsistent")
    length(v.ws) >= max(length(v.z), length(v.az)) ||
        error("workspace smaller than live/append capacity")
    for i in 2:v.n
        _key_lt((v.z[i-1], v.x[i-1]), (v.z[i], v.x[i])) ||
            error("live keys not strictly increasing at index $i")
    end
    return true
end

# ------------------------------------------------------------
# Dict-idiom API (drop-in parity with PauliSum = Dict)
# ------------------------------------------------------------

# Binary search of the sorted live keys. Returns (found, index); when not
# found, index is the insertion point. O(log n), allocation-free.
@inline function _find(v::SparsePauliVector{N,W}, z::W, x::W) where {N,W}
    lo, hi = 1, v.n
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        if v.z[mid] == z && v.x[mid] == x
            return true, mid
        elseif _key_lt((v.z[mid], v.x[mid]), (z, x))
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return false, lo
end

function Base.get(v::SparsePauliVector{N,W,T}, p::PauliBasis{N}, default) where {N,W,T}
    z, x = _pack(W, p)
    found, idx = _find(v, z, x)
    return found ? v.c[idx] : default
end

function Base.getindex(v::SparsePauliVector{N,W,T}, p::PauliBasis{N}) where {N,W,T}
    z, x = _pack(W, p)
    found, idx = _find(v, z, x)
    found || throw(KeyError(p))
    return v.c[idx]
end
Base.getindex(v::SparsePauliVector, s::String) = v[PauliBasis(s)]

function Base.setindex!(v::SparsePauliVector{N,W,T}, val, p::PauliBasis{N}) where {N,W,T}
    z, x = _pack(W, p)
    found, idx = _find(v, z, x)
    if found
        @inbounds v.c[idx] = val
    else
        v.n == length(v.z) && _grow_live!(v, v.n + 1)
        @inbounds for i in v.n:-1:idx
            v.z[i+1] = v.z[i]
            v.x[i+1] = v.x[i]
            v.c[i+1] = v.c[i]
        end
        @inbounds begin
            v.z[idx] = z
            v.x[idx] = x
            v.c[idx] = val
        end
        v.n += 1
    end
    return v
end

function Base.get!(v::SparsePauliVector{N,W,T}, p::PauliBasis{N}, default) where {N,W,T}
    z, x = _pack(W, p)
    found, idx = _find(v, z, x)
    found && return v.c[idx]
    v[p] = default
    return convert(T, default)
end

function Base.delete!(v::SparsePauliVector{N,W,T}, p::PauliBasis{N}) where {N,W,T}
    z, x = _pack(W, p)
    found, idx = _find(v, z, x)
    found || return v
    @inbounds for i in idx:v.n-1
        v.z[i] = v.z[i+1]
        v.x[i] = v.x[i+1]
        v.c[i] = v.c[i+1]
    end
    v.n -= 1
    return v
end

function Base.haskey(v::SparsePauliVector{N,W}, p::PauliBasis{N}) where {N,W}
    z, x = _pack(W, p)
    return _find(v, z, x)[1]
end

Base.length(v::SparsePauliVector) = v.n
Base.isempty(v::SparsePauliVector) = v.n == 0
Base.empty!(v::SparsePauliVector) = (v.n = 0; v.an = 0; v)

Base.eltype(::Type{SparsePauliVector{N,W,T}}) where {N,W,T} = Pair{PauliBasis{N},T}

function Base.iterate(v::SparsePauliVector{N,W,T}, i::Int=1) where {N,W,T}
    i > v.n && return nothing
    @inbounds pr = Pair(_unpack(PauliBasis{N}, v.z[i], v.x[i]), v.c[i])
    return pr, i + 1
end

Base.keys(v::SparsePauliVector{N,W,T}) where {N,W,T} =
    (_unpack(PauliBasis{N}, v.z[i], v.x[i]) for i in 1:v.n)
Base.values(v::SparsePauliVector) = view(v.c, 1:v.n)
Base.pairs(v::SparsePauliVector) = v

function Base.:(==)(v1::SparsePauliVector{N}, v2::SparsePauliVector{N}) where {N}
    v1.n == v2.n || return false
    @inbounds for i in 1:v1.n
        (v1.z[i] == v2.z[i] && v1.x[i] == v2.x[i] && v1.c[i] == v2.c[i]) ||
            return false
    end
    return true
end

# In-place order-preserving filter over pairs — the Dict `filter!` analogue.
function Base.filter!(f, v::SparsePauliVector{N,W,T}) where {N,W,T}
    out = 0
    @inbounds for i in 1:v.n
        pr = Pair(_unpack(PauliBasis{N}, v.z[i], v.x[i]), v.c[i])
        f(pr)::Bool || continue
        out += 1
        if out != i
            v.z[out] = v.z[i]
            v.x[out] = v.x[i]
            v.c[out] = v.c[i]
        end
    end
    v.n = out
    return v
end
Base.filter(f, v::SparsePauliVector) = filter!(f, copy(v))

function Base.size(v::SparsePauliVector{N}) where {N}
    return (BigInt(2)^N, BigInt(2)^N)
end

# ------------------------------------------------------------
# Display and adjoint wrapper (parity with type_PauliSum.jl)
# ------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", v::SparsePauliVector)
    for (key, val) in v
        @printf(io, " %12.8f +%12.8fi %s\n", real(val), imag(val), key)
    end
end

Base.show(io::IO, v::SparsePauliVector{N,W,T}) where {N,W,T} =
    print(io, "SparsePauliVector{$N,$W,$T}: $(v.n) terms, capacity $(length(v.z))")

Base.adjoint(v::SparsePauliVector) = Adjoint(v)
Base.parent(v::Adjoint{<:Any, <:SparsePauliVector}) = v.parent

function Base.show(io::IO, ::MIME"text/plain",
                   v::Adjoint{<:Any, <:SparsePauliVector{N,W,T}}) where {N,W,T}
    for (key, val) in v.parent
        @printf(io, " %12.8f +%12.8fi %s\n", real(val), -imag(val), key)
    end
end

function Base.getindex(v::Adjoint{<:Any, <:SparsePauliVector{N,W,T}},
                       a::PauliBasis{N}) where {N,W,T}
    return v.parent[a]'
end
Base.keys(v::Adjoint{<:Any, <:SparsePauliVector}) = keys(v.parent)
