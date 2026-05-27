"""
    abstract type CliffordGate

Supertype for Clifford operations. Concrete subtypes include single-qubit primitives
(`Hadamard`, `PhaseGate`, `PhaseDg`, `SqrtX`, `SqrtY`, `PauliX`, `PauliY`, `PauliZ`),
two-qubit primitives (`CNOT`, `CZ`, `SWAP`), and `CliffordTableau{N}` for arbitrary
N-qubit Cliffords stored as a symplectic tableau.

All Cliffords map Hermitian Pauli strings to ±(Hermitian Pauli strings), enabling
direct bit-level transformation of `PauliSum`s and computational-basis state updates
on `KetSum`s without rotation-based evolution.
"""
abstract type CliffordGate end

# ------- Primitive gate types ------------------------------------------------

struct Hadamard  <: CliffordGate; q::Int; end
struct PhaseGate <: CliffordGate; q::Int; end   # S = diag(1, i)
struct PhaseDg   <: CliffordGate; q::Int; end   # S†
struct SqrtX     <: CliffordGate; q::Int; end   # √X = exp(-iπ/4 X)
struct SqrtY     <: CliffordGate; q::Int; end   # √Y = exp(-iπ/4 Y)
struct PauliX    <: CliffordGate; q::Int; end
struct PauliY    <: CliffordGate; q::Int; end
struct PauliZ    <: CliffordGate; q::Int; end

struct CNOT <: CliffordGate; c::Int; t::Int; end
struct CZ   <: CliffordGate; c::Int; t::Int; end
struct SWAP <: CliffordGate; a::Int; b::Int; end

# ------- CliffordTableau -----------------------------------------------------

"""
    CliffordTableau{N}

Aaronson-Gottesman symplectic tableau describing an arbitrary N-qubit Clifford `C`
by its action on each generator:

- `C · X_i · C† = (-1)^x_sign[i] · PauliBasis(z=x_to_z[i], x=x_to_x[i])`
- `C · Z_i · C† = (-1)^z_sign[i] · PauliBasis(z=z_to_z[i], x=z_to_x[i])`

Constructors:

- `CliffordTableau{N}()` — identity tableau.
- `CliffordTableau{N}(g::CliffordGate)` — tableau for a primitive on N qubits.
- `CliffordTableau{N}(gates::AbstractVector{<:CliffordGate})` — tableau composed
   from a sequence of primitives (applied left-to-right).
- `CliffordTableau{N}(U::AbstractMatrix, qs::Vector{Int})` — tableau identified
   from a small 2^length(qs) × 2^length(qs) dense unitary acting on qubits `qs`.
"""
struct CliffordTableau{N} <: CliffordGate
    x_to_z::Vector{Int128}
    x_to_x::Vector{Int128}
    z_to_z::Vector{Int128}
    z_to_x::Vector{Int128}
    x_sign::BitVector
    z_sign::BitVector
end

function CliffordTableau{N}() where N
    x_to_z = zeros(Int128, N)
    x_to_x = Int128[Int128(1) << (i-1) for i in 1:N]
    z_to_z = Int128[Int128(1) << (i-1) for i in 1:N]
    z_to_x = zeros(Int128, N)
    return CliffordTableau{N}(x_to_z, x_to_x, z_to_z, z_to_x, falses(N), falses(N))
end

Base.copy(C::CliffordTableau{N}) where N = CliffordTableau{N}(
    copy(C.x_to_z), copy(C.x_to_x), copy(C.z_to_z), copy(C.z_to_x),
    copy(C.x_sign), copy(C.z_sign))

const _SUBSCRIPT_DIGITS = ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉')
_sub(i::Integer) = join(_SUBSCRIPT_DIGITS[Int(d - '0') + 1] for d in string(i))

# Compact one-line form (used inside containers, `print`, etc.)
Base.show(io::IO, C::CliffordTableau{N}) where N = print(io, "CliffordTableau{", N, "}(…)")

# Multi-line form (REPL / show(stdout, MIME"text/plain", C)). Lists each generator
# image: "X_i → ± Pauli string" then "Z_i → ± Pauli string".
function Base.show(io::IO, ::MIME"text/plain", C::CliffordTableau{N}) where N
    print(io, "CliffordTableau{", N, "}:")
    width = ndigits(N)
    for i in 1:N
        sgn = C.x_sign[i] ? '-' : '+'
        p   = string(PauliBasis{N}(C.x_to_z[i], C.x_to_x[i]))
        print(io, "\n  X", _sub(i), " "^(width - ndigits(i)), " → ", sgn, " ", p)
    end
    for i in 1:N
        sgn = C.z_sign[i] ? '-' : '+'
        p   = string(PauliBasis{N}(C.z_to_z[i], C.z_to_x[i]))
        print(io, "\n  Z", _sub(i), " "^(width - ndigits(i)), " → ", sgn, " ", p)
    end
end

function Base.:(==)(A::CliffordTableau{N}, B::CliffordTableau{N}) where N
    A.x_to_z == B.x_to_z && A.x_to_x == B.x_to_x &&
    A.z_to_z == B.z_to_z && A.z_to_x == B.z_to_x &&
    A.x_sign == B.x_sign && A.z_sign == B.z_sign
end

# ------- Tableau builders for primitives -------------------------------------

@inline _bit(i) = Int128(1) << (i - 1)

function CliffordTableau{N}(g::Hadamard) where N
    1 <= g.q <= N || throw(BoundsError(1:N, g.q))
    C = CliffordTableau{N}()
    m = _bit(g.q)
    # X_q ↔ Z_q
    C.x_to_x[g.q] = Int128(0); C.x_to_z[g.q] = m
    C.z_to_x[g.q] = m;          C.z_to_z[g.q] = Int128(0)
    return C
end

function CliffordTableau{N}(g::PhaseGate) where N
    1 <= g.q <= N || throw(BoundsError(1:N, g.q))
    C = CliffordTableau{N}()
    m = _bit(g.q)
    # S X S† = Y;  S Z S† = Z
    C.x_to_z[g.q] = m   # Y has both bits on at q
    return C
end

function CliffordTableau{N}(g::PhaseDg) where N
    1 <= g.q <= N || throw(BoundsError(1:N, g.q))
    C = CliffordTableau{N}()
    m = _bit(g.q)
    # S† X S = -Y;  S† Z S = Z
    C.x_to_z[g.q] = m
    C.x_sign[g.q] = true
    return C
end

function CliffordTableau{N}(g::SqrtX) where N
    1 <= g.q <= N || throw(BoundsError(1:N, g.q))
    C = CliffordTableau{N}()
    m = _bit(g.q)
    # √X X √X† = X;  √X Z √X† = -Y;  (and √X Y √X† = Z follows)
    C.z_to_x[g.q] = m
    C.z_sign[g.q] = true   # Z -> -Y => sign of image of Z is negative
    return C
end

function CliffordTableau{N}(g::SqrtY) where N
    1 <= g.q <= N || throw(BoundsError(1:N, g.q))
    C = CliffordTableau{N}()
    m = _bit(g.q)
    # √Y X √Y† = -Z;  √Y Z √Y† = X
    C.x_to_x[g.q] = Int128(0); C.x_to_z[g.q] = m
    C.x_sign[g.q] = true       # X -> -Z
    C.z_to_z[g.q] = Int128(0); C.z_to_x[g.q] = m
    return C
end

function CliffordTableau{N}(g::PauliX) where N
    1 <= g.q <= N || throw(BoundsError(1:N, g.q))
    C = CliffordTableau{N}()
    # X X X = X;  X Z X = -Z
    C.z_sign[g.q] = true
    return C
end

function CliffordTableau{N}(g::PauliY) where N
    1 <= g.q <= N || throw(BoundsError(1:N, g.q))
    C = CliffordTableau{N}()
    # Y X Y = -X;  Y Z Y = -Z
    C.x_sign[g.q] = true
    C.z_sign[g.q] = true
    return C
end

function CliffordTableau{N}(g::PauliZ) where N
    1 <= g.q <= N || throw(BoundsError(1:N, g.q))
    C = CliffordTableau{N}()
    # Z X Z = -X;  Z Z Z = Z
    C.x_sign[g.q] = true
    return C
end

function CliffordTableau{N}(g::CNOT) where N
    1 <= g.c <= N || throw(BoundsError(1:N, g.c))
    1 <= g.t <= N || throw(BoundsError(1:N, g.t))
    g.c != g.t || throw(ArgumentError("CNOT control and target must differ"))
    C = CliffordTableau{N}()
    mc = _bit(g.c); mt = _bit(g.t)
    # X_c -> X_c X_t
    C.x_to_x[g.c] = mc | mt
    # Z_t -> Z_c Z_t
    C.z_to_z[g.t] = mc | mt
    return C
end

function CliffordTableau{N}(g::CZ) where N
    1 <= g.c <= N || throw(BoundsError(1:N, g.c))
    1 <= g.t <= N || throw(BoundsError(1:N, g.t))
    g.c != g.t || throw(ArgumentError("CZ qubits must differ"))
    C = CliffordTableau{N}()
    mc = _bit(g.c); mt = _bit(g.t)
    # X_c -> X_c Z_t;  X_t -> Z_c X_t
    C.x_to_z[g.c] = mt
    C.x_to_z[g.t] = mc
    return C
end

function CliffordTableau{N}(g::SWAP) where N
    1 <= g.a <= N || throw(BoundsError(1:N, g.a))
    1 <= g.b <= N || throw(BoundsError(1:N, g.b))
    g.a != g.b || throw(ArgumentError("SWAP qubits must differ"))
    C = CliffordTableau{N}()
    ma = _bit(g.a); mb = _bit(g.b)
    # X_a ↔ X_b;  Z_a ↔ Z_b
    C.x_to_x[g.a] = mb; C.x_to_x[g.b] = ma
    C.z_to_z[g.a] = mb; C.z_to_z[g.b] = ma
    return C
end

CliffordTableau{N}(C::CliffordTableau{N}) where N = copy(C)

function CliffordTableau{N}(gates::AbstractVector{<:CliffordGate}) where N
    out = CliffordTableau{N}()
    for g in gates
        out = CliffordTableau{N}(g) * out
    end
    return out
end

# Catch the common mistake of calling `CliffordTableau(g)` without the qubit
# count — primitives don't carry N, so it has to be supplied explicitly.
function CliffordTableau(g::CliffordGate)
    throw(ArgumentError(
        "CliffordTableau(g) needs the qubit count: write `CliffordTableau{N}(g)` " *
        "(e.g. `CliffordTableau{3}(CNOT(1, 2))`)."))
end
function CliffordTableau(gates::AbstractVector{<:CliffordGate})
    throw(ArgumentError(
        "CliffordTableau(gates) needs the qubit count: write " *
        "`CliffordTableau{N}(gates)`."))
end

# ------- Apply: CliffordTableau on PauliBasis --------------------------------

"""
    apply(C::CliffordTableau{N}, p::PauliBasis{N}) -> (sgn::Int, p′::PauliBasis{N})

Conjugate a Hermitian Pauli string by `C`. Returns the sign (`+1` or `-1`) and the
transformed Hermitian Pauli string such that `C · p · C† = sgn · p′`.
"""
function apply(C::CliffordTableau{N}, p::PauliBasis{N}) where N
    acc = Pauli{N}(ComplexF64(1im^symplectic_phase(p)), Int128(0), Int128(0))
    @inbounds for i in 1:N
        if (p.z >> (i-1)) & 1 == 1
            zi = C.z_to_z[i]; xi = C.z_to_x[i]
            θ_im = (4 - count_ones(zi & xi) % 4) % 4
            s = C.z_sign[i] ? -one(ComplexF64) : one(ComplexF64)
            acc = acc * Pauli{N}(s * (1im)^θ_im, zi, xi)
        end
    end
    @inbounds for i in 1:N
        if (p.x >> (i-1)) & 1 == 1
            zi = C.x_to_z[i]; xi = C.x_to_x[i]
            θ_im = (4 - count_ones(zi & xi) % 4) % 4
            s = C.x_sign[i] ? -one(ComplexF64) : one(ComplexF64)
            acc = acc * Pauli{N}(s * (1im)^θ_im, zi, xi)
        end
    end
    c = coeff(acc)
    # For a Clifford acting on a Hermitian Pauli, the result is ±(Hermitian Pauli).
    sgn = real(c) > 0 ? 1 : -1
    return sgn, PauliBasis{N}(acc.z, acc.x)
end

# ------- Apply: CliffordTableau on PauliSum ----------------------------------

function apply(C::CliffordTableau{N}, ps::PauliSum{N,T}) where {N,T}
    out = PauliSum(N, T)
    for (p, c) in ps
        sgn, p′ = apply(C, p)
        out[p′] = sgn * c
    end
    return out
end

function apply!(C::CliffordTableau{N}, ps::PauliSum{N,T}) where {N,T}
    n = length(ps)
    keys_new = Vector{PauliBasis{N}}(undef, n)
    vals_new = Vector{T}(undef, n)
    i = 0
    for (p, c) in ps
        sgn, p′ = apply(C, p)
        i += 1
        keys_new[i] = p′
        vals_new[i] = sgn * c
    end
    empty!(ps)
    for k in 1:n
        ps[keys_new[k]] = vals_new[k]
    end
    return ps
end

# Primitives dispatch through their tableau representation.
apply(g::CliffordGate, p::PauliBasis{N}) where N = apply(CliffordTableau{N}(g), p)
apply(g::CliffordGate, ps::PauliSum{N,T}) where {N,T} = apply(CliffordTableau{N}(g), ps)
apply!(g::CliffordGate, ps::PauliSum{N,T}) where {N,T} = apply!(CliffordTableau{N}(g), ps)

# ------- Composition ---------------------------------------------------------

"""
    *(C1::CliffordGate, C2::CliffordGate) :: CliffordTableau

Compose two Cliffords: `C1 * C2` applied to a Pauli `P` gives `C1 (C2 P C2†) C1†`.
At least one operand must be a `CliffordTableau` (to fix the qubit count `N`);
primitives can be composed via `CliffordTableau{N}([g1, g2, ...])`.
"""
function Base.:*(C1::CliffordTableau{N}, C2::CliffordTableau{N}) where N
    out = CliffordTableau{N}()
    for i in 1:N
        # X_i image under (C1 ∘ C2) = C1(C2 X_i C2†) C1†
        img_p = PauliBasis{N}(C2.x_to_z[i], C2.x_to_x[i])
        s_init = C2.x_sign[i] ? -1 : 1
        sgn, img = apply(C1, img_p)
        out.x_to_z[i] = img.z
        out.x_to_x[i] = img.x
        out.x_sign[i] = (s_init * sgn) < 0

        img_p = PauliBasis{N}(C2.z_to_z[i], C2.z_to_x[i])
        s_init = C2.z_sign[i] ? -1 : 1
        sgn, img = apply(C1, img_p)
        out.z_to_z[i] = img.z
        out.z_to_x[i] = img.x
        out.z_sign[i] = (s_init * sgn) < 0
    end
    return out
end

Base.:*(C1::CliffordTableau{N}, g2::CliffordGate) where N = C1 * CliffordTableau{N}(g2)
Base.:*(g1::CliffordGate, C2::CliffordTableau{N}) where N = CliffordTableau{N}(g1) * C2

# Two primitives can't be multiplied directly because the qubit count N isn't known.
function Base.:*(g1::CliffordGate, g2::CliffordGate)
    throw(ArgumentError(
        "cannot compose two primitive CliffordGates with `*` because the qubit " *
        "count N isn't known from the operands. Lift one to a tableau first: " *
        "`CliffordTableau{N}(g1) * g2`, or build the full composition with " *
        "`CliffordTableau{N}([g2, g1])` (gates listed in left-to-right order, " *
        "i.e. g2 applied first)."))
end

# ------- Adjoint / inverse ---------------------------------------------------

# Self-inverse primitives
Base.adjoint(g::Hadamard) = g
Base.adjoint(g::PauliX)   = g
Base.adjoint(g::PauliY)   = g
Base.adjoint(g::PauliZ)   = g
Base.adjoint(g::CNOT)     = g
Base.adjoint(g::CZ)       = g
Base.adjoint(g::SWAP)     = g

# Phase-gate inverse pair
Base.adjoint(g::PhaseGate) = PhaseDg(g.q)
Base.adjoint(g::PhaseDg)   = PhaseGate(g.q)

# √X and √Y are not self-inverse and have no dedicated primitive type for the
# adjoint. Users can obtain the inverse by lifting to a tableau:
#     adjoint(CliffordTableau{N}(SqrtX(q)))

# Tableau adjoint: invert via the symplectic-rep relation
# (For a symplectic tableau over GF(2), the inverse is the symplectic transpose.)
function Base.adjoint(C::CliffordTableau{N}) where N
    # Build the inverse by inverting the action on each generator. We compute
    # the inverse generator images by solving: find P such that C·P·C† = X_i (and Z_i).
    # Direct approach: try all 4^k? Too slow. Instead use the symplectic transpose:
    # If we write the tableau as the 2N×2N binary matrix M acting on (x, z) vectors,
    # then the inverse symplectic action is given by M⁻¹ = Λ Mᵀ Λ where Λ is the
    # symplectic form. Signs are recovered by applying the forward tableau to each
    # inverted generator image and reading off the sign correction.

    # Step 1: build the symplectic matrix S of shape (2N, 2N) over GF(2). Convention:
    # rows index generators (X_1..X_N, Z_1..Z_N); columns index image's (x_1..x_N, z_1..z_N).
    S = falses(2N, 2N)
    @inbounds for i in 1:N
        # Row for X_i image
        for k in 1:N
            S[i, k]      = (C.x_to_x[i] >> (k-1)) & 1 == 1   # x_k component
            S[i, N+k]    = (C.x_to_z[i] >> (k-1)) & 1 == 1   # z_k component
        end
        for k in 1:N
            S[N+i, k]    = (C.z_to_x[i] >> (k-1)) & 1 == 1
            S[N+i, N+k]  = (C.z_to_z[i] >> (k-1)) & 1 == 1
        end
    end

    # Step 2: invert S over GF(2). Symplectic inverse: S⁻¹ = Λ Sᵀ Λ where Λ swaps the
    # two halves of the (x, z) vector. So S⁻¹[i, j] = S[σ(j), σ(i)] with σ flipping
    # i ↔ i+N on {1..N} ↔ {N+1..2N}.
    σ(k) = k <= N ? k + N : k - N
    Sinv = falses(2N, 2N)
    @inbounds for i in 1:2N, j in 1:2N
        Sinv[i, j] = S[σ(j), σ(i)]
    end

    # Step 3: build the inverse tableau's bit data from Sinv (without signs yet).
    Cinv = CliffordTableau{N}()
    @inbounds for i in 1:N
        xx = Int128(0); xz = Int128(0); zx = Int128(0); zz = Int128(0)
        for k in 1:N
            if Sinv[i, k];     xx |= _bit(k); end
            if Sinv[i, N+k];   xz |= _bit(k); end
            if Sinv[N+i, k];   zx |= _bit(k); end
            if Sinv[N+i, N+k]; zz |= _bit(k); end
        end
        Cinv.x_to_x[i] = xx; Cinv.x_to_z[i] = xz
        Cinv.z_to_x[i] = zx; Cinv.z_to_z[i] = zz
    end

    # Step 4: fix signs. For each generator G ∈ {X_i, Z_i}, applying C to Cinv's image
    # of G should give G back. If instead we get -G, flip the sign bit in Cinv.
    @inbounds for i in 1:N
        # Test X_i: apply C to Cinv-image of X_i; should equal +X_i
        img = PauliBasis{N}(Cinv.x_to_z[i], Cinv.x_to_x[i])
        sgn, p′ = apply(C, img)
        # p′ should equal X_i (z=0, x=_bit(i))
        @assert p′ == PauliBasis{N}(Int128(0), _bit(i))
        Cinv.x_sign[i] = (sgn < 0)

        img = PauliBasis{N}(Cinv.z_to_z[i], Cinv.z_to_x[i])
        sgn, p′ = apply(C, img)
        @assert p′ == PauliBasis{N}(_bit(i), Int128(0))
        Cinv.z_sign[i] = (sgn < 0)
    end

    return Cinv
end

# ------- KetSum action (Schrödinger picture) ---------------------------------

"""
    apply(g::CliffordGate, ks::KetSum) -> KetSum

Apply a Clifford gate to a `KetSum` (Schrödinger picture). Specialized fast paths
exist for each primitive; for `CliffordTableau`, decomposition into primitives is
not yet implemented (apply the source primitive sequence directly).
"""
apply(g::CliffordGate, ks::KetSum) = error("apply on KetSum is implemented per-primitive; for CliffordTableau, apply the constituent primitive gates directly")

# --- diagonal primitives (multiply each amplitude by a state-dependent phase) ---

function apply(g::PauliZ, ks::KetSum{N,T}) where {N,T}
    out = KetSum(N, T=T)
    m = _bit(g.q)
    for (k, c) in ks
        s = (k.v & m) != 0 ? -one(T) : one(T)
        out[k] = s * c
    end
    return out
end

function apply(g::PhaseGate, ks::KetSum{N,T}) where {N,T}
    Tc = promote_type(T, ComplexF64)
    out = KetSum(N, T=Tc)
    m = _bit(g.q)
    for (k, c) in ks
        out[k] = (k.v & m) != 0 ? Tc(c * 1im) : Tc(c)
    end
    return out
end

function apply(g::PhaseDg, ks::KetSum{N,T}) where {N,T}
    Tc = promote_type(T, ComplexF64)
    out = KetSum(N, T=Tc)
    m = _bit(g.q)
    for (k, c) in ks
        out[k] = (k.v & m) != 0 ? Tc(c * (-1im)) : Tc(c)
    end
    return out
end

function apply(g::CZ, ks::KetSum{N,T}) where {N,T}
    out = KetSum(N, T=T)
    mc = _bit(g.c); mt = _bit(g.t)
    for (k, c) in ks
        s = ((k.v & mc) != 0) && ((k.v & mt) != 0) ? -one(T) : one(T)
        out[k] = s * c
    end
    return out
end

# --- permutation primitives -------------------------------------------------

function apply(g::PauliX, ks::KetSum{N,T}) where {N,T}
    out = KetSum(N, T=T)
    m = _bit(g.q)
    for (k, c) in ks
        out[Ket{N}(k.v ⊻ m)] = c
    end
    return out
end

function apply(g::PauliY, ks::KetSum{N,T}) where {N,T}
    Tc = promote_type(T, ComplexF64)
    out = KetSum(N, T=Tc)
    m = _bit(g.q)
    for (k, c) in ks
        # Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
        phase = (k.v & m) != 0 ? -1im : 1im
        out[Ket{N}(k.v ⊻ m)] = Tc(c * phase)
    end
    return out
end

function apply(g::CNOT, ks::KetSum{N,T}) where {N,T}
    out = KetSum(N, T=T)
    mc = _bit(g.c); mt = _bit(g.t)
    for (k, c) in ks
        v = (k.v & mc) != 0 ? (k.v ⊻ mt) : k.v
        out[Ket{N}(v)] = c
    end
    return out
end

function apply(g::SWAP, ks::KetSum{N,T}) where {N,T}
    out = KetSum(N, T=T)
    ma = _bit(g.a); mb = _bit(g.b)
    for (k, c) in ks
        bita = (k.v & ma) != 0
        bitb = (k.v & mb) != 0
        v = k.v
        if bita != bitb
            v ⊻= (ma | mb)
        end
        out[Ket{N}(v)] = c
    end
    return out
end

# --- branching primitives (each Ket spawns two output Kets) -----------------

function apply(g::Hadamard, ks::KetSum{N,T}) where {N,T}
    Tc = promote_type(T, ComplexF64)
    out = KetSum(N, T=Tc)
    inv_sqrt2 = Tc(1 / sqrt(2))
    m = _bit(g.q)
    for (k, c) in ks
        bit = (k.v & m) != 0
        k0 = Ket{N}(k.v & ~m)
        k1 = Ket{N}(k.v |  m)
        amp = Tc(c) * inv_sqrt2
        # H|0⟩ = |0⟩+|1⟩, H|1⟩ = |0⟩-|1⟩, normalized
        out[k0] = get(out, k0, zero(Tc)) + amp
        out[k1] = get(out, k1, zero(Tc)) + (bit ? -amp : amp)
    end
    return out
end

function apply(g::SqrtX, ks::KetSum{N,T}) where {N,T}
    Tc = promote_type(T, ComplexF64)
    out = KetSum(N, T=Tc)
    a = Tc((1 + 1im) / 2)   # diagonal element
    b = Tc((1 - 1im) / 2)   # off-diagonal element (same for both rows)
    m = _bit(g.q)
    for (k, c) in ks
        k_same = k
        k_flip = Ket{N}(k.v ⊻ m)
        out[k_same] = get(out, k_same, zero(Tc)) + Tc(c) * a
        out[k_flip] = get(out, k_flip, zero(Tc)) + Tc(c) * b
    end
    return out
end

function apply(g::SqrtY, ks::KetSum{N,T}) where {N,T}
    Tc = promote_type(T, ComplexF64)
    out = KetSum(N, T=Tc)
    inv_sqrt2 = Tc(1 / sqrt(2))
    m = _bit(g.q)
    # √Y = (I - iY)/√2.  Action: √Y|0⟩ = (|0⟩+|1⟩)/√2,  √Y|1⟩ = (-|0⟩+|1⟩)/√2
    for (k, c) in ks
        bit = (k.v & m) != 0
        k0 = Ket{N}(k.v & ~m)
        k1 = Ket{N}(k.v |  m)
        amp_c = Tc(c) * inv_sqrt2
        if bit
            out[k0] = get(out, k0, zero(Tc)) - amp_c
            out[k1] = get(out, k1, zero(Tc)) + amp_c
        else
            out[k0] = get(out, k0, zero(Tc)) + amp_c
            out[k1] = get(out, k1, zero(Tc)) + amp_c
        end
    end
    return out
end

# A bare `Ket` is promoted to a single-term `KetSum` for the primitive apply paths.
# CliffordTableau on KetSum is not implemented; falls through to the generic error
# defined at the top of the KetSum action section.
apply(g::CliffordGate, k::Ket{N}) where N = apply(g, KetSum(k; T=ComplexF64))

# ------- compose helper ------------------------------------------------------

"""
    compose(gates::AbstractVector{<:CliffordGate}, ::Val{N}) -> CliffordTableau{N}

Build the tableau representing the sequence `gates[end] ∘ ⋯ ∘ gates[2] ∘ gates[1]`
(applied left-to-right). Equivalent to `CliffordTableau{N}(gates)`.
"""
compose(gates::AbstractVector{<:CliffordGate}, ::Val{N}) where N = CliffordTableau{N}(gates)

# ------- Matrix → tableau ----------------------------------------------------

"""
    CliffordTableau{N}(U::AbstractMatrix, qs::Vector{Int}=collect(1:N); atol=1e-8)

Identify a Clifford tableau from a dense unitary matrix `U` (size `2^K × 2^K` with
`K = length(qs)`) acting on qubits `qs` of an N-qubit system. `U` must be Clifford
up to a global phase; throws `ArgumentError` if any generator's image is not a
±Pauli within `atol`.

Practical for small `K` (≤ 4 or so); for larger blocks the exhaustive Pauli match
becomes expensive.
"""
function CliffordTableau{N}(U::AbstractMatrix, qs::Vector{Int}; atol::Real=1e-8) where N
    K = length(qs)
    K > 0 || throw(ArgumentError("qs must be non-empty"))
    all(1 .<= qs .<= N) || throw(ArgumentError("qs must be in 1:$N"))
    allunique(qs) || throw(ArgumentError("qs entries must be unique"))
    dim = 1 << K
    size(U, 1) == dim && size(U, 2) == dim || throw(ArgumentError(
        "U must be $dim×$dim for K=$K target qubits, got $(size(U))"))
    isapprox(U * U', Matrix{ComplexF64}(I, dim, dim); atol=atol) ||
        throw(ArgumentError("U is not unitary within atol=$atol"))

    # Precompute dense matrices and signed lookup tables for K-qubit Paulis.
    pauli_matrices = Vector{Matrix{ComplexF64}}()
    pauli_keys = Vector{Tuple{Int128, Int128}}()  # (z, x)
    for zb in Int128(0):(Int128(1) << K - Int128(1))
        for xb in Int128(0):(Int128(1) << K - Int128(1))
            p = PauliBasis{K}(zb, xb)
            push!(pauli_matrices, ComplexF64.(Matrix(p)))
            push!(pauli_keys, (zb, xb))
        end
    end

    function identify(M::AbstractMatrix)
        for idx in eachindex(pauli_matrices)
            Mp = pauli_matrices[idx]
            if isapprox(M, Mp; atol=atol)
                z, x = pauli_keys[idx]
                return z, x, +1
            elseif isapprox(M, -Mp; atol=atol)
                z, x = pauli_keys[idx]
                return z, x, -1
            end
        end
        throw(ArgumentError("matrix image is not ±Pauli within atol=$atol — U is not a Clifford"))
    end

    Uc = ComplexF64.(U)
    Ud = adjoint(Uc)
    small = CliffordTableau{K}()
    for j in 1:K
        # Image of X_j
        Xj = ComplexF64.(Matrix(PauliBasis{K}(Int128(0), _bit(j))))
        Mj = Uc * Xj * Ud
        z, x, sgn = identify(Mj)
        small.x_to_z[j] = z; small.x_to_x[j] = x
        small.x_sign[j] = sgn < 0

        # Image of Z_j
        Zj = ComplexF64.(Matrix(PauliBasis{K}(_bit(j), Int128(0))))
        Mj = Uc * Zj * Ud
        z, x, sgn = identify(Mj)
        small.z_to_z[j] = z; small.z_to_x[j] = x
        small.z_sign[j] = sgn < 0
    end

    # Lift to N qubits on target positions qs.
    big = CliffordTableau{N}()
    for j in 1:K
        q = qs[j]
        big.x_to_z[q] = _remap_bits(small.x_to_z[j], qs, K)
        big.x_to_x[q] = _remap_bits(small.x_to_x[j], qs, K)
        big.x_sign[q] = small.x_sign[j]
        big.z_to_z[q] = _remap_bits(small.z_to_z[j], qs, K)
        big.z_to_x[q] = _remap_bits(small.z_to_x[j], qs, K)
        big.z_sign[q] = small.z_sign[j]
    end
    return big
end

CliffordTableau{N}(U::AbstractMatrix; atol::Real=1e-8) where N =
    CliffordTableau{N}(U, collect(1:N); atol=atol)

@inline function _remap_bits(bits::Int128, qs::Vector{Int}, K::Int)
    out = Int128(0)
    @inbounds for k in 1:K
        if (bits >> (k-1)) & 1 == 1
            out |= Int128(1) << (qs[k] - 1)
        end
    end
    return out
end

# ------- Tableau → dense matrix ---------------------------------------------

"""
    Matrix(C::CliffordTableau{N}) -> Matrix{ComplexF64}

Dense `2^N × 2^N` unitary realising `C` in the computational basis.

The tableau encodes a Clifford only up to a global phase, so the returned matrix is
unitary and matches `C` only **up to a single global phase factor**. For physical
use (channels, expectation values, comparing two Cliffords via `U₁·U₂'` ≈ scalar `I`)
this is sufficient.

Cost is O(`2^N` · support(C|0⟩)) ≤ O(`4^N`). Practical for `N` up to ~10–12.
"""
function Base.Matrix(C::CliffordTableau{N}) where N
    dim = 1 << N
    ψ0 = _stabilizer_state_from_tableau(C, N)
    U = zeros(ComplexF64, dim, dim)
    for c in Int128(0):(Int128(dim) - Int128(1))
        sgn, P_c = apply(C, PauliBasis{N}(Int128(0), c))
        ψc = P_c * ψ0
        for (k_, v_) in ψc
            U[Int(k_.v) + 1, Int(c) + 1] = sgn * v_
        end
    end
    return U
end

# Internal: compute C|0⟩ as a KetSum via iterated stabilizer projection.
# Returns a unit-norm KetSum; the global phase is determined by the first reference
# basis state that has nonzero overlap, fixing the one global-phase choice for `Matrix`.
function _stabilizer_state_from_tableau(C::CliffordTableau{N}, n::Int) where N
    paulis = Vector{PauliBasis{N}}(undef, n)
    signs  = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        paulis[i] = PauliBasis{N}(C.z_to_z[i], C.z_to_x[i])
        signs[i]  = C.z_sign[i] ? -1 : 1
    end
    full = Int128(1) << n
    @inbounds for v0 in Int128(0):(full - Int128(1))
        ψ = _project_onto_stabilizers(paulis, signs, Ket{N}(v0), n)
        if !isempty(ψ)
            nrm = sqrt(sum(abs2, values(ψ)))
            for k_ in keys(ψ)
                ψ[k_] /= nrm
            end
            return ψ
        end
    end
    error("Could not find a basis state with non-zero overlap with C|0⟩ — should not happen for a valid Clifford")
end

function _project_onto_stabilizers(paulis::Vector{PauliBasis{N}}, signs::Vector{Int},
                                   ref::Ket{N}, n::Int) where N
    ψ = KetSum(ref; T=ComplexF64)
    half = ComplexF64(0.5)
    @inbounds for i in 1:n
        Pψ = paulis[i] * ψ
        sgn = ComplexF64(signs[i])
        new_ψ = KetSum(N; T=ComplexF64)
        for (k_, v_) in ψ
            new_ψ[k_] = get(new_ψ, k_, zero(ComplexF64)) + v_ * half
        end
        for (k_, v_) in Pψ
            new_ψ[k_] = get(new_ψ, k_, zero(ComplexF64)) + sgn * v_ * half
        end
        for k_ in collect(keys(new_ψ))
            if abs(new_ψ[k_]) < 1e-14
                delete!(new_ψ, k_)
            end
        end
        ψ = new_ψ
        isempty(ψ) && return ψ
    end
    return ψ
end

# ------- Random Clifford sampling -------------------------------------------

"""
    rand([rng,] CliffordTableau{N}; depth=max(200, 20*N^2)) -> CliffordTableau{N}

Sample a random N-qubit Clifford as a `CliffordTableau`. The current implementation
composes `depth` uniformly-chosen primitive gates (`Hadamard`, `PhaseGate`, `PhaseDg`,
`CNOT` on random qubits) and then samples uniform random sign bits. With sufficient
depth this **approximates** the uniform distribution over the Clifford group
(the random walk mixes quickly), but is not provably uniform.

For applications requiring provable uniformity (e.g. randomized benchmarking,
unitary t-design protocols), a Bravyi-Maslov sampler would be preferable; this is a
straightforward extension to implement on top of the current building blocks.
"""
Base.rand(::Type{CliffordTableau{N}}; depth::Integer=max(200, 20 * N * N)) where N =
    rand(Random.default_rng(), CliffordTableau{N}; depth=depth)

function Base.rand(rng::AbstractRNG, ::Type{CliffordTableau{N}};
                   depth::Integer=max(200, 20 * N * N)) where N
    N >= 1 || throw(ArgumentError("N must be ≥ 1"))
    C = CliffordTableau{N}()
    one_qubit_kinds = (Hadamard, PhaseGate, PhaseDg)
    for _ in 1:depth
        # For N == 1 we can only apply single-qubit gates.
        if N == 1 || rand(rng) < 0.5
            kind = one_qubit_kinds[rand(rng, 1:3)]
            q = rand(rng, 1:N)
            C = CliffordTableau{N}(kind(q)) * C
        else
            c = rand(rng, 1:N)
            t = rand(rng, 1:N-1)
            t += t >= c ? 1 : 0   # ensures t != c
            C = CliffordTableau{N}(CNOT(c, t)) * C
        end
    end
    Random.rand!(rng, C.x_sign)
    Random.rand!(rng, C.z_sign)
    return C
end

