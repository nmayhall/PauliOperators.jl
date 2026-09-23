# =============================================================================
#  Majorana / Wick mean-field truncation (isolated; fermionic analog of the spin
#  mean-field fold in mean_field.jl).
#
#  Truncates by MAJORANA weight (the fermionic locality measure), folding around
#  a computational-basis determinant reference ψ (Fock occupation) via Wick's
#  theorem. Preserves ⟨ψ|O|ψ⟩ for every k and is exact when k ≥ majorana_weight.
#
#  Reference must be a computational-basis Ket (a Slater determinant); by
#  particle-hole symmetry |0…0⟩ is itself HF, so callers PH/orbital-transform so
#  the determinant sits at a basis state (mirroring the spin fold's neel_transform).
#
#  Conventions (matching jordan_wigner in transformations.jl):
#    γ_{2f-1} = X_f Z_{<f},  γ_{2f} = Y_f Z_{<f}
#  On a determinant the only nonzero Wick contraction is the within-mode pair
#    ⟨γ_{2f-1} γ_{2f}⟩_ψ = i(1 - 2 n_f),
#  and because same-mode Majoranas have consecutive (adjacent) indices, every
#  contraction sign is +1 — so the fold is: expand ∏_modes (mean + fluctuation)
#  and keep terms with ≤ k Majoranas. Verified against brute force.
# =============================================================================

# γ_{idx} as a Pauli{N} (built by multiplication so the JW phase is exact).
function _mmf_gamma(idx::Int, N::Int)
    f = (idx + 1) ÷ 2
    g = isodd(idx) ? Pauli(N, X=[f]) : Pauli(N, Y=[f])
    for q in 1:f-1
        g = Pauli(N, Z=[q]) * g
    end
    return g
end

# strict-above suffix parity of the x-bits (t_g = XOR_{f>g} x_f)
@inline function _mmf_tmask(x::Int128)
    s = unsigned(x)
    s ⊻= s >> 1; s ⊻= s >> 2; s ⊻= s >> 4; s ⊻= s >> 8
    s ⊻= s >> 16; s ⊻= s >> 32; s ⊻= s >> 64
    return Int128(s >> 1)
end

# Pauli (z,x) → Majorana presence masks: a_f = γ_{2f-1} present, b_f = γ_{2f} present.
@inline function _mmf_ab(z::Int128, x::Int128)
    t = _mmf_tmask(x)
    b = z ⊻ t
    a = x ⊻ b
    return a, b
end

_mmf_ident(N::Int) = (o = PauliSum(N, ComplexF64);
                      o[PauliBasis{N}(Int128(0), Int128(0))] = 1.0 + 0im; o)

# Add the operator for one choice of kept (uncontracted) pair-modes into `result`.
# Built in ascending mode order so all JW signs/phases come from Pauli `*`.
function _mmf_add_term!(result::PauliSum{N,ComplexF64}, cc::ComplexF64, ψ::Ket{N},
                        a::Int128, b::Int128, pairs::Vector{Int},
                        kept::Vector{Bool}) where {N}
    op = _mmf_ident(N)
    op[PauliBasis{N}(Int128(0), Int128(0))] = cc
    @inbounds for f in 1:N
        af = (a >> (f-1)) & 1
        bf = (b >> (f-1)) & 1
        if (af ⊻ bf) == 1                      # single Majorana
            op = op * _mmf_gamma(af == 1 ? 2*f-1 : 2*f, N)
        elseif af == 1 && bf == 1              # pair mode
            pidx = findfirst(==(f), pairs)::Int
            mean = im * (1 - 2 * Int((ψ.v >> (f-1)) & 1))
            if kept[pidx]                       # keep fluctuation (γγ − mean)
                gg = PauliSum(_mmf_gamma(2*f-1, N) * _mmf_gamma(2*f, N))
                op = op * (gg - mean * _mmf_ident(N))
            else                                # contract to the mean
                op = op * mean
            end
        end
    end
    sum!(result, op)
    return nothing
end

# enumerate size-t subsets of 1:n, invoking f(kept::Vector{Bool}) for each
function _mmf_foreach_kept(f::F, n::Int, t::Int) where {F}
    kept = fill(false, n)          # Vector{Bool} (not BitVector)
    _mmf_kept_rec(f, kept, 1, t, n)
end
function _mmf_kept_rec(f::F, kept::Vector{Bool}, start::Int, remaining::Int, n::Int) where {F}
    if remaining == 0
        f(kept); return
    end
    for i in start:(n - remaining + 1)
        kept[i] = true
        _mmf_kept_rec(f, kept, i+1, remaining-1, n)
        kept[i] = false
    end
    return
end

"""
    majorana_mean_field_factorize(pb::PauliBasis{N}, c, ψ::Ket{N}, k::Int) -> PauliSum{N,ComplexF64}

Order-`k` Majorana/Wick mean-field factorization of `c · pb` around the
computational-basis determinant `ψ`. Replaces `pb` with a sum of Pauli strings
of Majorana weight ≤ `k`, preserving `⟨ψ|·|ψ⟩` and exact when `k ≥ majorana_weight(pb)`.
"""
function majorana_mean_field_factorize(pb::PauliBasis{N}, c, ψ::Ket{N}, k::Int) where {N}
    result = PauliSum(N, ComplexF64)
    a, b = _mmf_ab(pb.z, pb.x)
    pairs = Int[f for f in 1:N if (((a>>(f-1))&1)==1) && (((b>>(f-1))&1)==1)]
    Q = count(f -> ((((a>>(f-1))&1)) ⊻ (((b>>(f-1))&1))) == 1, 1:N)   # #singles
    Q > k && return result                        # off-diagonal support exceeds budget
    # JW phase relating the ascending Majorana product to pb
    pfull = Pauli(N)
    @inbounds for f in 1:N
        ((a>>(f-1))&1)==1 && (pfull = pfull * _mmf_gamma(2*f-1, N))
        ((b>>(f-1))&1)==1 && (pfull = pfull * _mmf_gamma(2*f, N))
    end
    cc = ComplexF64(c) / coeff(pfull)
    budget_pairs = (k - Q) ÷ 2                     # each kept pair costs 2 Majoranas
    np = length(pairs)
    for t in 0:min(budget_pairs, np)
        _mmf_foreach_kept(np, t) do kept
            _mmf_add_term!(result, cc, ψ, a, b, pairs, kept)
        end
    end
    return result
end

"""
    majorana_mean_field_factorize!(O::PauliSum{N}, ψ::Ket{N}, k::Int)

In-place replacement of every term of `O` with `majorana_weight > k` by its
order-`k` Majorana mean-field factorization around `ψ`.
"""
function majorana_mean_field_factorize!(O::PauliSum{N,T}, ψ::Ket{N}, k::Int) where {N,T}
    high = [pb for (pb, _) in O if majorana_weight(pb) > k]
    for pb in high
        c = pop!(O, pb)
        fold = majorana_mean_field_factorize(pb, c, ψ, k)
        for (p2, c2) in fold
            O[p2] = get(O, p2, zero(T)) + convert(T, c2)
        end
    end
    return O
end

"""
    MajoranaMeanFieldTruncation(max_weight::Int, reference::Ket{N})

Expectation-preserving order-`k` (`k = max_weight`) fermionic mean-field
truncation: every term with Majorana weight > k is folded (via Wick's theorem
around the computational-basis determinant `reference`) into a sum of terms with
Majorana weight ≤ k, preserving `⟨reference|O|reference⟩`. The fermionic analog
of [`MeanFieldTruncation`](@ref), using Majorana weight (JW-aware locality)
rather than Pauli weight. `reference` must be a computational-basis `Ket` (Slater
determinant); PH/orbital-transform so it sits at a basis state.

Non-"pure-drop" (modifies survivors), so `truncate!` routes it through the
measured before/after correction path.
"""
struct MajoranaMeanFieldTruncation{N} <: TruncationStrategy
    max_weight::Int
    reference::Ket{N}
end

function _apply!(O::PauliSum{N}, s::MajoranaMeanFieldTruncation{N}) where {N}
    return majorana_mean_field_factorize!(O, s.reference, s.max_weight)
end

# SparsePauliVector path: fold high-Majorana-weight terms into a PauliSum,
# drop them from the flat buffer (Majorana-weight clip), merge the replacements
# back. (Uses a Dict internally; a zero-alloc staging variant is a follow-up.)
function _apply!(v::SparsePauliVector{N,W,T}, s::MajoranaMeanFieldTruncation{N}) where {N,W,T}
    folded = PauliSum(N, ComplexF64)
    @inbounds for i in 1:v.n
        pb = _unpack(PauliBasis{N}, v.z[i], v.x[i])
        majorana_weight(pb) > s.max_weight || continue
        sum!(folded, majorana_mean_field_factorize(pb, v.c[i], s.reference, s.max_weight))
    end
    majorana_weight_clip!(v, s.max_weight)
    isempty(folded) || sum!(v, SparsePauliVector(folded; T=T))
    return v
end
