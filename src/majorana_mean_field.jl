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
#  contraction sign is +1.
#
#  ANALYTIC KERNEL.  Split the modes of a Pauli string into
#    - "single" modes carrying exactly one Majorana (γ_{2f-1} XOR γ_{2f}), and
#    - "pair"   modes carrying both (γ_{2f-1} γ_{2f} = i Z_f).
#  The pair factors are DIAGONAL and mutually COMMUTING (each is i Z_f), so the
#  order-k fold factorizes exactly:
#      fold(c·pb) = γ_Q · [ order-(budget) spin fold of the pair-mode Z-string ],
#  where γ_Q is the fixed product of the single-mode Majoranas and
#  budget = ⌊(k − #singles)/2⌋ (each kept pair costs 2 Majoranas). The global JW
#  phase i^{#pairs} that relates ∏γ to the Pauli basis cancels EXACTLY against the
#  1/coeff normalization, leaving purely the real spin-fold coefficients. So the
#  whole thing reduces to bit-ops plus one call to the allocation-free spin kernel
#  (`_mean_field_accumulate!`, mean_field.jl) on the pair-mode Z-string, with each
#  emitted Z-subset re-based onto γ_Q's (z,x). Verified term-by-term against the
#  brute-force Wick expansion and against the earlier multiply-based reference.
# =============================================================================

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

# Sink wrapper: re-base each spin-fold output (a Z-string on the pair modes) onto
# the fixed single-mode Majorana product γ_Q by XORing its (z,x) offsets. The pair
# modes and single modes are disjoint, and Z_T commutes with γ_Q with no phase, so
# this XOR is the exact product γ_Q · Z_T (see the header). Forwards to the real
# target sink (`PauliSum` or `SparsePauliVector`), so it inherits their `_mf_emit!`.
struct _MMFSink{S}
    target::S
    z_off::Int128     # γ_Q z-bits (pb.z with the pair-mode Z removed)
    x_off::Int128     # γ_Q x-bits (= pb.x; single modes only)
end
@inline function _mf_emit!(s::_MMFSink, z::Int128, x::Int128, c)
    _mf_emit!(s.target, z ⊻ s.z_off, x ⊻ s.x_off, c)
    return nothing
end

# Emit the order-`k` Majorana mean-field factorization of `c · pb` around `ψ` into
# `sink` (a `PauliSum` or `SparsePauliVector`). Allocation-free core: no Pauli
# multiplication, no γ construction — just bit-ops plus one spin-fold pass over the
# pair-mode Z-string, re-based onto γ_Q by `_MMFSink`.
function _majorana_mean_field_accumulate!(sink, pb::PauliBasis{N},
                                          c::T, ψ::Ket{N}, k::Int) where {N,T}
    a, b   = _mmf_ab(pb.z, pb.x)
    P_mask = a & b                       # pair modes: both Majoranas present (= i Z_f)
    Q      = count_ones(a ⊻ b)           # single modes: exactly one Majorana each
    Q > k && return sink                 # #singles alone exceeds the Majorana budget
    budget = (k - Q) ÷ 2                 # each kept pair costs 2 Majoranas
    z_Q    = pb.z ⊻ P_mask               # γ_Q z-bits: pb.z with pair-mode Z removed
    Z_P    = PauliBasis{N}(P_mask, Int128(0))
    wrapped = _MMFSink(sink, z_Q, pb.x)
    # Spin-fold the pair-mode Z-string to weight ≤ budget around ψ; the wrapper
    # re-bases each Z-subset onto γ_Q and carries c through unchanged.
    _mean_field_accumulate!(wrapped, Z_P, c, ψ, budget)
    return sink
end

"""
    majorana_mean_field_factorize(pb::PauliBasis{N}, c, ψ::Ket{N}, k::Int) -> PauliSum{N,ComplexF64}

Order-`k` Majorana/Wick mean-field factorization of `c · pb` around the
computational-basis determinant `ψ`. Replaces `pb` with a sum of Pauli strings
of Majorana weight ≤ `k`, preserving `⟨ψ|·|ψ⟩` and exact when `k ≥ majorana_weight(pb)`.
"""
function majorana_mean_field_factorize(pb::PauliBasis{N}, c, ψ::Ket{N}, k::Int) where {N}
    out = PauliSum(N, ComplexF64)
    _majorana_mean_field_accumulate!(out, pb, ComplexF64(c), ψ, k)
    return out
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
        _majorana_mean_field_accumulate!(O, pb, c, ψ, k)
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

# SparsePauliVector path: stage each high-Majorana-weight term's fold straight into
# the flat append region (Dict-free, via the append-region `_mf_emit!` sink), drop
# the folded originals with a Majorana-weight clip, then sort-merge the appends back
# in with dedup+accumulation — mirroring the `MeanFieldTruncation` SPV pass. Staging
# reads the live buffer and writes only the append arrays, so it is safe before the clip.
function _apply!(v::SparsePauliVector{N,W,T}, s::MajoranaMeanFieldTruncation{N}) where {N,W,T}
    k = s.max_weight
    ψ = s.reference
    @inbounds for i in 1:v.n
        pb = _unpack(PauliBasis{N}, v.z[i], v.x[i])
        majorana_weight(pb) > k || continue
        _majorana_mean_field_accumulate!(v, pb, v.c[i], ψ, k)     # stage folds into v.a*
    end
    v.an == 0 && return v                                         # nothing folded
    majorana_weight_clip!(v, k)                                   # drop the folded terms
    merge_pending!(v)                                             # sort+merge appends into live
    return v
end
