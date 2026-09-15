# Analytic gradients of rotation-circuit expectation values (Krotov-style
# adjoint sweep).
#
# Cost:  C(θ) = tr(H_M ρ),   H_M = U_M'⋯U_1' H U_1⋯U_M,   U_k = exp(-iθ_k/2 G_k)
# (the `evolve` sequence convention). With A_i = U_i'⋯U_1' H U_1⋯U_i,
#
#   ∂C/∂θ_i = (i/2) tr( [G_i, A_i] ρ_i ),    ρ_i = U_{i+1}⋯U_M ρ U_M'⋯U_{i+1}'
#
# and both A and ρ obey the same reverse recursion, conjugation by U_i — i.e.
# `evolve!(·, G_i, -θ_i)`. The full gradient therefore costs one forward pass
# of H plus one reverse pass each of H and ρ.
#
# ρ is represented by the coefficient vector r_P = tr(Pρ) projected onto the
# REACHABLE set R_M of the circuit: R_0 = supp(H), and R_k adds to R_{k-1}
# the partner G_k·P of every P ∈ R_{k-1} anticommuting with G_k. Reachable
# sets are closed under partnering with their own generators and only grow,
# so supp((i/2)[G_i, A_i]) ⊆ R_i and every forward image stays inside R_M —
# the projection is exact UNCONDITIONALLY (no genericity assumption): angles
# where coefficients cancel exactly (θ = 0, odd multiples of π/2, symmetry-
# driven interference) shrink the *support* but never the reachable set.
#
# TRUNCATION (the production regime — large-N runs always truncate): the
# forward pass never applies the standard `truncate!`. Instead it filters
# only terms not already in the sum (`_evolve_monotone!`), so a Pauli,
# once introduced, is never deleted — the monotone-growth assumption holds
# by construction and the reachable-set logic above survives truncation
# unchanged. Because each per-step truncation is then a fixed projection
# T_k onto the kept set S_k, and projections are self-adjoint under the
# trace pairing, the reverse recursion "project onto S_k, then un-evolve"
# is the exact adjoint of the truncated forward map. The one remaining
# approximation is reconstructing the forward intermediates A_i by
# un-evolution (truncation is not invertible); its error is first order in
# the coefficients of the terms truncation removed, so the gradient matches finite
# differences of the truncated cost to O(threshold) — and exactly as the
# threshold → 0.
#
# The reverse evolution of both H and ρ is always constrained to the
# subspace built by the forward pass; two projection modes are implemented
# for comparison (`method=`), sharing the same forward map:
#
#   :dynamic   The forward pass records the step at which each Pauli first
#              appears, so S_k = {P : intro[P] ≤ k} is the per-step kept
#              set. The reverse sweep projects A and ρ onto S_k as it goes
#              — the exact adjoint of the truncated forward map — and the
#              reverse passes mirror the forward sizes: total ≈ 3x one cost
#              evaluation.
#
#   :static    The reverse evolution is constrained only to the FINAL
#              subspace S_M (no per-step record needed). Reverse passes run
#              at full |S_M| size throughout — simpler, slower. Under
#              truncation this differentiates a slightly different map: it
#              keeps reverse flow through Paulis that had not yet appeared
#              at earlier steps.
#
# Both are identical and exact without truncation.

"""
    expectation_value(H::PauliSum{N,T}, generators::Vector{PauliBasis{N}},
                      angles::Vector{<:Real}, ψ::Ket{N};
                      truncation::TruncationStrategy=NoTruncation(),
                      monotone::Bool=false)

Expectation value of a Hamiltonian evolved through a Pauli-rotation circuit:

    C(θ) = ⟨ψ| U_M'⋯U_1' H U_1⋯U_M |ψ⟩,    U_k = exp(-iθ_k/2 G_k)

matching the `evolve` sequence convention. By default equivalent to
`expectation_value(evolve(H, generators, angles; truncation), ψ)`.

With `monotone=true`, `truncation` is applied only to terms that are not
already in the sum, so no Pauli is ever deleted once present. This is the
same truncated cost function `expectation_value_gradient` differentiates —
use this variant when comparing against gradients (e.g. finite differences).
"""
function expectation_value(H::PauliSum{N,T}, generators::Vector{PauliBasis{N}},
                           angles::Vector{<:Real}, ψ::Ket{N};
                           truncation::TruncationStrategy=NoTruncation(),
                           monotone::Bool=false) where {N,T}
    A = monotone ? _forward_monotone(H, generators, angles, truncation) :
                   evolve(H, generators, angles; truncation=truncation)
    return expectation_value(A, ψ)
end

"""
    expectation_value_gradient(H::PauliSum{N,T}, generators::Vector{PauliBasis{N}},
                               angles::Vector{<:Real}, state;
                               truncation::TruncationStrategy=NoTruncation(),
                               method::Symbol=:dynamic)
        -> (cost::Float64, grad::Vector{Float64})

Cost C(θ) = tr(U_M'⋯U_1' H U_1⋯U_M ρ) with U_k = exp(-iθ_k/2 G_k) (the
`evolve` sequence convention) and its full analytic gradient ∂C/∂θ_k,
computed with a Krotov-style adjoint sweep: one forward pass of H, then one
reverse pass each of H and of the state's Pauli coefficients projected onto
the circuit's reachable Pauli set.

`state` is either a `Ket{N}` (ρ = |ψ⟩⟨ψ|) or a `PauliSum{N}` whose values
are `r_P = tr(Pρ)` — the normalization `expectation_value` produces — so
that `cost = Σ_P h_P r_P = tr(H_M ρ)`. A caller holding the density-operator
expansion ρ = Σ_P c_P P should pass values `2^N·c_P`.

Truncation: the forward pass applies `truncation` only to terms that are
not already in the sum, so no Pauli is ever deleted once present and the
monotone-growth assumption underlying the state projection holds by
construction. The returned cost is that of this monotone-truncated
evolution (evaluate it independently with `expectation_value(...;
truncation, monotone=true)`), and the gradient matches finite differences
of that cost to first order in the truncation threshold — exactly, at *all*
angles (including exact 0, π/2, and interference-cancellation points), as
the threshold → 0 or for the default `NoTruncation()`.

`method` selects how the reverse evolution is constrained to the forward
subspace (identical forward map; identical results without truncation):
- `:dynamic` (default): the forward pass records when each Pauli first
  appears, and the reverse sweep projects onto the per-step subspace as it
  goes — the exact adjoint of the truncated forward map — so reverse passes
  mirror the forward sizes, total ≈ 3x one cost evaluation.
- `:static`: the reverse evolution is constrained only to the final
  subspace of the forward pass (no per-step record); reverse passes run at
  full final-subspace size (slower), and under truncation this keeps
  reverse flow through Paulis that had not yet appeared at earlier steps.
"""
function expectation_value_gradient(H::PauliSum{N,T}, generators::Vector{PauliBasis{N}},
                                    angles::Vector{<:Real},
                                    state::Union{Ket{N}, PauliSum{N}};
                                    truncation::TruncationStrategy=NoTruncation(),
                                    method::Symbol=:dynamic) where {N,T}
    if method === :dynamic
        A, intro = _forward(H, generators, angles, truncation)
        ρ = _project_state(A, state)
        return _reverse_sweep_dynamic!(A, ρ, intro, generators, angles)
    elseif method === :static
        A = _forward_monotone(H, generators, angles, truncation)
        ρ = _project_state(A, state)
        return _reverse_sweep_static!(A, ρ, generators, angles)
    else
        throw(ArgumentError("method must be :dynamic or :static, got :$method"))
    end
end

# ρ̃ = Σ_{P ∈ keys(A)} tr(Pρ)·P. For a basis ket only diagonal Paulis
# contribute; for a general state all of keys(A) is kept (off-diagonal
# tr(Pρ) ≠ 0 in general).
function _project_state(A::PauliSum{N}, ψ::Ket{N}) where {N}
    ρ = PauliSum(N, ComplexF64)
    for (p, _) in A
        p.x == 0 || continue
        ρ[p] = expectation_value(p, ψ)
    end
    return ρ
end

function _project_state(A::PauliSum{N}, ρin::PauliSum{N}) where {N}
    ρ = PauliSum(N, ComplexF64)
    for (p, _) in A
        r = get(ρin, p, zero(valtype(ρin)))
        iszero(r) || (ρ[p] = r)
    end
    return ρ
end

function _check_lengths(generators, angles)
    length(generators) == length(angles) ||
        throw(DimensionMismatch("generators and angles must have same length"))
end

# ------------------------------------------------------------------
# :dynamic engine — per-step subspace projection
# ------------------------------------------------------------------

# Forward pass. Returns (A, intro) where A = H_M and intro[p] is the step at
# which Pauli p first became reachable (0 for supp(H)). The rotation step
# never deletes keys — even a branch with an exactly zero coefficient (θ = 0
# or π/2) inserts its key, and truncation only filters terms not already present — so
# growth is monotone and S_k = {p : intro[p] ≤ k} is exactly the kept set
# after step k.
function _forward(H::PauliSum{N,T}, generators::Vector{PauliBasis{N}},
                  angles::Vector{<:Real}, truncation::TruncationStrategy) where {N,T}
    _check_lengths(generators, angles)
    A = deepcopy(H)
    intro = Dict{PauliBasis{N},Int}()
    for (p, _) in A
        intro[p] = 0
    end
    for (k, (g, θ)) in enumerate(zip(generators, angles))
        if truncation isa NoTruncation
            evolve!(A, g, θ)
        else
            _evolve_monotone!(A, g, θ, truncation)
        end
        for (p, _) in A
            get!(intro, p, k)
        end
    end
    return A, intro
end

function _reverse_sweep_dynamic!(A::PauliSum{N}, ρ::PauliSum{N}, intro::Dict{PauliBasis{N},Int},
                                 generators::Vector{PauliBasis{N}},
                                 angles::Vector{<:Real}) where {N}
    M = length(generators)
    cost = real(_pair(A, ρ))
    grad = zeros(Float64, M)
    for i in M:-1:1
        grad[i] = _grad_kernel(A, ρ, generators[i])
        evolve!(A, generators[i], -angles[i])
        evolve!(ρ, generators[i], -angles[i])
        # Paulis first reachable at step i are identically zero in A_{i-1}
        # (only floating-point residue remains) — delete them exactly, by
        # bookkeeping rather than by coefficient magnitude. Keys the forward
        # pass never recorded (possible only under truncation) go too.
        lim = i - 1
        filter!(kv -> get(intro, kv.first, typemax(Int)) <= lim, A)
        # ρ_{i-1} restricted to the reachable set R_{i-1} is exact: every
        # later gradient read is at a key of some R_j (reachable sets are
        # closed under partnering), and the coefficient flow into those keys
        # comes from forward images, which never leave the reachable set.
        filter!(kv -> haskey(A, kv.first), ρ)
    end
    return cost, grad
end

# ------------------------------------------------------------------
# :static engine — final-subspace projection
# ------------------------------------------------------------------

function _forward_monotone(H::PauliSum{N,T}, generators::Vector{PauliBasis{N}},
                           angles::Vector{<:Real}, truncation::TruncationStrategy) where {N,T}
    _check_lengths(generators, angles)
    A = deepcopy(H)
    for (g, θ) in zip(generators, angles)
        if truncation isa NoTruncation
            evolve!(A, g, θ)
        else
            _evolve_monotone!(A, g, θ, truncation)
        end
    end
    return A
end

# `evolve!` with the truncation applied only to sin-branch terms that would
# CREATE a new key: contributions onto established keys always merge, so a
# Pauli, once introduced, is never deleted and the final key set contains
# every Pauli that ever appeared. (Same math as `evolve!` for the surviving
# terms; per-key accumulation order is identical.)
function _evolve_monotone!(O::PauliSum{N,T}, G::PauliBasis{N}, θ::Real,
                           truncation::TruncationStrategy) where {N,T}
    _cos = cos(θ)
    _sin = 1im * sin(θ)
    onto_existing = PauliSum(N)
    onto_new = PauliSum(N)
    for (p, c) in O
        commute(p, G) && continue
        tmp = c * _sin * G * p
        q = PauliBasis(tmp)
        dest = haskey(O, q) ? onto_existing : onto_new
        dest[q] = get(dest, q, zero(ComplexF64)) + coeff(tmp)
        O[p] *= _cos
    end
    truncate!(onto_new, truncation)
    sum!(O, onto_existing)
    sum!(O, onto_new)
    return O
end

function _reverse_sweep_static!(A::PauliSum{N}, ρ::PauliSum{N},
                                generators::Vector{PauliBasis{N}},
                                angles::Vector{<:Real}) where {N}
    S = Set(keys(A))                # the final subspace of the forward pass
    M = length(generators)
    cost = real(_pair(A, ρ))
    grad = zeros(Float64, M)
    for i in M:-1:1
        grad[i] = _grad_kernel(A, ρ, generators[i])
        evolve!(A, generators[i], -angles[i])
        evolve!(ρ, generators[i], -angles[i])
        # constrain the reverse evolution to the final forward subspace
        filter!(kv -> kv.first in S, A)
        filter!(kv -> kv.first in S, ρ)
    end
    return cost, grad
end

# ------------------------------------------------------------------
# shared kernels
# ------------------------------------------------------------------

# (i/2)[G, A_i] paired with ρ_i: for anticommuting P, [G,P] = 2 G·P, so
# grad = Re( i Σ a_P·φ_P·r_{Q_P} ) with G·P = φ_P·Q_P, φ_P = ±i.
function _grad_kernel(A::PauliSum{N}, ρ::PauliSum{N}, G::PauliBasis{N}) where {N}
    acc = zero(ComplexF64)
    for (p, c) in A
        commute(p, G) && continue
        tmp = G * p
        acc += c * coeff(tmp) * get(ρ, PauliBasis(tmp), zero(ComplexF64))
    end
    return real(1im * acc)
end

# Unconjugated pairing Σ_P a_P r_P. (Not inner_product: that conjugates its
# first argument and normalizes by 2^N.)
function _pair(A::PauliSum{N}, R::PauliSum{N}) where {N}
    small, big = length(A) < length(R) ? (A, R) : (R, A)
    out = zero(ComplexF64)
    for (p, c) in small
        out += c * get(big, p, zero(ComplexF64))
    end
    return out
end
