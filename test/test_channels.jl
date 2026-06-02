using PauliOperators
using Test
using LinearAlgebra
using Random


# --- helpers (test-local; not part of the public API) -----------------------

const _I2 = ComplexF64[1 0; 0 1]
const _X  = ComplexF64[0 1; 1 0]
const _Y  = ComplexF64[0 -im; im 0]
const _Z  = ComplexF64[1 0; 0 -1]

# Build the i.i.d. tensor product of single-qubit Pauli Kraus operators.
# `qubits` controls which qubits the channel acts on; identity is applied
# to all other qubits. Kron order matches `Base.Matrix(::PauliBasis)`:
# qubit 1 is the rightmost (least-significant) factor.
function _iid_pauli_kraus(N::Int, pI::Real, pX::Real, pY::Real, pZ::Real;
                          qubits=1:N)
    qset = Set(Int.(qubits))
    sqrtps = (sqrt(pI), sqrt(pX), sqrt(pY), sqrt(pZ))
    paulis = (_I2, _X, _Y, _Z)
    nact = length(qset)
    krauss = Matrix{ComplexF64}[]
    for idx in 0:(4^nact - 1)
        coef = 1.0 + 0im
        mat  = ComplexF64[1.0;;]   # 1×1 unit
        active_idx = 0
        for i in 1:N
            if i in qset
                choice = (idx ÷ 4^active_idx) % 4 + 1
                coef *= sqrtps[choice]
                mat = kron(paulis[choice], mat)
                active_idx += 1
            else
                mat = kron(_I2, mat)
            end
        end
        push!(krauss, coef * mat)
    end
    return krauss
end

function _heisenberg_apply(O::Matrix, krauss)
    out = zero(O)
    for K in krauss
        out += K' * O * K
    end
    return out
end

function _random_real_paulisum(N::Int; n_paulis::Int=8, T=ComplexF64)
    O = PauliSum(N, T)
    for _ in 1:n_paulis
        P = rand(PauliBasis{N})
        O[P] = T(randn() + im*randn())
    end
    return O
end


# --- 1. Identity at p = 0 ---------------------------------------------------

@testset "Channels: identity at p=0" begin
    Random.seed!(1)
    N = 4
    O0 = _random_real_paulisum(N; n_paulis=12)

    for ch! in (depolarizing_channel!, dephasing_channel!,
                phase_flip_channel!,    bit_flip_channel!,
                bit_phase_flip_channel!)
        O = deepcopy(O0)
        ch!(O, 0.0)
        @test keys(O) == keys(O0)
        for P in keys(O0)
            @test O[P] ≈ O0[P]
        end
    end

    O = deepcopy(O0)
    pauli_channel!(O, 0.0, 0.0, 0.0)
    for P in keys(O0)
        @test O[P] ≈ O0[P]
    end
end


# --- 2. Identity term invariant (unitality) ---------------------------------

@testset "Channels: identity Pauli term invariant (unitality)" begin
    Random.seed!(2)
    N = 4
    Pid = PauliBasis{N}(Int128(0), Int128(0))

    for ch! in (depolarizing_channel!, dephasing_channel!,
                bit_flip_channel!,     bit_phase_flip_channel!)
        for p in (0.0, 0.1, 0.4, 0.7, 1.0)
            O = _random_real_paulisum(N; n_paulis=6)
            O[Pid] = 3.14 + 0im
            ch!(O, p)
            @test O[Pid] ≈ 3.14 + 0im
            ch!(O, p; qubits=[1, 3])
            @test O[Pid] ≈ 3.14 + 0im
            ch!(O, p; qubits=2)
            @test O[Pid] ≈ 3.14 + 0im
        end
    end

    for (pX, pY, pZ) in ((0.0,0.0,0.0), (0.1,0.2,0.3), (0.5,0.0,0.5))
        O = _random_real_paulisum(N; n_paulis=6)
        O[Pid] = -2.0 + 1im
        pauli_channel!(O, pX, pY, pZ)
        @test O[Pid] ≈ -2.0 + 1im
    end
end


# --- 3. Closed-form scaling on hand-picked Pauli strings --------------------

@testset "Channels: closed-form scaling" begin
    N = 3

    # depolarizing: each non-I qubit contributes (1-4p/3)
    p = 0.1
    λd = 1 - 4p/3
    O = PauliSum(N)
    O[PauliBasis("XIZ")] = 1.0 + 0im   # weight 2 → λd^2
    O[PauliBasis("YYY")] = 2.0 + 0im   # weight 3 → λd^3
    O[PauliBasis("III")] = 7.0 + 0im   # weight 0 → unchanged
    depolarizing_channel!(O, p)
    @test O[PauliBasis("XIZ")] ≈ λd^2
    @test O[PauliBasis("YYY")] ≈ 2.0 * λd^3
    @test O[PauliBasis("III")] ≈ 7.0

    # dephasing: each X or Y qubit contributes (1-2p)
    p = 0.2
    λ = 1 - 2p
    O = PauliSum(N)
    O[PauliBasis("XIZ")] = 1.0 + 0im   # one X → λ
    O[PauliBasis("YYZ")] = 1.0 + 0im   # two X/Y → λ^2
    O[PauliBasis("ZIZ")] = 1.0 + 0im   # no X/Y → 1
    dephasing_channel!(O, p)
    @test O[PauliBasis("XIZ")] ≈ λ
    @test O[PauliBasis("YYZ")] ≈ λ^2
    @test O[PauliBasis("ZIZ")] ≈ 1.0

    # bit-flip: each Y or Z qubit contributes (1-2p)
    p = 0.3
    λ = 1 - 2p
    O = PauliSum(N)
    O[PauliBasis("XIZ")] = 1.0 + 0im   # one Z → λ
    O[PauliBasis("YYZ")] = 1.0 + 0im   # three Y/Z → λ^3
    O[PauliBasis("XIX")] = 1.0 + 0im   # no Y/Z → 1
    bit_flip_channel!(O, p)
    @test O[PauliBasis("XIZ")] ≈ λ
    @test O[PauliBasis("YYZ")] ≈ λ^3
    @test O[PauliBasis("XIX")] ≈ 1.0

    # bit-phase-flip: each X or Z qubit contributes (1-2p)
    p = 0.25
    λ = 1 - 2p
    O = PauliSum(N)
    O[PauliBasis("XIZ")] = 1.0 + 0im   # X and Z → λ^2
    O[PauliBasis("YYY")] = 1.0 + 0im   # no X/Z → 1
    O[PauliBasis("XYZ")] = 1.0 + 0im   # X, Z → λ^2
    bit_phase_flip_channel!(O, p)
    @test O[PauliBasis("XIZ")] ≈ λ^2
    @test O[PauliBasis("YYY")] ≈ 1.0
    @test O[PauliBasis("XYZ")] ≈ λ^2

    # general pauli_channel!: λX = 1-2(pY+pZ), etc.
    pX, pY, pZ = 0.05, 0.10, 0.15
    λX = 1 - 2(pY + pZ); λY = 1 - 2(pX + pZ); λZ = 1 - 2(pX + pY)
    O = PauliSum(N)
    O[PauliBasis("XYZ")] = 1.0 + 0im
    O[PauliBasis("XIX")] = 1.0 + 0im
    pauli_channel!(O, pX, pY, pZ)
    @test O[PauliBasis("XYZ")] ≈ λX * λY * λZ
    @test O[PauliBasis("XIX")] ≈ λX * λX
end


# --- 4. Brute-force Kraus equivalence (small N) -----------------------------

@testset "Channels: brute-force Kraus equivalence" begin
    Random.seed!(3)
    tol = 1e-11

    cases = [
        (:depol,    depolarizing_channel!,    p -> (1-p,  p/3, p/3, p/3)),
        (:deph,     dephasing_channel!,       p -> (1-p,  0.0, 0.0, p)),
        (:phase,    phase_flip_channel!,      p -> (1-p,  0.0, 0.0, p)),
        (:bitflip,  bit_flip_channel!,        p -> (1-p,  p,   0.0, 0.0)),
        (:bitphase, bit_phase_flip_channel!,  p -> (1-p,  0.0, p,   0.0)),
    ]

    for N in 1:3
        for (_, ch!, weights) in cases
            for p in (0.0, 0.07, 0.3, 0.55, 0.91, 1.0)
                O = _random_real_paulisum(N; n_paulis=2^N)
                pI, pX, pY, pZ = weights(p)
                kr = _iid_pauli_kraus(N, pI, pX, pY, pZ; qubits=1:N)
                M_ref = _heisenberg_apply(Matrix(O), kr)
                ch!(O, p)
                M_got = Matrix(O)
                @test norm(M_got - M_ref) < tol
            end
        end

        # general pauli_channel! with asymmetric weights
        for (pX, pY, pZ) in ((0.1,0.2,0.3), (0.4,0.0,0.4), (0.0,0.5,0.0), (0.33,0.33,0.33))
            O = _random_real_paulisum(N; n_paulis=2^N)
            pI = 1 - pX - pY - pZ
            kr = _iid_pauli_kraus(N, pI, pX, pY, pZ; qubits=1:N)
            M_ref = _heisenberg_apply(Matrix(O), kr)
            pauli_channel!(O, pX, pY, pZ)
            @test norm(Matrix(O) - M_ref) < tol
        end
    end
end


# --- 5. Subset selectivity --------------------------------------------------

@testset "Channels: qubits subset selectivity" begin
    Random.seed!(4)
    tol = 1e-11

    for N in 2:3
        for qubits in (Int[1], Int[N], collect(1:N), [1, N])
            for (pX, pY, pZ) in ((0.0,0.0,0.0), (0.1,0.2,0.3), (0.4,0.0,0.4))
                O = _random_real_paulisum(N; n_paulis=2^N)
                pI = 1 - pX - pY - pZ
                kr = _iid_pauli_kraus(N, pI, pX, pY, pZ; qubits=qubits)
                M_ref = _heisenberg_apply(Matrix(O), kr)
                pauli_channel!(O, pX, pY, pZ; qubits=qubits)
                @test norm(Matrix(O) - M_ref) < tol
            end
        end
    end
end


# --- 6. Layer commutativity (i.i.d. on disjoint qubits) ---------------------

@testset "Channels: i.i.d. layers on disjoint qubits commute" begin
    Random.seed!(5)
    N = 4
    p = 0.2
    O0 = _random_real_paulisum(N; n_paulis=12)

    for ch! in (depolarizing_channel!, dephasing_channel!,
                bit_flip_channel!,     bit_phase_flip_channel!)
        # Apply to qubits [1] then [2,3] then [4]
        Oa = deepcopy(O0)
        ch!(Oa, p; qubits=[1])
        ch!(Oa, p; qubits=[2, 3])
        ch!(Oa, p; qubits=[4])

        # Apply to all qubits in one go
        Ob = deepcopy(O0)
        ch!(Ob, p)

        @test keys(Oa) == keys(Ob)
        for P in keys(O0)
            @test Oa[P] ≈ Ob[P]
        end

        # Reverse order on disjoint qubits also matches
        Oc = deepcopy(O0)
        ch!(Oc, p; qubits=[4])
        ch!(Oc, p; qubits=[2, 3])
        ch!(Oc, p; qubits=[1])
        for P in keys(O0)
            @test Oc[P] ≈ Ob[P]
        end
    end
end


# --- 7. Composition multiplicativity for depolarizing -----------------------

@testset "Channels: depolarizing composition multiplicativity" begin
    Random.seed!(6)
    N = 3
    p1, p2 = 0.1, 0.25
    λ1 = 1 - 4p1/3
    λ2 = 1 - 4p2/3

    O0 = _random_real_paulisum(N; n_paulis=2^N)

    O = deepcopy(O0)
    depolarizing_channel!(O, p1)
    depolarizing_channel!(O, p2)

    for (P, c0) in O0
        w = count_ones(P.x | P.z)
        @test O[P] ≈ c0 * (λ1*λ2)^w
    end
end


# --- 8. Argument validation -------------------------------------------------

@testset "Channels: argument validation" begin
    N = 3
    O = PauliSum(N)
    O[PauliBasis("XIZ")] = 1.0 + 0im

    @test_throws ArgumentError depolarizing_channel!(deepcopy(O), -0.1)
    @test_throws ArgumentError depolarizing_channel!(deepcopy(O),  1.1)
    @test_throws ArgumentError dephasing_channel!(deepcopy(O), -0.1)
    @test_throws ArgumentError bit_flip_channel!(deepcopy(O),  2.0)
    @test_throws ArgumentError bit_phase_flip_channel!(deepcopy(O), -1.0)

    @test_throws ArgumentError pauli_channel!(deepcopy(O), -0.1, 0.1, 0.1)
    @test_throws ArgumentError pauli_channel!(deepcopy(O),  0.5, 0.5, 0.5)

    @test_throws ArgumentError depolarizing_channel!(deepcopy(O), 0.1; qubits=[0])
    @test_throws ArgumentError depolarizing_channel!(deepcopy(O), 0.1; qubits=[N+1])
    @test_throws ArgumentError pauli_channel!(deepcopy(O), 0.1, 0.1, 0.1; qubits=4)

    Oint = Dict{PauliBasis{N},Int}(PauliBasis("XIZ") => 1)
    @test_throws ArgumentError depolarizing_channel!(Oint, 0.1)
end


# --- 9. Aliasing ------------------------------------------------------------

@testset "Channels: phase_flip is alias for dephasing" begin
    Random.seed!(7)
    N = 3
    O0 = _random_real_paulisum(N; n_paulis=8)
    for p in (0.0, 0.13, 0.5, 0.8, 1.0)
        Oa = deepcopy(O0); dephasing_channel!(Oa, p)
        Ob = deepcopy(O0); phase_flip_channel!(Ob, p)
        @test keys(Oa) == keys(Ob)
        for P in keys(Oa)
            @test Oa[P] ≈ Ob[P]
        end
    end
    @test phase_flip_channel! === dephasing_channel!
    @test phase_flip_channel  === dephasing_channel
end


# --- 10. Weight-decay equivalence -------------------------------------------

@testset "Channels: depolarizing ⇔ exp(-γΔt·w) damping" begin
    Random.seed!(8)
    N = 4
    γ, Δt = 0.7, 0.4

    O0 = _random_real_paulisum(N; n_paulis=12)
    p  = depolarizing_p_for_weight_decay(γ, Δt)
    @test 0 ≤ p ≤ 1

    O = deepcopy(O0)
    depolarizing_channel!(O, p)

    for (P, c0) in O0
        w = count_ones(P.x | P.z)
        @test O[P] ≈ c0 * exp(-γ*Δt*w)
    end

    # Edge cases
    @test depolarizing_p_for_weight_decay(0.0, 1.0) ≈ 0.0
    @test depolarizing_p_for_weight_decay(1.0, 0.0) ≈ 0.0
end
