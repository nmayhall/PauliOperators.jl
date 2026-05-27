using PauliOperators
using LinearAlgebra
using Test
using Random

# --- Reference dense matrices --------------------------------------------------

const H1q    = ComplexF64[1 1; 1 -1] / sqrt(2)
const S1q    = ComplexF64[1 0; 0 1im]
const Sd1q   = ComplexF64[1 0; 0 -1im]
const X1q    = ComplexF64[0 1; 1 0]
const Y1q    = ComplexF64[0 -1im; 1im 0]
const Z1q    = ComplexF64[1 0; 0 -1]
# Textbook √X = (1/2)[[1+i, 1-i], [1-i, 1+i]] satisfies (√X)² = X.
const SqX1q  = ComplexF64[(1+1im) (1-1im); (1-1im) (1+1im)] / 2
# √Y = exp(-iπ/4 Y) up to global phase: (I - iY)/√2.
const SqY1q  = (Matrix{ComplexF64}(I,2,2) - 1im * Y1q) / sqrt(2)

# CNOT(c=1,t=2) in PauliOperators' qubit-1=least-significant ordering.
const CNOT4  = ComplexF64[1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]
const CZ4    = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]
const SWAP4  = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]

# Lift a 1-qubit dense gate to qubit q of N qubits (qubit 1 = least-significant slot).
function lift1q(g::AbstractMatrix, q::Int, N::Int)
    out = ComplexF64[1.0;;]
    for i in 1:N
        out = kron(i == q ? g : Matrix{ComplexF64}(I, 2, 2), out)
    end
    return out
end

# Lift a 2-qubit gate acting on neighboring qubits (a, a+1) to N qubits.
# We use general explicit construction by swapping into position only when needed.
# For tests we only need to lift to adjacent slots.
function lift2q_adjacent(g4::AbstractMatrix, low::Int, N::Int)
    out = ComplexF64[1.0;;]
    i = 1
    while i <= N
        if i == low
            out = kron(g4, out)
            i += 2
        else
            out = kron(Matrix{ComplexF64}(I, 2, 2), out)
            i += 1
        end
    end
    return out
end

# Conjugation check: apply(gate, P) matches U Mat(P) U†
function pauli_conj_ok(gate::CliffordGate, U::AbstractMatrix, P::PauliSum; atol=1e-10)
    actual   = Matrix(apply(gate, P))
    expected = U * Matrix(P) * U'
    return isapprox(actual, expected; atol=atol)
end

# Schrödinger check: apply(gate, |ψ⟩) matches U |ψ⟩
function ket_action_ok(gate::CliffordGate, U::AbstractMatrix, ks::KetSum; atol=1e-10)
    actual   = Vector(apply(gate, ks))
    expected = U * ComplexF64.(Vector(ks))
    return isapprox(actual, expected; atol=atol)
end

@testset "Cliffords" begin

@testset "Primitive Pauli-conjugation on PauliSum (Heisenberg)" begin
    N = 3
    all_paulis = String[]
    for s1 in "IXYZ", s2 in "IXYZ", s3 in "IXYZ"
        push!(all_paulis, string(s1, s2, s3))
    end

    @testset "Hadamard" begin
        for q in 1:N
            U = lift1q(H1q, q, N)
            for s in all_paulis
                P = PauliSum(Pauli(s))
                @test pauli_conj_ok(Hadamard(q), U, P)
            end
        end
    end

    @testset "PhaseGate / PhaseDg" begin
        for q in 1:N
            U = lift1q(S1q, q, N)
            Ud = lift1q(Sd1q, q, N)
            for s in all_paulis
                P = PauliSum(Pauli(s))
                @test pauli_conj_ok(PhaseGate(q), U, P)
                @test pauli_conj_ok(PhaseDg(q), Ud, P)
            end
        end
    end

    @testset "SqrtX / SqrtY" begin
        for q in 1:N
            UX = lift1q(SqX1q, q, N)
            UY = lift1q(SqY1q, q, N)
            for s in all_paulis
                P = PauliSum(Pauli(s))
                @test pauli_conj_ok(SqrtX(q), UX, P)
                @test pauli_conj_ok(SqrtY(q), UY, P)
            end
        end
    end

    @testset "PauliX / PauliY / PauliZ" begin
        for q in 1:N
            for (gate, U) in ((PauliX(q), lift1q(X1q, q, N)),
                              (PauliY(q), lift1q(Y1q, q, N)),
                              (PauliZ(q), lift1q(Z1q, q, N)))
                for s in all_paulis
                    P = PauliSum(Pauli(s))
                    @test pauli_conj_ok(gate, U, P)
                end
            end
        end
    end

    @testset "CNOT / CZ / SWAP" begin
        N2 = 2
        all_paulis_2q = [string(s1, s2) for s1 in "IXYZ" for s2 in "IXYZ"]
        for (gate, U) in ((CNOT(1, 2), CNOT4), (CZ(1, 2), CZ4), (SWAP(1, 2), SWAP4))
            for s in all_paulis_2q
                P = PauliSum(Pauli(s))
                @test pauli_conj_ok(gate, U, P)
            end
        end

        # On a 3-qubit system, the third qubit should pass through unchanged.
        N3 = 3
        for (gate, U_2q) in ((CNOT(1, 2), CNOT4), (CZ(1, 2), CZ4), (SWAP(1, 2), SWAP4))
            U = kron(Matrix{ComplexF64}(I, 2, 2), U_2q)
            for s in ("XXI", "ZZI", "YYZ", "XIY", "IZX")
                P = PauliSum(Pauli(s))
                @test pauli_conj_ok(gate, U, P)
            end
        end
    end
end

@testset "Primitive action on KetSum (Schrödinger)" begin
    N = 2
    basis_kets = [[0,0], [1,0], [0,1], [1,1]]
    for q in 1:N
        for (gate, U) in ((Hadamard(q), lift1q(H1q, q, N)),
                          (PhaseGate(q), lift1q(S1q, q, N)),
                          (PhaseDg(q),   lift1q(Sd1q, q, N)),
                          (SqrtX(q),     lift1q(SqX1q, q, N)),
                          (SqrtY(q),     lift1q(SqY1q, q, N)),
                          (PauliX(q),    lift1q(X1q, q, N)),
                          (PauliY(q),    lift1q(Y1q, q, N)),
                          (PauliZ(q),    lift1q(Z1q, q, N)))
            for b in basis_kets
                ks = KetSum(Ket(b); T=ComplexF64)
                @test ket_action_ok(gate, U, ks)
            end
        end
    end

    for (gate, U) in ((CNOT(1, 2), CNOT4), (CZ(1, 2), CZ4), (SWAP(1, 2), SWAP4))
        for b in basis_kets
            ks = KetSum(Ket(b); T=ComplexF64)
            @test ket_action_ok(gate, U, ks)
        end
    end

    # Linear superposition: H on the |+⟩ component.
    ks_super = KetSum(N; T=ComplexF64)
    ks_super[Ket([0,0])] = 1/sqrt(2)
    ks_super[Ket([1,0])] = 1/sqrt(2)
    @test ket_action_ok(Hadamard(1), lift1q(H1q, 1, 2), ks_super)
end

@testset "Sign rules: a few explicit checks" begin
    # H Y H = -Y
    sgn, p = apply(Hadamard(1), PauliBasis(Pauli("Y")))
    @test sgn == -1
    @test p == PauliBasis(Pauli("Y"))

    # S X S† = Y
    sgn, p = apply(PhaseGate(1), PauliBasis(Pauli("X")))
    @test sgn == +1
    @test p == PauliBasis(Pauli("Y"))

    # PauliX on Z gives -Z
    sgn, p = apply(PauliX(1), PauliBasis(Pauli("Z")))
    @test sgn == -1
    @test p == PauliBasis(Pauli("Z"))

    # CNOT IX → IX (X on target stays)
    sgn, p = apply(CNOT(1, 2), PauliBasis(Pauli("IX")))
    @test sgn == +1
    @test p == PauliBasis(Pauli("IX"))

    # CNOT XX → XI (X on control propagates, then X on target squares to I)
    sgn, p = apply(CNOT(1, 2), PauliBasis(Pauli("XX")))
    @test sgn == +1
    @test p == PauliBasis(Pauli("XI"))
end

@testset "Tableau construction from primitive" begin
    N = 3
    for g in (Hadamard(2), PhaseGate(1), PhaseDg(3), SqrtX(1), SqrtY(2),
              PauliX(1), PauliY(2), PauliZ(3), CNOT(1, 2), CZ(2, 3), SWAP(1, 3))
        C = CliffordTableau{N}(g)
        for s in ("XYZ", "ZZZ", "IXI", "YII", "XII")
            P = PauliSum(Pauli(s))
            @test Matrix(apply(g, P)) ≈ Matrix(apply(C, P)) atol=1e-12
        end
    end
end

@testset "Composition: (C1 * C2) P = C1 (C2 P)" begin
    N = 3
    rng = Random.Xoshiro(42)
    gates = [Hadamard(1), CNOT(1, 2), PhaseGate(3), Hadamard(2), CZ(2, 3), SWAP(1, 3)]
    for _ in 1:3
        g1 = rand(rng, gates); g2 = rand(rng, gates)
        C1 = CliffordTableau{N}(g1); C2 = CliffordTableau{N}(g2)
        comp = C1 * C2
        for s in ("XYZ", "ZZZ", "IXI", "YXZ")
            P = PauliSum(Pauli(s))
            left  = apply(comp, P)
            right = apply(C1, apply(C2, P))
            @test Matrix(left) ≈ Matrix(right) atol=1e-12
        end
    end

    # CliffordTableau{N}([...]) folds left-to-right.
    seq = [Hadamard(1), CNOT(1, 2), PhaseGate(2)]
    C = CliffordTableau{N}(seq)
    P = PauliSum(Pauli("ZII"))
    step = P
    for g in seq
        step = apply(g, step)
    end
    @test Matrix(apply(C, P)) ≈ Matrix(step) atol=1e-12
end

@testset "Adjoint / inverse" begin
    N = 3
    C = CliffordTableau{N}([Hadamard(1), CNOT(1, 2), PhaseGate(3), CZ(2, 3),
                            SqrtX(2), SWAP(1, 3)])
    Ci = adjoint(C)

    for s in ("XYZ", "ZZZ", "IXI", "YII", "ZXY")
        P = PauliSum(Pauli(s))
        @test Matrix(apply(Ci, apply(C, P))) ≈ Matrix(P) atol=1e-12
        @test Matrix(apply(C, apply(Ci, P))) ≈ Matrix(P) atol=1e-12
    end

    # Phase-gate adjoint pair
    @test adjoint(PhaseGate(2)) == PhaseDg(2)
    @test adjoint(PhaseDg(2))   == PhaseGate(2)
    # Self-inverse primitives
    for g in (Hadamard(1), PauliX(2), PauliY(3), PauliZ(1), CNOT(1, 2), CZ(2, 3), SWAP(1, 3))
        @test adjoint(g) == g
    end
end

@testset "Matrix → tableau identification" begin
    # 1-qubit primitives
    for (g, U) in ((Hadamard(1), H1q), (PhaseGate(1), S1q), (PhaseDg(1), Sd1q),
                   (PauliX(1), X1q), (PauliY(1), Y1q), (PauliZ(1), Z1q),
                   (SqrtX(1), SqX1q), (SqrtY(1), SqY1q))
        @test CliffordTableau{1}(g) == CliffordTableau{1}(U)
    end

    # 2-qubit primitives
    @test CliffordTableau{2}(CNOT(1, 2)) == CliffordTableau{2}(CNOT4)
    @test CliffordTableau{2}(CZ(1, 2))   == CliffordTableau{2}(CZ4)
    @test CliffordTableau{2}(SWAP(1, 2)) == CliffordTableau{2}(SWAP4)

    # Lift small block onto target qubits of larger system
    @test CliffordTableau{4}(CNOT4, [2, 3]) == CliffordTableau{4}(CNOT(2, 3))
    @test CliffordTableau{5}(SWAP4, [1, 4]) == CliffordTableau{5}(SWAP(1, 4))
    @test CliffordTableau{3}(H1q,  [2])     == CliffordTableau{3}(Hadamard(2))

    # Non-Clifford rejection
    T1q = ComplexF64[1 0; 0 exp(1im*π/4)]
    @test_throws ArgumentError CliffordTableau{1}(T1q)

    # Non-unitary rejection
    @test_throws ArgumentError CliffordTableau{1}(ComplexF64[1 0; 0 2])

    # Wrong size rejection
    @test_throws ArgumentError CliffordTableau{2}(H1q)   # 2×2 passed for N=2 (expects 4×4)
end

@testset "Rewired named gates use the new API" begin
    N = 3
    # hadamard on PauliSum still gives H Z H = X
    P = PauliSum(Pauli("ZII"))
    @test Matrix(hadamard(P, 1)) ≈ Matrix(PauliSum(Pauli("XII"))) atol=1e-12

    # S_gate on KetSum: textbook S|1⟩ = i|1⟩
    k = KetSum(Ket([1, 0, 0]); T=ComplexF64)
    @test Vector(S_gate(k, 1))[2] ≈ 1im atol=1e-12

    # cnot permutes basis states: |100⟩ -> |110⟩, vector index = 1 + 2 + 1 = 4
    k10 = KetSum(Ket([1, 0, 0]); T=ComplexF64)
    @test Vector(cnot(k10, 1, 2))[4] ≈ 1.0 atol=1e-12
end

# Two matrices agree up to a global phase iff U₁ · U₂' is a scalar multiple of I.
function agree_up_to_phase(A::AbstractMatrix, B::AbstractMatrix; atol=1e-10)
    size(A) == size(B) || return false
    M = A * B'
    dim = size(M, 1)
    α = tr(M) / dim
    isapprox(abs(α), 1; atol=atol) || return false
    return isapprox(M, α * Matrix{ComplexF64}(I, dim, dim); atol=atol)
end

@testset "Matrix(CliffordTableau{N})" begin
    @testset "primitive matrices match analytic up to global phase" begin
        N = 2
        for (g, U1) in ((Hadamard(1),  H1q),
                        (PhaseGate(1), S1q),
                        (PhaseDg(1),   Sd1q),
                        (PauliX(1),    X1q),
                        (PauliY(1),    Y1q),
                        (PauliZ(1),    Z1q),
                        (SqrtX(1),     SqX1q),
                        (SqrtY(1),     SqY1q))
            U = Matrix(CliffordTableau{N}(g))
            U_ref = lift1q(U1, 1, N)
            @test agree_up_to_phase(U, U_ref)
        end

        for (g, U_ref) in ((CNOT(1, 2), CNOT4), (CZ(1, 2), CZ4), (SWAP(1, 2), SWAP4))
            U = Matrix(CliffordTableau{2}(g))
            @test agree_up_to_phase(U, U_ref)
        end
    end

    @testset "Matrix is unitary" begin
        for N in 2:4
            for g in (Hadamard(1), CNOT(1, 2), PhaseGate(1))
                U = Matrix(CliffordTableau{N}(g))
                dim = 1 << N
                @test isapprox(U * U', Matrix{ComplexF64}(I, dim, dim); atol=1e-10)
            end
        end
    end

    @testset "composition: Matrix(C1*C2) ≈ Matrix(C1)*Matrix(C2) up to phase" begin
        N = 3
        C1 = CliffordTableau{N}(Hadamard(2))
        C2 = CliffordTableau{N}(CNOT(1, 2))
        @test agree_up_to_phase(Matrix(C1 * C2), Matrix(C1) * Matrix(C2))

        C3 = CliffordTableau{N}([Hadamard(1), CNOT(1, 2), PhaseGate(3)])
        prod_matrix = Matrix(CliffordTableau{N}(PhaseGate(3))) *
                      Matrix(CliffordTableau{N}(CNOT(1, 2))) *
                      Matrix(CliffordTableau{N}(Hadamard(1)))
        @test agree_up_to_phase(Matrix(C3), prod_matrix)
    end

    @testset "adjoint: Matrix(adjoint(C)) ≈ Matrix(C)' up to phase" begin
        N = 3
        C = CliffordTableau{N}([Hadamard(1), CNOT(1, 2), PhaseGate(3), CZ(2, 3)])
        @test agree_up_to_phase(Matrix(adjoint(C)), Matrix(C)')
    end
end

@testset "rand(CliffordTableau{N})" begin
    @testset "result is a valid Clifford" begin
        rng = Random.Xoshiro(42)
        for N in 1:4
            C = rand(rng, CliffordTableau{N})
            # adjoint(C) * C must reduce to the identity tableau (Cliffords form a group;
            # the adjoint-inversion code asserts the input is a valid Clifford internally).
            @test adjoint(C) * C == CliffordTableau{N}()
            @test C * adjoint(C) == CliffordTableau{N}()
        end
    end

    @testset "dense matrix is unitary" begin
        rng = Random.Xoshiro(7)
        for N in 2:4
            C = rand(rng, CliffordTableau{N})
            U = Matrix(C)
            dim = 1 << N
            @test isapprox(U * U', Matrix{ComplexF64}(I, dim, dim); atol=1e-10)
        end
    end

    @testset "reproducibility with fixed RNG" begin
        rng1 = Random.Xoshiro(2024)
        rng2 = Random.Xoshiro(2024)
        @test rand(rng1, CliffordTableau{3}) == rand(rng2, CliffordTableau{3})
    end

    @testset "different seeds give different samples" begin
        C1 = rand(Random.Xoshiro(1), CliffordTableau{4})
        C2 = rand(Random.Xoshiro(2), CliffordTableau{4})
        @test C1 != C2
    end

    @testset "PauliSum conjugation is consistent with the tableau" begin
        # apply(C, P) must match the underlying tableau action: this is automatic for any
        # CliffordTableau, but we check that random samples don't accidentally violate it.
        rng = Random.Xoshiro(99)
        N = 3
        C = rand(rng, CliffordTableau{N})
        for s in ("XYZ", "ZIX", "YYY")
            P = PauliSum(Pauli(s))
            U = Matrix(C)
            @test isapprox(Matrix(apply(C, P)), U * Matrix(P) * U'; atol=1e-10)
        end
    end
end

end  # @testset "Cliffords"
