"""
    hadamard(p, q::Int)

Apply a Hadamard gate on qubit `q` to a `PauliSum` (Heisenberg picture) or
`KetSum` (Schrödinger picture). Dispatches to [`apply`](@ref)`(Hadamard(q), p)`.
"""
hadamard(p::PauliSum, q::Int) = apply(Hadamard(q), p)
hadamard(k::KetSum,   q::Int) = apply(Hadamard(q), k)

"""
    cnot(p, c::Int, t::Int)

Apply a CNOT gate (control `c`, target `t`).
"""
cnot(p::PauliSum, c::Int, t::Int) = apply(CNOT(c, t), p)
cnot(k::KetSum,   c::Int, t::Int) = apply(CNOT(c, t), k)

"""
    X_gate(p, q)

Apply a Pauli-X gate on qubit `q`.
"""
X_gate(p::PauliSum, q::Int) = apply(PauliX(q), p)
X_gate(k::KetSum,   q::Int) = apply(PauliX(q), k)

"""
    Y_gate(p, q)

Apply a Pauli-Y gate on qubit `q`.
"""
Y_gate(p::PauliSum, q::Int) = apply(PauliY(q), p)
Y_gate(k::KetSum,   q::Int) = apply(PauliY(q), k)

"""
    Z_gate(p, q)

Apply a Pauli-Z gate on qubit `q`.
"""
Z_gate(p::PauliSum, q::Int) = apply(PauliZ(q), p)
Z_gate(k::KetSum,   q::Int) = apply(PauliZ(q), k)

"""
    S_gate(p, q)

Apply the S = diag(1, i) phase gate on qubit `q`.
"""
S_gate(p::PauliSum, q::Int) = apply(PhaseGate(q), p)
S_gate(k::KetSum,   q::Int) = apply(PhaseGate(q), k)

"""
    T_gate(p, q)

Apply the T = diag(1, exp(iπ/4)) gate on qubit `q`. T is **not** a Clifford and is
still applied via Pauli rotation: `exp(-iπ/4 Z) = exp(-iπ/8) · T`.
"""
function T_gate(p::PauliSum{N}, q::Int) where N
    1 <= q <= N || throw(DimensionMismatch("qubit index $q exceeds N=$N"))
    return evolve(p, PauliBasis(Pauli(N, Z=[q])), π/4)
end
function T_gate(k::KetSum{N}, q::Int) where N
    1 <= q <= N || throw(DimensionMismatch("qubit index $q exceeds N=$N"))
    return evolve(k, PauliBasis(Pauli(N, Z=[q])), π/4)
end
