"""
    hadamard(p::PauliSum{N}, q::Int) where N

Apply a Hadamard gate on qubit `q` via Pauli rotations (Heisenberg picture).
"""
function hadamard(p::AnyPauliSum{N}, q::Int) where N
    q <= N || throw(DimensionMismatch("qubit index $q exceeds N=$N"))
    Z = PauliBasis(Pauli(N, Z=[q]))
    X = PauliBasis(Pauli(N, X=[q]))
    out = evolve(p, Z, π/2)
    out = evolve(out, X, π/2)
    out = evolve(out, Z, π/2)
    return out
end

"""
    hadamard(p::KetSum{N}, q::Int) where N

Apply a Hadamard gate on qubit `q` via Pauli rotations (Schrödinger picture).
"""
function hadamard(p::KetSum{N}, q::Int) where N
    q <= N || throw(DimensionMismatch("qubit index $q exceeds N=$N"))
    Z = PauliBasis(Pauli(N, Z=[q]))
    X = PauliBasis(Pauli(N, X=[q]))
    out = evolve(p, Z, π/2)
    out = evolve(out, X, π/2)
    out = evolve(out, Z, π/2)
    return 1im * out
end

"""
    cnot(p::PauliSum{N}, c::Int, t::Int) where N

Apply a CNOT gate (control `c`, target `t`) via Pauli rotations (Heisenberg picture).
"""
function cnot(p::AnyPauliSum{N}, c::Int, t::Int) where N
    c <= N || throw(DimensionMismatch("control qubit $c exceeds N=$N"))
    t <= N || throw(DimensionMismatch("target qubit $t exceeds N=$N"))
    Zc = PauliBasis(Pauli(N, Z=[c]))
    Xt = PauliBasis(Pauli(N, X=[t]))
    ZXct = PauliBasis(Pauli(N, Z=[c], X=[t]))
    out = evolve(p, ZXct, π/2)
    out = evolve(out, Xt, -π/2)
    out = evolve(out, Zc, -π/2)
    return out
end

"""
    cnot(p::KetSum{N}, c::Int, t::Int) where N

Apply a CNOT gate (control `c`, target `t`) via Pauli rotations (Schrödinger picture).
"""
function cnot(p::KetSum{N}, c::Int, t::Int) where N
    c <= N || throw(DimensionMismatch("control qubit $c exceeds N=$N"))
    t <= N || throw(DimensionMismatch("target qubit $t exceeds N=$N"))
    Zc = PauliBasis(Pauli(N, Z=[c]))
    Xt = PauliBasis(Pauli(N, X=[t]))
    ZXct = PauliBasis(Pauli(N, Z=[c], X=[t]))
    out = evolve(p, Zc, -π/2)
    out = evolve(out, Xt, -π/2)
    out = evolve(out, ZXct, π/2)
    return exp(1im * π/4) * out
end

"""
    X_gate(p::PauliSum{N}, q) where N

Apply Pauli X gate on qubit `q` (Heisenberg picture).
"""
function X_gate(p::AnyPauliSum{N}, q) where N
    return evolve(p, PauliBasis(Pauli(N, X=[q])), π)
end

"""
    X_gate(p::KetSum{N}, q) where N

Apply Pauli X gate on qubit `q` (Schrödinger picture).
"""
function X_gate(p::KetSum{N}, q) where N
    return 1im * evolve(p, PauliBasis(Pauli(N, X=[q])), π)
end

"""
    Y_gate(p::PauliSum{N}, q) where N

Apply Pauli Y gate on qubit `q` (Heisenberg picture).
"""
function Y_gate(p::AnyPauliSum{N}, q) where N
    return evolve(p, PauliBasis(Pauli(N, Y=[q])), π)
end

"""
    Y_gate(p::KetSum{N}, q) where N

Apply Pauli Y gate on qubit `q` (Schrödinger picture).
"""
function Y_gate(p::KetSum{N}, q) where N
    return 1im * evolve(p, PauliBasis(Pauli(N, Y=[q])), π)
end

"""
    Z_gate(p::PauliSum{N}, q) where N

Apply Pauli Z gate on qubit `q` (Heisenberg picture).
"""
function Z_gate(p::AnyPauliSum{N}, q) where N
    return evolve(p, PauliBasis(Pauli(N, Z=[q])), π)
end

"""
    Z_gate(p::KetSum{N}, q) where N

Apply Pauli Z gate on qubit `q` (Schrödinger picture).
"""
function Z_gate(p::KetSum{N}, q) where N
    return 1im * evolve(p, PauliBasis(Pauli(N, Z=[q])), π)
end

"""
    S_gate(p::Union{PauliSum{N}, KetSum{N}}, q) where N

Apply S gate (π/2 Z-rotation) on qubit `q`.
"""
function S_gate(p::Union{AnyPauliSum{N}, KetSum{N}}, q) where N
    return evolve(p, PauliBasis(Pauli(N, Z=[q])), π/2)
end

"""
    T_gate(p::Union{PauliSum{N}, KetSum{N}}, q) where N

Apply T gate (π/4 Z-rotation) on qubit `q`.
"""
function T_gate(p::Union{AnyPauliSum{N}, KetSum{N}}, q) where N
    return evolve(p, PauliBasis(Pauli(N, Z=[q])), π/4)
end

"""
    hadamard_to_paulis(N, q::Int)

Return `(generators, angles)` for a Hadamard gate on qubit `q` in an N-qubit system.
"""
function hadamard_to_paulis(N, q::Int)
    q <= N || throw(DimensionMismatch("qubit index $q exceeds N=$N"))
    Z = PauliBasis(Pauli(N, Z=[q]))
    X = PauliBasis(Pauli(N, X=[q]))
    return PauliBasis{N}[Z, X, Z], Float64[π/2, π/2, π/2]
end

"""
    cnot_to_paulis(N, c::Int, t::Int)

Return `(generators, angles)` for a CNOT gate (control `c`, target `t`) in an N-qubit system.
"""
function cnot_to_paulis(N, c::Int, t::Int)
    c <= N || throw(DimensionMismatch("control qubit $c exceeds N=$N"))
    t <= N || throw(DimensionMismatch("target qubit $t exceeds N=$N"))
    Zc = PauliBasis(Pauli(N, Z=[c]))
    Xt = PauliBasis(Pauli(N, X=[t]))
    ZXct = PauliBasis(Pauli(N, Z=[c], X=[t]))
    return PauliBasis{N}[ZXct, Xt, Zc], Float64[π/2, -π/2, -π/2]
end

"""
    X_gate_to_paulis(N, q)

Return `(generators, angles)` for an X gate on qubit `q`.
"""
function X_gate_to_paulis(N, q)
    return PauliBasis{N}[PauliBasis(Pauli(N, X=[q]))], Float64[π]
end

"""
    Z_gate_to_paulis(N, q)

Return `(generators, angles)` for a Z gate on qubit `q`.
"""
function Z_gate_to_paulis(N, q)
    return PauliBasis{N}[PauliBasis(Pauli(N, Z=[q]))], Float64[π]
end
