module PauliOperators

    using Printf
    using LinearAlgebra
    using StaticArrays
    using Random

    include("helpers.jl")
    include("type_PauliBasis.jl")
    include("type_Pauli.jl")
    include("type_PauliSum.jl")
    include("type_Ket.jl")
    include("type_KetSum.jl")
    include("type_DyadBasis.jl")
    include("type_Dyad.jl")
    include("type_DyadSum.jl")
    include("multiplication.jl")
    include("addition.jl")
    include("conversions.jl")
    include("expectation_value.jl")
    include("inner_product.jl")
    include("norms.jl")
    include("statistics.jl")
    include("commutator.jl")
    include("clip.jl")
    include("stochastic.jl")
    include("truncation.jl")
    include("evolve.jl")
    include("stochastic_propagation.jl")
    include("decompose.jl")
    include("gates.jl")
    include("analysis.jl")

    const ⊗ = otimes
    const ⊕ = osum
    const PHASE_TBL = SVector{4}([1, 1im, -1, -1im])

    export Pauli
    export PauliBasis
    export PauliSum
    export Ket
    export Bra
    export DyadBasis
    export Dyad
    export DyadSum
    export KetSum
    export clip!
    export ⊗
    export ⊕
    export otimes, osum
    export expectation_value
    export matrix_element
    export inner_product

    export symplectic_phase
    export coeff
    export commute

    export evolve, evolve!
    export weight, coeff_clip!, weight_clip!
    export majorana_weight, majorana_weight_clip!
    export stochastic_clip!
    export stochastic_propagate

    export variance, covariance
    export commutator, anticommutator
    export offdiag

    # Truncation strategy system
    export TruncationStrategy, CorrectionAccumulator
    export NoTruncation, CoeffTruncation, WeightTruncation
    export MajoranaWeightTruncation, CompositeTruncation
    export StochasticCoeffTruncation, StochasticSamplingTruncation
    export AdaptiveTruncation
    export NoCorrection, EnergyCorrection, EnergyVarianceCorrection
    export truncate!

    # Decomposition
    export trotterize, qdrift

    # Gates
    export hadamard, cnot
    export X_gate, Y_gate, Z_gate, S_gate, T_gate
    export hadamard_to_paulis, cnot_to_paulis
    export X_gate_to_paulis, Z_gate_to_paulis

    # Analysis
    export get_weight_counts, get_weight_probs
    export get_majorana_weight_counts, get_majorana_weight_probs
    export find_top_k, largest, largest_diag
end
