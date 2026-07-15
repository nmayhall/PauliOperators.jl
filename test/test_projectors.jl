using Test
using PauliOperators
using Random
using LinearAlgebra

@testset "PauliSubspaceProjector" begin
    Random.seed!(2)
    N = 4
    O = PauliSum(N, ComplexF64)
    while length(O) < 40
        O[PauliBasis{N}(rand(0:15), rand(0:15))] = randn() + 0im
    end
    keep = Set(collect(keys(O))[1:15])
    P = PauliSubspaceProjector(keep)

    # plain filter: survivors are exactly the kept keys, coefficients untouched
    O2 = deepcopy(O)
    truncate!(O2, P, NoCorrection())
    @test Set(keys(O2)) == keep ∩ Set(keys(O))
    @test all(O2[p] == O[p] for p in keys(O2))

    # constructor from an operator's support
    O3 = deepcopy(O)
    truncate!(O3, PauliSubspaceProjector(O2), NoCorrection())
    @test Set(keys(O3)) == Set(keys(O2))

    # SparsePauliVector path agrees with the Dict path
    v = SparsePauliVector(O)
    truncate!(v, P, NoCorrection())
    Ov = PauliSum(v)
    @test Set(keys(Ov)) == Set(keys(O2))
    @test all(isapprox(Ov[p], O2[p]; atol = 1e-14) for p in keys(Ov))

    # as a per-rotation truncation inside sequence evolution: support can never
    # leave the subspace
    H = PauliSum(N, ComplexF64)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, X = [i, i + 1]))] = 1.0
        H[PauliBasis(Pauli(N, Z = [i]))] = 0.5
    end
    gens, angs = trotterize(H, 0.1; order = 2)
    Ostart = PauliSum(N, ComplexF64)
    Ostart[first(keep)] = 1.0 + 0im
    sub = PauliSubspaceProjector(keep)
    Ot = evolve(Ostart, gens, angs; truncation = sub)
    @test all(p in keep for p in keys(Ot))

    # correction accumulator composes
    corr = EnergyCorrection(Ket(N, 0))
    truncate!(deepcopy(O), P, corr)
    @test isfinite(corr.accumulated_energy)
end

@testset "QubitSubspaceProjector" begin
    Random.seed!(3)
    N = 3
    randop() = begin
        O = PauliSum(N, ComplexF64)
        while length(O) < 30
            O[PauliBasis{N}(rand(0:7), rand(0:7))] = randn() + 0im
        end
        O
    end

    # (a) :maxmixed = pure drop projection onto the kept register
    O = randop()
    Q = QubitSubspaceProjector(N, [1])
    O2 = deepcopy(O)
    truncate!(O2, Q, NoCorrection())
    env = ~(UInt64(1))
    expected = Dict(p => c for (p, c) in O if (p.z | p.x) & env == 0)
    @test Set(keys(O2)) == Set(keys(expected))
    @test all(isapprox(O2[p], expected[p]; atol = 1e-14) for p in keys(O2))

    # (b) computational-basis reference: exact for any matching product state.
    # env qubits 2,3 in |1⟩,|0⟩; system qubit 1 arbitrary basis state.
    ref = Ket(N, 0b010)
    Qk = QubitSubspaceProjector(N, [1]; reference = ref)
    O = randop()
    EO = deepcopy(O)
    truncate!(EO, Qk, NoCorrection())
    @test all((p.z | p.x) & env == 0 for p in keys(EO))
    for sysbit in 0:1
        k = Ket(N, 0b010 | sysbit)
        @test expectation_value(O, k) ≈ expectation_value(EO, k) atol = 1e-12
    end
    # hand check: Z1Z2 with qubit 2 in |1⟩ folds to -Z1
    Ozz = PauliSum(Pauli(N, Z = [1, 2]))
    truncate!(Ozz, Qk, NoCorrection())
    @test Ozz[PauliBasis(Pauli(N, Z = [1]))] ≈ -1.0

    # (c) general Bloch reference: exact for arbitrary pure product env states
    amps = [normalize(randn(ComplexF64, 2)) for _ in 1:N]
    bloch = zeros(3, N)
    for i in 1:N
        a, b = amps[i]
        s = conj(a) * b
        bloch[1, i] = 2real(s)
        bloch[2, i] = 2imag(s)
        bloch[3, i] = abs2(a) - abs2(b)
    end
    Qb = QubitSubspaceProjector(N, [1]; reference = bloch)
    O = randop()
    EO = deepcopy(O)
    truncate!(EO, Qb, NoCorrection())
    envstate = begin                      # |ψ₂⟩ ⊗ |ψ₃⟩ as a 2-qubit KetSum
        k2 = KetSum(1; T = ComplexF64); k2[Ket(1, 0)] = amps[2][1]; k2[Ket(1, 1)] = amps[2][2]
        k3 = KetSum(1; T = ComplexF64); k3[Ket(1, 0)] = amps[3][1]; k3[Ket(1, 1)] = amps[3][2]
        otimes(k2, k3)
    end
    for sysbit in 0:1
        sys = KetSum(1; T = ComplexF64); sys[Ket(1, sysbit)] = 1.0 + 0im
        ψ = otimes(sys, envstate)         # qubit 1 = lowest bit
        @test expectation_value(O, ψ) ≈ expectation_value(EO, ψ) atol = 1e-12
    end

    # argument validation
    @test_throws ArgumentError QubitSubspaceProjector(N, [5])
    @test_throws ArgumentError QubitSubspaceProjector(N, [1]; reference = zeros(2, N))
    @test_throws ArgumentError QubitSubspaceProjector(N, [1]; reference = Ket(2, 0))
end
