using PauliOperators
using LinearAlgebra
using Test
using Random

# Maximum coefficient difference between a PauliSum and a SparsePauliVector
# result, over the union of their keys.
function _maxdiff(a::PauliSum{N}, v::SparsePauliVector{N}) where {N}
    b = PauliSum(v)
    allk = union(keys(a), keys(b))
    isempty(allk) && return 0.0
    return maximum(abs(get(a, k, zero(valtype(a))) - get(b, k, zero(valtype(b))))
                   for k in allk)
end

function _heisenberg_chain(N)
    H = PauliSum(N)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, X=[i, i+1]))] = 1.0
        H[PauliBasis(Pauli(N, Y=[i, i+1]))] = 0.7
        H[PauliBasis(Pauli(N, Z=[i, i+1]))] = 1.1
    end
    for i in 1:N
        H[PauliBasis(Pauli(N, Z=[i]))] = 0.3
    end
    return H
end

@testset "SparsePauliVector evolution" begin
    Random.seed!(31)
    N = 8

    @testset "single-rotation parity" begin
        for θ in (0.0, 0.3, π / 2, π, -1.234), _ in 1:3
            ps = rand(PauliSum{N}; n_paulis=40)
            v = SparsePauliVector(ps)
            G = PauliBasis(rand(Pauli{N}))
            ref = deepcopy(ps)
            evolve!(ref, G, θ)
            evolve!(v, G, θ)
            @test _maxdiff(ref, v) < 1e-13
            @test PauliOperators.check_spv(v)
            @test v.an == 0
        end
        # non-mutating form leaves the input untouched
        ps = rand(PauliSum{N}; n_paulis=10)
        v = SparsePauliVector(ps)
        G = PauliBasis(rand(Pauli{N}))
        v2 = evolve(v, G, 0.4)
        @test PauliSum(v) == ps
        @test _maxdiff(evolve(ps, G, 0.4), v2) < 1e-13
    end

    @testset "sequence parity at window=1 (all deterministic strategies)" begin
        H = _heisenberg_chain(N)
        gens, angs = trotterize(H, 0.05, n_trotter=2, order=2)
        O0 = PauliSum(N)
        O0[PauliBasis(Pauli(N, Z=[4]))] = 1.0

        for strat in (NoTruncation(),
                      CoeffTruncation(1e-4),
                      WeightTruncation(3),
                      XWeightTruncation(2),
                      MajoranaWeightTruncation(6),
                      WeightDampedTruncation(0.5, 1e-4),
                      XWeightDampedTruncation(0.5, 1e-4),
                      CompositeTruncation(WeightTruncation(4), CoeffTruncation(1e-6)),
                      AdaptiveTruncation(50, 1e-12))
            ref = evolve(O0, gens, angs; truncation=strat)
            v = SparsePauliVector(O0)
            evolve!(v, gens, angs; window=1, truncation=strat)
            @test _maxdiff(ref, v) < 1e-12
            @test PauliOperators.check_spv(v)
        end
    end

    @testset "threaded sequence parity at window=1" begin
        O0 = rand(PauliSum{N}; n_paulis=120)
        gens = PauliBasis{N}[PauliBasis(rand(Pauli{N})) for _ in 1:14]
        angs = [0.02 * randn() for _ in eachindex(gens)]

        for strat in (NoTruncation(), CoeffTruncation(1e-7), WeightTruncation(5))
            serial = SparsePauliVector(O0)
            threaded = SparsePauliVector(O0)
            evolve!(serial, gens, angs; window=1, truncation=strat, threaded=false)
            evolve!(threaded, gens, angs; window=1, truncation=strat, threaded=true)
            @test isapprox(serial, threaded; atol=1e-12)
            @test PauliOperators.check_spv(threaded)
        end
    end

    @testset "correction parity at window=1" begin
        H = _heisenberg_chain(N)
        gens, angs = trotterize(H, 0.05, n_trotter=2, order=2)
        O0 = PauliSum(N)
        O0[PauliBasis(Pauli(N, Z=[4]))] = 1.0
        ψ = Ket{N}(Int128(0b1010))
        strat = CompositeTruncation(WeightTruncation(3), CoeffTruncation(1e-5))

        c1 = EnergyCorrection(ψ)
        c2 = EnergyCorrection(ψ)
        ref = evolve(O0, gens, angs; truncation=strat, correction=c1)
        v = SparsePauliVector(O0)
        evolve!(v, gens, angs; window=1, truncation=strat, correction=c2)
        @test abs(c1.accumulated_energy) > 0        # correction actually fired
        @test isapprox(c1.accumulated_energy, c2.accumulated_energy;
                       atol=1e-11, rtol=1e-9)
        @test _maxdiff(ref, v) < 1e-12

        c1 = EnergyVarianceCorrection(ψ)
        c2 = EnergyVarianceCorrection(ψ)
        ref = evolve(O0, gens, angs; truncation=strat, correction=c1)
        v = SparsePauliVector(O0)
        evolve!(v, gens, angs; window=1, truncation=strat, correction=c2)
        @test isapprox(c1.accumulated_energy, c2.accumulated_energy;
                       atol=1e-11, rtol=1e-9)
        @test isapprox(c1.accumulated_variance, c2.accumulated_variance;
                       atol=1e-10, rtol=1e-8)
    end

    @testset "windowed cadence" begin
        H = _heisenberg_chain(6)
        gens, angs = trotterize(H, 0.05, n_trotter=2, order=2)
        O0 = PauliSum(6)
        O0[PauliBasis(Pauli(6, Z=[3]))] = 1.0

        # Untruncated: window must not change the result (dedup is linear)
        v1 = SparsePauliVector(O0)
        evolve!(v1, gens, angs; window=1)
        for w in (3, 7, length(gens))
            vw = SparsePauliVector(O0)
            evolve!(vw, gens, angs; window=w)
            @test isapprox(v1, vw; atol=1e-10)
        end

        # Truncated, window=w: reference is the Dict path truncating every w
        # rotations. WeightTruncation drops on an integer criterion, so the
        # comparison is immune to FP summation-order effects near a
        # coefficient threshold. Buffers are sized generously so no early
        # merge fires — an early merge applies the strict filter mid-window
        # (documented cadence difference) and would break exact parity.
        w = 5
        strat = WeightTruncation(3)
        ref = deepcopy(O0)
        for (i, (g, θ)) in enumerate(zip(gens, angs))
            evolve!(ref, g, θ)
            (i % w == 0 || i == length(gens)) && truncate!(ref, strat)
        end
        vw = SparsePauliVector(O0; capacity_factor=5000, append_factor=10)
        cnt = PauliOperators.WindowCounters(cld(length(gens), w))
        evolve!(vw, gens, angs; window=w, truncation=strat, counters=cnt)
        @test sum(cnt.early_merges) == 0
        @test _maxdiff(ref, vw) < 1e-12

        # local_truncation must be compilable
        v = SparsePauliVector(O0)
        @test_throws ArgumentError evolve!(v, gens, angs;
                                           local_truncation=AdaptiveTruncation(10, 1e-12))

        # local weight truncation: exact at append time, equals strict result
        strat = WeightTruncation(2)
        refv = SparsePauliVector(O0)
        evolve!(refv, gens, angs; window=1, truncation=strat)
        vloc = SparsePauliVector(O0)
        evolve!(vloc, gens, angs; window=1, truncation=strat, local_truncation=strat)
        @test isapprox(refv, vloc; atol=1e-11)
    end

    @testset "early merge under tight append capacity" begin
        H = _heisenberg_chain(N)
        gens, angs = trotterize(H, 0.05, n_trotter=2, order=2)
        O0 = PauliSum(N)
        O0[PauliBasis(Pauli(N, Z=[4]))] = 1.0
        v = SparsePauliVector(O0; append_factor=0.01, min_capacity=4)
        nw = cld(length(gens), 10)
        cnt = PauliOperators.WindowCounters(nw)
        evolve!(v, gens, angs; window=10, counters=cnt)
        @test sum(cnt.early_merges) > 0     # tight buffers actually forced merges
        ref = SparsePauliVector(O0)
        evolve!(ref, gens, angs; window=1)
        @test isapprox(ref, v; atol=1e-10)  # early merges don't change the result
    end

    @testset "stochastic strategies (smoke, seeded)" begin
        Random.seed!(99)
        ps = PauliSum(N, Float64)
        while length(ps) < 100
            ps[PauliBasis(rand(Pauli{N}))] = 0.02 * (rand() - 0.5)
        end
        v = SparsePauliVector(ps; T=Float64)

        rng = Xoshiro(1)
        vs = copy(v)
        truncate!(vs, StochasticSamplingTruncation(20, rng))
        @test length(vs) <= 20
        @test PauliOperators.check_spv(vs)
        @test isapprox(norm(vs), norm(v); rtol=1e-10)  # norm-preserving rescale

        rng = Xoshiro(2)
        vc = copy(v)
        truncate!(vc, StochasticCoeffTruncation(0.02, rng))
        @test length(vc) <= length(v)
        @test PauliOperators.check_spv(vc)
        # promoted coefficients sit at ±ε
        @test all(abs(c) >= 0.02 - 1e-12 for c in values(vc))

        # unbiasedness of stochastic_clip! in expectation (loose statistical check)
        ψ = Ket{N}(Int128(0))
        ev0 = expectation_value(v, ψ)
        acc = 0.0
        M = 400
        rng = Xoshiro(3)
        for _ in 1:M
            vt = copy(v)
            stochastic_clip!(vt, 0.02; rng=rng)
            acc += real(expectation_value(vt, ψ))
        end
        @test isapprox(acc / M, real(ev0); atol=0.02)
    end

    @testset "gate parity (kernel-backed evolve path)" begin
        for _ in 1:5
            ps = rand(PauliSum{4}; n_paulis=10)
            v = SparsePauliVector(ps)
            q = rand(1:4)
            c, t = 1, 3
            @test norm(Matrix(hadamard(ps, q)) - Matrix(hadamard(v, q))) < 1e-12
            @test norm(Matrix(cnot(ps, c, t)) - Matrix(cnot(v, c, t))) < 1e-12
            @test norm(Matrix(X_gate(ps, q)) - Matrix(X_gate(v, q))) < 1e-12
            @test norm(Matrix(Y_gate(ps, q)) - Matrix(Y_gate(v, q))) < 1e-12
            @test norm(Matrix(Z_gate(ps, q)) - Matrix(Z_gate(v, q))) < 1e-12
            @test norm(Matrix(S_gate(ps, q)) - Matrix(S_gate(v, q))) < 1e-12
            @test norm(Matrix(T_gate(ps, q)) - Matrix(T_gate(v, q))) < 1e-12
        end
    end
end
