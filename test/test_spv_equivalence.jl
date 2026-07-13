using PauliOperators
using LinearAlgebra
using Test
using Random

# A/B parity: every algebraic/mathematical operation on PauliSum must give
# the same answer on SparsePauliVector. `same(op_ps, op_spv)` converts the
# SPV result back to a Dict and compares coefficients exactly-ish.
function _same(a::PauliSum{N}, v::SparsePauliVector{N}; tol=1e-12) where {N}
    b = PauliSum(v)
    allk = union(keys(a), keys(b))
    isempty(allk) && return true
    return maximum(abs(get(a, k, zero(valtype(a))) - get(b, k, zero(valtype(b))))
                   for k in allk) < tol
end

@testset "SparsePauliVector ⇔ PauliSum equivalence" begin
    Random.seed!(21)
    N = 6

    ps1 = rand(PauliSum{N}; n_paulis=25)
    ps2 = rand(PauliSum{N}; n_paulis=25)
    v1 = SparsePauliVector(ps1)
    v2 = SparsePauliVector(ps2)
    p = rand(Pauli{N})
    pb = PauliBasis(rand(Pauli{N}))
    ψ = rand(Ket{N})

    @testset "linear algebra" begin
        @test _same(ps1 + ps2, v1 + v2)
        @test _same(ps1 - ps2, v1 - v2)
        @test _same(ps1 + ps2', v1 + v2')
        @test _same(ps2' + ps1, v2' + v1)
        @test _same(-ps1, -v1)
        @test _same(ps1 * 2.3, v1 * 2.3)
        @test _same(2.3 * ps1, 2.3 * v1)
        @test _same(ps1 * (1.1 + 0.2im), v1 * (1.1 + 0.2im))
        @test _same(ps1' * 2.3, v1' * 2.3)

        a = deepcopy(ps1); b = copy(v1)
        mul!(a, 0.5); mul!(b, 0.5)
        @test _same(a, b)

        a = deepcopy(ps1); b = copy(v1)
        sum!(a, ps2); sum!(b, v2)
        @test _same(a, b)

        a = deepcopy(ps1); b = copy(v1)
        sum!(a, ps2'); sum!(b, v2')
        @test _same(a, b)

        a = deepcopy(ps1); b = copy(v1)
        sum!(a, p); sum!(b, p)
        @test _same(a, b)
        sum!(pb, a); sum!(pb, b)
        @test _same(a, b)
    end

    @testset "Singles ± / × sums" begin
        @test _same(p + ps1, p + v1)
        @test _same(ps1 + p, v1 + p)
        @test _same(p - ps1, p - v1)
        @test _same(ps1 - p, v1 - p)
        @test _same(pb + ps1, pb + v1)
        @test _same(pb - ps1, pb - v1)
        @test _same(p * ps1, p * v1; tol=1e-12)
        @test _same(ps1 * p, v1 * p; tol=1e-12)
        @test _same(pb * ps1, pb * v1; tol=1e-12)
        @test _same(ps1 * pb, v1 * pb; tol=1e-12)
    end

    @testset "products" begin
        @test _same(ps1 * ps2, v1 * v2)
        @test _same(ps1' * ps2, v1' * v2)
        @test _same(ps1 * ps2', v1 * v2')
        @test _same(ps1' * ps2', v1' * v2')
        @test _same(commutator(ps1, ps2), commutator(v1, v2))
        @test _same(anticommutator(ps1, ps2), anticommutator(v1, v2))
        # in-place commutator into presized output
        out = SparsePauliVector(N, ComplexF64; capacity=length(v1) * length(v2))
        commutator!(out, v1, v2)
        @test _same(commutator(ps1, ps2), out)
        # real-typed commutator output is an error (coefficients imaginary)
        outr = SparsePauliVector(N, Float64; capacity=16)
        r1 = SparsePauliVector(N, Float64; capacity=16)
        @test_throws ErrorException commutator!(outr, r1, r1)
    end

    @testset "tensor / direct sum" begin
        psA = rand(PauliSum{3}; n_paulis=5)
        psB = rand(PauliSum{2}; n_paulis=5)
        vA = SparsePauliVector(psA)
        vB = SparsePauliVector(psB)
        @test _same(psA ⊗ psB, vA ⊗ vB)
        @test _same(osum(psA, psB), osum(vA, vB))
    end

    @testset "observables" begin
        @test expectation_value(v1, ψ) ≈ expectation_value(ps1, ψ)
        d = rand(Dyad{N})
        db = rand(DyadBasis{N})
        ds = rand(DyadSum{N, ComplexF64})
        @test expectation_value(v1, d) ≈ expectation_value(ps1, d)
        @test expectation_value(v1, db) ≈ expectation_value(ps1, db)
        @test expectation_value(v1, ds) ≈ expectation_value(ps1, ds)
        ks = rand(KetSum{N}; n_terms=3)
        @test expectation_value(v1, ks) ≈ expectation_value(ps1, ks)
        k2 = rand(Ket{N})
        @test matrix_element(k2', v1, ψ) ≈ matrix_element(k2', ps1, ψ)

        @test inner_product(v1, v2) ≈ inner_product(ps1, ps2)
        for pn in (1, 2, Inf, 3)
            @test norm(v1, pn) ≈ norm(ps1, pn)
        end
        @test tr(v1) ≈ tr(ps1)
        @test ishermitian(v1) == ishermitian(ps1)
        @test variance(v1, ψ) ≈ variance(ps1, ψ)
        @test covariance(v1, v2, ψ) ≈ covariance(ps1, ps2, ψ)
        @test isapprox(v1, SparsePauliVector(deepcopy(ps1)))
        @test !isapprox(v1, v2)
        @test norm(Matrix(v1) - Matrix(ps1)) < 1e-12
        @test norm(Matrix(v1') - Matrix(ps1')) < 1e-12
        # v * Ket returns the same KetSum. (The KetSum is Float64-typed on
        # both paths, so use a Y-free real sum — Y factors introduce
        # imaginary phases that error identically on the Dict path.)
        psr = PauliSum(N, Float64)
        while length(psr) < 15
            q = rand(PauliBasis{N})
            q.z & q.x == 0 && (psr[q] = rand() - 0.5)   # no Y factors
        end
        vr = SparsePauliVector(psr; T=Float64)
        @test vr * ψ == psr * ψ
        # subspace matrix
        S = unique([rand(Ket{N}) for _ in 1:6])
        @test norm(Matrix(v1, S) - Matrix(ps1, S)) < 1e-12
    end

    @testset "clips / diagonal filters" begin
        big = PauliSum(N, Float64)
        while length(big) < 80
            big[PauliBasis(rand(Pauli{N}))] = rand() - 0.5
        end
        vb = SparsePauliVector(big; T=Float64)
        for (f!, args) in ((coeff_clip!, (0.2,)),
                           (weight_clip!, (3,)),
                           (x_weight_clip!, (2,)),
                           (majorana_weight_clip!, (5,)),
                           (weight_damped_clip!, (0.5, 0.05)),
                           (x_weight_damped_clip!, (0.5, 0.05)))
            a = deepcopy(big)
            b = copy(vb)
            f!(a, args...)
            f!(b, args...)
            @test PauliSum(b) == a
            @test PauliOperators.check_spv(b)
        end
        @test PauliSum(offdiag(vb)) == offdiag(big)
        @test PauliSum(diag(vb)) == diag(big)
    end

    @testset "analysis" begin
        @test get_weight_counts(v1) == get_weight_counts(ps1)
        @test get_weight_probs(v1) ≈ get_weight_probs(ps1)
        @test get_majorana_weight_counts(v1) == get_majorana_weight_counts(ps1)
        @test get_majorana_weight_probs(v1) ≈ get_majorana_weight_probs(ps1)
        topa = find_top_k(ps1, 5)
        topb = find_top_k(v1, 5)
        @test Dict(topa) == Dict(topb)
        la = largest(ps1)
        lb = largest(v1)
        @test PauliSum(lb) == la
        big = rand(PauliSum{N}; n_paulis=30)
        big[PauliBasis{N}(Int128(3), Int128(0))] = 7.0 + 0im   # dominant diagonal term
        vbig = SparsePauliVector(big)
        @test largest_diag(vbig) == largest_diag(big)
    end

    @testset "decompose" begin
        H = PauliSum(N)
        for i in 1:N-1
            H[PauliBasis(Pauli(N, X=[i, i+1]))] = 1.0
            H[PauliBasis(Pauli(N, Z=[i, i+1]))] = 1.1
        end
        vH = SparsePauliVector(H)
        g1, a1 = trotterize(H, 0.1, n_trotter=2, order=2)
        g2, a2 = trotterize(vH, 0.1, n_trotter=2, order=2)
        # iteration order differs (Dict vs sorted); the multiset must match
        @test sort(collect(zip(string.(g1), a1))) == sort(collect(zip(string.(g2), a2)))
        rng1 = Xoshiro(5)
        rng2 = Xoshiro(5)
        gq1, aq1 = qdrift(H, 0.1; n_samples=20, rng=rng1)
        gq2, aq2 = qdrift(vH, 0.1; n_samples=20, rng=rng2)
        @test length(gq1) == length(gq2)
    end

    @testset "channels" begin
        big = PauliSum(N, Float64)
        while length(big) < 40
            big[PauliBasis(rand(Pauli{N}))] = rand() - 0.5
        end
        vb = SparsePauliVector(big; T=Float64)
        for (f!, f, args) in ((pauli_channel!, pauli_channel, (0.01, 0.02, 0.03)),
                              (depolarizing_channel!, depolarizing_channel, (0.05,)),
                              (dephasing_channel!, dephasing_channel, (0.05,)),
                              (bit_flip_channel!, bit_flip_channel, (0.05,)),
                              (bit_phase_flip_channel!, bit_phase_flip_channel, (0.05,)))
            a = deepcopy(big)
            b = copy(vb)
            f!(a, args...)
            f!(b, args...)
            @test _same(a, b)
            # allocating wrappers, restricted qubit sets
            a2 = f(big, args...; qubits=[1, 3])
            b2 = f(vb, args...; qubits=[1, 3])
            @test _same(a2, b2)
            @test PauliSum(vb) == big       # input untouched by wrapper
        end
    end
end
