using PauliOperators
using LinearAlgebra
using Test
using Random
using BenchmarkTools
using PauliOperators: word_type, _nbit_mask, _to_word, _majorana_weight_bits

# Reference Majorana weight: per-bit scan (the pre-optimization algorithm).
function _naive_majorana_weight(p::PauliBasis{N}) where N
    w = 0
    control = true
    for i in N:-1:1
        x = (p.x >> (i-1)) & 1 == 1
        z = (p.z >> (i-1)) & 1 == 1
        if x            # X or Y site
            w += 1
            control = !control
        elseif z        # Z-only site
            control && (w += 2)
        else            # I site
            control || (w += 2)
        end
    end
    return w
end

@testset "large N (word-boundary sweep)" begin
    Random.seed!(7)

    boundary_Ns = (63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024)

    @testset "construction, rand, string round-trip (N=$N)" for N in boundary_Ns
        W = word_type(N)

        # all-bits-on via the legacy signed convention and via typemax
        p = PauliBasis{N}(-1, -1)
        @test p isa PauliBasis{N,W}
        @test p.z == _nbit_mask(W, N) && p.x == _nbit_mask(W, N)
        @test count_ones(p.z) == N
        @test string(p) == "Y"^N

        # string constructor round-trip
        s = join(rand(['I','X','Y','Z'], N))
        pb = PauliBasis(s)
        @test pb isa PauliBasis{N,W}
        @test string(pb) == s
        pl = Pauli(s)
        @test pl isa Pauli{N,W}
        @test PauliBasis(pl) == pb

        # rand hits high bits (probability of failure ~ 2^-60 per draw)
        @test any(rand(PauliBasis{N}).z >> (N - 40) != 0 for _ in 1:10)
        r = rand(Pauli{N})
        @test r isa Pauli{N,W}
        @test r.z & ~_nbit_mask(W, N) == zero(W)

        # Ket round-trip incl. masking semantics
        k = Ket{N}(-1)
        @test k isa Ket{N,W}
        @test count_ones(k.v) == N
        @test Ket(N, 5).v == W(5)
        @test rand(Ket{N}) isa Ket{N,W}
    end

    @testset "algebra matches small-N reference embedded in high bits (N=$N)" for N in (129, 200, 256, 513, 1000)
        W = word_type(N)
        # take random 10-qubit Paulis and embed them at the top of the register
        shift = N - 10
        for _ in 1:20
            a = rand(PauliBasis{10}); b = rand(PauliBasis{10})
            A = PauliBasis{N}(W(a.z) << shift, W(a.x) << shift)
            B = PauliBasis{N}(W(b.z) << shift, W(b.x) << shift)
            # products carry the same phase and shifted bitstrings
            ab = Pauli(a) * Pauli(b)
            AB = Pauli(A) * Pauli(B)
            @test AB.s == ab.s
            @test AB.z == W(ab.z) << shift && AB.x == W(ab.x) << shift
            @test commute(A, B) == commute(a, b)
            @test weight(A) == weight(a)
            @test x_weight(A) == x_weight(a)
            @test symplectic_phase(A) == symplectic_phase(a)
        end
    end

    @testset "majorana_weight vs per-bit scan (N=$N)" for N in (64, 128, 129, 200, 511, 1024)
        for _ in 1:30
            p = rand(PauliBasis{N})
            @test majorana_weight(p) == _naive_majorana_weight(p)
        end
        @test majorana_weight(PauliBasis{N}(0, 0)) == 0        # identity
        @test majorana_weight(PauliBasis{N}(-1, -1)) == _naive_majorana_weight(PauliBasis{N}(-1, -1))
    end

    @testset "Dict insert/lookup/equality at N=$N" for N in (200, 1024)
        ps = PauliSum(N)
        seen = Set{PauliBasis{N,word_type(N)}}()
        for i in 1:200
            p = rand(PauliBasis{N})
            ps[p] = ComplexF64(i)
            push!(seen, p)
        end
        @test length(ps) == length(seen)
        for p in seen
            p2 = PauliBasis{N}(p.z | zero(word_type(N)), p.x)   # reconstructed key
            @test haskey(ps, p2)
        end
    end

    @testset "otimes width crossing" begin
        a64 = rand(PauliBasis{64});  b64 = rand(PauliBasis{64})
        c = a64 ⊗ b64
        @test c isa PauliBasis{128, UInt128}
        @test c.z == UInt128(a64.z) | UInt128(b64.z) << 64

        a100 = rand(PauliBasis{100}); b100 = rand(PauliBasis{100})
        c = a100 ⊗ b100
        @test c isa PauliBasis{200, PauliOperators.UInt256}
        @test string(c) == string(a100) * string(b100)

        a512 = rand(PauliBasis{512}); b512 = rand(PauliBasis{512})
        c = a512 ⊗ b512
        @test c isa PauliBasis{1024, PauliOperators.UInt1024}
        @test string(c) == string(a512) * string(b512)

        k = rand(Ket{100}) ⊗ rand(Ket{100})
        @test k isa Ket{200, PauliOperators.UInt256}
    end

    @testset "jordan_wigner at large N" begin
        for (f, N) in ((200, 400), (77, 200))
            adag = jordan_wigner(f, N)
            @test length(adag) == 2
            # (a†)² = 0
            sq = adag * adag
            coeff_clip!(sq, 1e-14)
            @test isempty(sq)
            # a†a + aa† = I  (a has conjugated coefficients on the same basis)
            a_op = deepcopy(adag)
            map!(conj, values(a_op))
            acomm = anticommutator(adag, a_op)
            coeff_clip!(acomm, 1e-14)
            @test length(acomm) == 1
            @test acomm[PauliBasis{N}(0, 0)] ≈ 1.0
        end
    end

    @testset "Dict evolve! at N=200" begin
        N = 200
        H = PauliSum(N, Float64)
        for i in 1:N-1
            H[PauliBasis(Pauli(N, X=[i, i+1]))] = 1.0
            H[PauliBasis(Pauli(N, Z=[i]))] = 0.7
        end
        O = PauliSum(N)
        O[PauliBasis(Pauli(N, Z=[100]))] = 1.0 + 0im
        gens, angs = trotterize(H, 0.05, n_trotter=2, order=2)
        Ot = evolve(O, gens, angs; truncation=CoeffTruncation(1e-8))
        @test length(Ot) > 1
        @test ishermitian(Ot)
        # expectation against a product state stays real and bounded
        ψ = Ket{N}(0)
        ev = expectation_value(Ot, ψ)
        @test abs(imag(ev)) < 1e-10
        @test abs(ev) <= 1.0 + 1e-9
    end

    @testset "SPV evolve!/truncate! equivalence at N=$N" for N in (200, 1000)
        H = PauliSum(N, Float64)
        for i in 1:20
            H[PauliBasis(Pauli(N, X=[i, i+1]))] = 1.0
            H[PauliBasis(Pauli(N, Z=[i]))] = 1.1
        end
        gens, angs = trotterize(H, 0.1, n_trotter=1, order=1)
        O = PauliSum(N, Float64)
        O[PauliBasis(Pauli(N, Z=[10]))] = 1.0
        # Dict path
        Od = deepcopy(O)
        for (g, a) in zip(gens, angs)
            evolve!(Od, g, a)
            coeff_clip!(Od, 1e-12)
        end
        # SPV path
        v = SparsePauliVector(O; T=Float64)
        evolve!(v, gens, angs; truncation=CoeffTruncation(1e-12))
        back = PauliSum(v)
        for (p, c) in Od
            @test isapprox(get(back, p, 0.0), real(c); atol=1e-8)
        end
        ψ = rand(Ket{N})
        @test expectation_value(v, ψ) ≈ real(expectation_value(Od, ψ)) atol=1e-10
    end

    @testset "type stability and zero allocations at N=200 (UInt256)" begin
        N = 200
        p1 = rand(Pauli{N}); p2 = rand(Pauli{N})
        @inferred p1 * p2
        @inferred PauliBasis{N}(p1.z, p1.x)
        @inferred rand(PauliBasis{N, PauliOperators.UInt256})
        b1 = PauliBasis(p1); b2 = PauliBasis(p2)
        @test (@ballocated $p1 * $p2) == 0
        @test (@ballocated commute($b1, $b2)) == 0
        @test (@ballocated majorana_weight($b1)) == 0

        # SPV rotation kernel stays zero-alloc on UInt256 words
        O = PauliSum(N, Float64)
        while length(O) < 100
            O[PauliBasis(rand(Pauli{N}))] = rand() - 0.5
        end
        v = SparsePauliVector(O; T=Float64, capacity_factor=50.0, append_factor=2.0)
        G = PauliBasis(rand(Pauli{N}))
        evolve!(v, G, 0.1)     # warm-up
        @test (@ballocated evolve!($v, $G, 0.1) evals = 1) == 0
        ψ = rand(Ket{N})
        @test (@ballocated expectation_value($v, $ψ)) == 0
    end

    @testset "negative-Integer constructor semantics" begin
        # two's complement reinterpret, masked to N bits
        @test PauliBasis{200}(-1, 0).z == _nbit_mask(PauliOperators.UInt256, 200)
        @test PauliBasis{64}(Int128(-1), 0).z == typemax(UInt64)
        @test Ket{129}(-2).v == _nbit_mask(PauliOperators.UInt256, 129) - 1
    end

    @testset "word too narrow / N too large" begin
        @test_throws ArgumentError PauliBasis{200, UInt64}(UInt64(1), UInt64(1))
        @test_throws ArgumentError word_type(1025)
        @test_throws ArgumentError rand(PauliSum{2000})
    end
end
