using PauliOperators
using LinearAlgebra
using Test
using Random
using PauliOperators: check_spv, _word_type, _pack, _unpack

@testset "SparsePauliVector core" begin
    Random.seed!(2)

    @testset "word type selection" begin
        @test _word_type(1) == UInt64
        @test _word_type(64) == UInt64
        @test _word_type(65) == UInt128
        @test _word_type(128) == UInt128
        @test_throws ErrorException _word_type(129)

        v = SparsePauliVector(4)
        @test v isa SparsePauliVector{4, UInt64, ComplexF64}
        v = SparsePauliVector(70, Float64)
        @test v isa SparsePauliVector{70, UInt128, Float64}
        @test_throws ErrorException SparsePauliVector(129)
    end

    @testset "pack/unpack round-trip incl. sign bit" begin
        for N in (8, 64, 128)
            W = _word_type(N)
            for _ in 1:20
                p = rand(PauliBasis{N})
                z, x = _pack(W, p)
                @test _unpack(PauliBasis{N}, z, x) == p
            end
        end
        # N=128 with the Int128 sign bit set
        p = PauliBasis{128}(Int128(-1), Int128(-1))   # all 128 bits on
        z, x = _pack(UInt128, p)
        @test z == typemax(UInt128)
        @test _unpack(PauliBasis{128}, z, x) == p
    end

    @testset "PauliSum round-trip" begin
        for (N, T) in ((3, ComplexF64), (8, ComplexF64), (70, ComplexF64), (128, ComplexF64))
            ps = rand(PauliSum{N}; n_paulis=30, T=T)
            v = SparsePauliVector(ps)
            @test check_spv(v)
            @test length(v) == length(ps)
            @test PauliSum(v) == ps
        end
        # real-T conversion of a real-coefficient sum
        ps = PauliSum(6, Float64)
        while length(ps) < 20
            ps[PauliBasis(rand(Pauli{6}))] = rand() - 0.5
        end
        v = SparsePauliVector(ps; T=Float64)
        @test v isa SparsePauliVector{6, UInt64, Float64}
        @test PauliSum(v) == ps
        # real-T rejection of complex coefficients
        psc = PauliSum(6)
        psc[rand(PauliBasis{6})] = 1.0 + 2.0im
        @test_throws ErrorException SparsePauliVector(psc; T=Float64)
        # convert() both ways
        ps = rand(PauliSum{5}; n_paulis=10)
        v = convert(SparsePauliVector{5, UInt64, ComplexF64}, ps)
        @test PauliSum(v) == ps
        @test convert(PauliSum{5, ComplexF64}, v) == ps
    end

    @testset "Dict-idiom parity" begin
        N = 8
        ps = rand(PauliSum{N}; n_paulis=40)
        v = SparsePauliVector(ps)

        for (p, c) in ps
            @test v[p] == c
            @test get(v, p, 0.0) == c
            @test haskey(v, p)
        end
        # getindex throws KeyError on a missing key (Dict parity)
        absent = PauliBasis("YXZIZXYZ")
        haskey(ps, absent) && delete!(ps, absent)
        haskey(v, absent) && delete!(v, absent)
        @test_throws KeyError v[absent]
        @test get(v, absent, 0.0) == 0.0
        @test !haskey(v, absent)

        # iteration yields Pair{PauliBasis,T}; order-insensitive equality
        @test eltype(v) == Pair{PauliBasis{N}, ComplexF64}
        @test Dict(collect(v)) == ps
        @test Set(collect(keys(v))) == Set(collect(keys(ps)))
        @test sort(abs.(collect(values(v)))) ≈ sort(abs.(collect(values(ps))))
        @test length(collect(pairs(v))) == length(ps)

        # setindex! insert + overwrite, sorted invariant maintained
        n0 = length(v)
        v[absent] = 2.5 + 0im
        @test check_spv(v)
        @test length(v) == n0 + 1
        @test v[absent] == 2.5 + 0im
        v[absent] = 1.5 + 0im
        @test length(v) == n0 + 1
        @test v[absent] == 1.5 + 0im

        # get! inserts default
        absent2 = PauliBasis("ZZZZZZZZ")
        haskey(v, absent2) && delete!(v, absent2)
        c = get!(v, absent2, 0.25 + 0im)
        @test c == 0.25 + 0im
        @test v[absent2] == 0.25 + 0im

        # delete! removes; no-op on missing keys
        delete!(v, absent)
        @test !haskey(v, absent)
        delete!(v, absent)      # no-op
        @test check_spv(v)

        # String indexing
        v2 = SparsePauliVector(2)
        v2[PauliBasis("XY")] = 3.0
        @test v2["XY"] == 3.0

        # filter! / filter
        vb = copy(v)
        filter!(pr -> abs(pr.second) > 0.3, vb)
        @test check_spv(vb)
        psb = PauliSum(v)
        filter!(pr -> abs(pr.second) > 0.3, psb)
        @test PauliSum(vb) == psb
        vc = filter(pr -> abs(pr.second) > 0.3, v)
        @test vc == vb
        @test vc !== v && length(v) != length(vb) || length(v) == length(vb)

        # == and copy independence
        vcp = copy(v)
        @test vcp == v
        p1 = first(keys(v))
        vcp[p1] = 99.0
        @test vcp != v
        @test v[p1] != 99.0 + 0im

        # empty!
        empty!(vcp)
        @test isempty(vcp)
        @test length(vcp) == 0
        @test check_spv(vcp)

        # size parity
        @test size(v) == size(ps)
    end

    @testset "growth through setindex!" begin
        v = SparsePauliVector(10, Float64; capacity=2)
        added = Set{PauliBasis{10}}()
        while length(added) < 50
            push!(added, rand(PauliBasis{10}))
        end
        for (i, p) in enumerate(added)
            v[p] = Float64(i)
        end
        @test length(v) == 50
        @test check_spv(v)
    end

    @testset "constructors from single Paulis and rand" begin
        p = PauliBasis("XYZ")
        v = SparsePauliVector(p)
        @test length(v) == 1
        @test v[p] == 1.0 + 0im

        pl = Pauli("XYZ")
        v = SparsePauliVector(pl)
        @test v[PauliBasis(pl)] == coeff(pl)

        v = rand(SparsePauliVector{7}; n_paulis=5)
        @test v isa SparsePauliVector{7, UInt64, ComplexF64}
        @test 1 <= length(v) <= 5
        @test check_spv(v)
        v = rand(SparsePauliVector{7, UInt64, ComplexF64}; n_paulis=5)
        @test v isa SparsePauliVector{7, UInt64, ComplexF64}
        @test check_spv(v)
    end

    @testset "adjoint wrapper" begin
        ps = rand(PauliSum{5}; n_paulis=10)
        v = SparsePauliVector(ps)
        va = v'
        @test parent(va) === v
        p = first(keys(ps))
        @test va[p] == ps[p]'
        @test Set(collect(keys(va))) == Set(collect(keys(ps)))
    end

    @testset "show smoke" begin
        v = SparsePauliVector(rand(PauliSum{3}; n_paulis=3))
        @test sprint(show, v) isa String
        @test sprint(show, MIME("text/plain"), v) isa String
        @test sprint(show, MIME("text/plain"), v') isa String
    end
end
