using PauliOperators
using Test
using Printf
using LinearAlgebra
using Random

@testset "Multiplication" begin
    Random.seed!(1)
  
    N  = 3
    types = []
    push!(types, PauliBasis{N})
    push!(types, Pauli{N})
    push!(types, PauliSum{N, PauliOperators.word_type(N), ComplexF64})
    push!(types, DyadBasis{N})
    push!(types, Dyad{N})
    push!(types, DyadSum{N, PauliOperators.word_type(N), ComplexF64})

    for T1 in types
        for T2 in types
            for i in 1:10
                a = rand(T1)
                b = rand(T2)
                err = norm(Matrix(a)*Matrix(b) - Matrix(a*b)) < 1e-14
                if err == false
                    @show a b err
                end
                @test err
            end 
        end 
    end 
end 
    

@testset "Multiplication Vector" begin
    Random.seed!(1)
  
    N  = 3
    types1 = []
    types2 = []
    push!(types1, PauliBasis{N})
    push!(types1, Pauli{N})
    # push!(types1, PauliSum{N, PauliOperators.word_type(N), ComplexF64})
    # push!(types1, DyadBasis{N})
    # push!(types1, Dyad{N})
    # push!(types1, DyadSum{N, PauliOperators.word_type(N), ComplexF64})
    # push!(types2, Ket{N})
    push!(types2, KetSum{N})

    for T1 in types1
        for T2 in types2
            for i in 1:10
                a = rand(T1)
                b = rand(T2)
                err = norm(Matrix(a)*Vector(b) - Vector(a*b)) < 1e-14
                if err == false
                    @show a b err
                end
                @test err
            end 
        end 
    end 
end 
    

@testset "Multiplication Adjoint" begin
    Random.seed!(1)
  
    N  = 3
    types = []
    push!(types, PauliBasis{N})
    push!(types, Pauli{N})
    push!(types, PauliSum{N, PauliOperators.word_type(N), ComplexF64})
    push!(types, DyadBasis{N})
    push!(types, Dyad{N})
    push!(types, DyadSum{N, PauliOperators.word_type(N), ComplexF64})

    for T1 in types
        for T2 in types
            for i in 1:10
                a = rand(T1)
                b = rand(T2)
                err = norm(Matrix(a)*Matrix(b)' - Matrix(a*b')) < 1e-14
                if err == false
                    @show a b err
                end
                @test err
            end 
        end 
    end 
end 
    

@testset "Multiplication Adjoint2" begin
    Random.seed!(1)
  
    N  = 3
    types = []
    push!(types, PauliBasis{N})
    push!(types, Pauli{N})
    push!(types, PauliSum{N, PauliOperators.word_type(N), ComplexF64})
    push!(types, DyadBasis{N})
    push!(types, Dyad{N})
    push!(types, DyadSum{N, PauliOperators.word_type(N), ComplexF64})

    for T1 in types
        for T2 in types
            for i in 1:10
                a = rand(T1)
                b = rand(T2)
                err = norm(Matrix(a)'*Matrix(b) - Matrix(a'*b)) < 1e-14
                if err == false
                    @show a b err
                end
                @test err
            end 
        end 
    end 
end 
    

@testset "Multiplication Adjoint3" begin
    Random.seed!(1)
  
    N  = 3
    types = []
    push!(types, PauliBasis{N})
    push!(types, Pauli{N})
    push!(types, PauliSum{N, PauliOperators.word_type(N), ComplexF64})
    push!(types, DyadBasis{N})
    push!(types, Dyad{N})
    push!(types, DyadSum{N, PauliOperators.word_type(N), ComplexF64})

    for T1 in types
        for T2 in types
            for i in 1:10
                a = rand(T1)
                b = rand(T2)
                err = norm(Matrix(a)'*Matrix(b)' - Matrix(a'*b')) < 1e-14
                if err == false
                    @show a b err
                end
                @test err
            end 
        end 
    end 
end 
    

@testset "Multiplication Scalar" begin
    Random.seed!(1)
  
    N  = 2
    types = []
    push!(types, PauliBasis{N})
    push!(types, Pauli{N})
    push!(types, PauliSum{N, PauliOperators.word_type(N), ComplexF64})
    push!(types, DyadBasis{N})
    push!(types, Dyad{N})
    push!(types, DyadSum{N, PauliOperators.word_type(N), ComplexF64})

    for T1 in types
        for i in 1:10
            a = rand(T1)
            b = rand()
            err = norm(Matrix(a)*b - Matrix(a*b)) < 1e-14
            if err == false
                @show a b err
            end
            @test err
            err = norm(Matrix(a')*b - Matrix(a'*b)) < 1e-14
            if err == false
                @show a b err
            end
            @test err
        end 
    end 
end 
    

@testset "Multiplication otimes" begin
    Random.seed!(1)
  
    N  = 2
    types = []
    push!(types, PauliBasis{N})
    push!(types, Pauli{N})
    push!(types, PauliSum{N, PauliOperators.word_type(N), ComplexF64})
    push!(types, DyadBasis{N})
    push!(types, Dyad{N})
    push!(types, DyadSum{N, PauliOperators.word_type(N), ComplexF64})

    for T1 in types
        for i in 1:10
            a = rand(T1)
            b = rand(T1)
            err = norm(kron(Matrix(b),Matrix(a)) - Matrix(a⊗b)) < 1e-14
            if err == false
                @show a b err
            end
            @test err
            # err = norm(Matrix(a')*b - Matrix(a'*b)) < 1e-14
            # if err == false
            #     @show a b err
            # end
            # @test err
        end 
    end 
end 
    
@testset "Multiplication ad hoc" begin

    N = 4
    a = rand(KetSum{N}, n_terms=50)
    @test norm(Vector(a*3.4) - 3.4*Vector(a)) < 1e-15
    @test norm(Vector(3.4*a) - 3.4*Vector(a)) < 1e-15
    @test norm(Vector(a/3.4) - Vector(a)/3.4) < 1e-15
end

@testset "Multiplication PauliSum-KetSum" begin
    Random.seed!(1)

    N = 4
    for i in 1:10
        O = rand(PauliSum{N, PauliOperators.word_type(N), ComplexF64}, n_paulis=10)
        v = rand(KetSum{N}, n_terms=5)

        σ = O*v
        @test σ isa KetSum{N, PauliOperators.word_type(N), ComplexF64}
        @test norm(Matrix(O)*Vector(v) - Vector(σ)) < 1e-14

        # SparsePauliVector goes through the same AnyPauliSum method
        Os = SparsePauliVector(O)
        @test norm(Vector(Os*v) - Vector(σ)) < 1e-14

        # matrix_element(KetSum, AnyPauliSum, KetSum) computes ⟨b|O|k⟩ via O*k
        b = rand(KetSum{N}, n_terms=5)
        me = matrix_element(b, O, v)
        @test abs(Vector(b)'*Matrix(O)*Vector(v) - me) < 1e-13
    end
end

@testset "inner product" begin
    Random.seed!(1)
  
    N  = 5

    for i in 1:10
        a = rand(PauliSum{N, PauliOperators.word_type(N), ComplexF64}, n_paulis=10)
        b = rand(PauliSum{N, PauliOperators.word_type(N), ComplexF64}, n_paulis=12)
        err = tr(Matrix(a)'*Matrix(b))/2^N - inner_product(a,b)
        @test abs(err) < 1e-12
        
        a = rand(KetSum{N, PauliOperators.word_type(N), ComplexF64}, n_terms=10)
        b = rand(KetSum{N, PauliOperators.word_type(N), ComplexF64}, n_terms=12)
        err = Vector(a)'*Vector(b) - inner_product(a,b)
        @test abs(err) < 1e-12
    end
end 