using Test
using QCMaterial

@testset "util.count_qubit_in_qubit_operator" begin
    num_of_pauli_term_ref = 4
    op = jordan_wigner(FermionOperator("1^"))
    #OFQubitOperator has no field terms
    op = OFQubitOperator("X1 X2", 1.0) + OFQubitOperator("Y3 Y4", 2.0)
    a=2
    i=1
    #generator = FermionOperator([(a, 1), (i, 0)], 1.0)
    num_of_pauli_term = count_qubit_in_qubit_operator(op.pyobj)
    @test all(num_of_pauli_term_ref .==num_of_pauli_term)
end


@testset "util.topylist" begin
    org_array = [1, 2, 3.0]
    pylist = topylist(org_array)
    @test all(org_array .== pylist)
end

@testset "util.doublefunc" begin
    org_array = [1, 2, 3]
    @test all(doublefunc(org_array) .== 2 .* org_array)
end

@testset "util.numerical_grad" begin
    # Some test for numerical_grad
    f(x) = x[1] + 2*x[2]
    deriv = numerical_grad(f, zeros(2))
    @assert deriv ≈ [1.0, 2.0]
end


#@testset "util.parse_of_general_operators" begin
    
#end


@testset "util.mk_scipy_minimize" begin
    # Some test for numerical_grad
    f(x) = x[1] + 2*x[2]
    deriv = numerical_grad(f, zeros(2))
    @assert deriv ≈ [1.0, 2.0]
end
