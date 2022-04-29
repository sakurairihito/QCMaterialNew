using Test
using LinearAlgebra
using QCMaterial

import PyCall: pyimport
import Random


@testset "ucc.UCCQuantumCircuit" begin
    n_qubit = 2
    n_electron = 1
    c = UCCQuantumCircuit(n_qubit)
    #@test c == UCCQuantumCircuit(QulacsParametricQuantumCircuit(n_qubit), [], [])
end

@testset "ucc.num_theta" begin
    n_qubit = 2
    n_electron = 1

    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 1.0)
    @test num_theta(c) == 1
end

@testset "ucc.get_thetas" begin
    n_qubit = 2
    n_electron = 1

    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 1.0)
    @test get_thetas(c) == [1.0]
end

@testset "ucc.num_pauli" begin
    n_qubit = 2
    n_electron = 1

    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 1.0)
    @test num_pauli(c, 1) == 2 #１番目のパラメータに関するパウリ行列の個数
end


@testset "ucc.pauli_coeff" begin
    n_qubit = 2
    n_electron = 1
    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 1.0)
    @test pauli_coeff(c, 1, 1) == -0.5 # X2 Y1
    @test pauli_coeff(c, 1, 2) == 0.5 # X1 Y2 
end


@testset "ucc.theta_offset" begin
    n_qubit = 4
    n_electron = 2
    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    generator2 = gen_t1(3, 4) 
    add_parametric_circuit_using_generator!(c, generator, 1.0)
    add_parametric_circuit_using_generator!(c, generator2, 1.0)
    @test theta_offset(c, 1) == 0 #先頭から数えた場合の項の数　０、１
    @test theta_offset(c, 2) == 2 #２、３
end


@testset "ucc.Base.copy" begin
    n_qubit = 2
    n_electron = 1

    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 1.0)
    c_ = copy(c)
    @test get_thetas(c_) == get_thetas(c)
    @test num_pauli(c_, 1) == num_pauli(c, 1)
    @test theta_offset(c_, 1) == theta_offset(c, 1)
end


@testset "ucc.add_parametric_circuit_using_generator!" begin
    n_qubit = 2
    n_electron = 1
    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 1.0)
    println(" get_term_count(generator)=", get_term_count(generator)) #2
    println(" terms_dict(generator)=", terms_dict(generator))  #Dict{Any, Any}(((1, "Y"), (2, "X")) => 0.0 - 0.5im, ((1, "X"), (2, "Y")) => 0.0 + 0.5im)
    println("theta_offset(c, num_theta(c))= ", theta_offset(c, num_theta(c))) #0
    println(" num_pauli(circuit, num_thetas)=",  num_pauli(c, num_theta(c))) #2
    println("ioff=", theta_offset(c, num_theta(c)) + num_pauli(c, num_theta(c))) #2
end

@testset "ucc.update_circuit_param!" begin
    n_qubit = 2
    n_electron = 1
    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 1.0)
    update_circuit_param!(c, [2.0])
    @test get_thetas(c) == [2.0]
end

@testset "ucc.update_quantum_state!" begin
    n_qubit = 1
    theta = 1e-5
    c = UCCQuantumCircuit(n_qubit)
    add_parametric_multi_Pauli_rotation_gate!(
            c.circuit, [1], [pauli_Y], theta)
    state = QulacsQuantumState(n_qubit, 0b1)
    update_quantum_state!(c, state)
    vec = get_vector(state)
    @test vec ≈ [0.5*theta, 1.0]
end


@testset "ucc.uccgsd" begin
    #Random.seed!(1)
    scipy_opt = pyimport("scipy.optimize")

    nsite = 2 
    n_qubit = 2*nsite 
    U = 1.0
    t = -0.01
    μ = 0

    ham = FermionOperator()
    for i in 1:nsite
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^ $(down)^ $(up) $(down)", -U) 
    end

    for i in 1:nsite-1
        ham += FermionOperator("$(up_index(i+1))^ $(up_index(i))", t) 
        ham += FermionOperator("$(up_index(i))^ $(up_index(i+1))", t) 
        ham += FermionOperator("$(down_index(i+1))^ $(down_index(i))", t) 
        ham += FermionOperator("$(down_index(i))^ $(down_index(i+1))", t) 
    end

    for i in 1:nsite
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^  $(up) ", -μ) 
        ham += FermionOperator("$(down)^ $(down)", -μ)
    end

    n_electron = 2　
    @assert mod(n_electron, 2) == 0
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron);　

    enes_ed = eigvals(sparse_mat.toarray());　

    ham_jw = jordan_wigner(ham)

    circuit = uccgsd(n_qubit, orbital_rot=true)

    function cost(theta_list)
        state = create_hf_state(n_qubit, n_electron)
        update_circuit_param!(circuit, theta_list) 
        update_quantum_state!(circuit, state) 
        get_expectation_value(ham_jw, state) 
    end

    theta_init = rand(num_theta(circuit))
    cost_history = Float64[] 
    init_theta_list = theta_init
    push!(cost_history, cost(init_theta_list))
    

    method = "BFGS"
    options = Dict("disp" => true, "maxiter" => 200, "gtol" => 1e-5)
    callback(x) = push!(cost_history, cost(x))
    opt = scipy_opt.minimize(cost, init_theta_list, method=method, callback=callback)
    println("Eigval_vqe=", cost_history[end])
    EigVal_min = minimum(enes_ed)
    println("EigVal_min=", EigVal_min)
    @test abs(EigVal_min-cost_history[end]) < 1e-3
end


@testset "ucc.UCCQuantumCircuit" begin
    n_qubit = 4
    c = uccgsd(n_qubit, orbital_rot=true)
    c.thetas .= 1.0
    c_copy = copy(c)
    @test all(c_copy.thetas == c.thetas)
end

@testset "ucc.one_rotation_gate" begin
    using LinearAlgebra
    n_qubit = 1
    theta = 1e-5

    c = UCCQuantumCircuit(n_qubit)
    add_parametric_multi_Pauli_rotation_gate!(
            c.circuit, [1], [pauli_Y], theta)

    state = QulacsQuantumState(n_qubit, 0b1)
    update_quantum_state!(c, state)
    vec = get_vector(state)
    vec ≈ [0.5*theta, 1.0]
end