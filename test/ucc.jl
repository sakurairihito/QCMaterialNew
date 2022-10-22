using Test
using LinearAlgebra
using QCMaterial

import PyCall: pyimport, pybuiltin
import Random

@testset "ucc.UCCQuantumCircuit" begin
    n_qubit = 2
    n_electron = 1
    c = UCCQuantumCircuit(n_qubit)
    c_expected = UCCQuantumCircuit(QulacsParametricQuantumCircuit(n_qubit), [], [])
    pybuiltin("type") #pythonのtype関数を呼び出す
    @test pybuiltin("type")(c.circuit.pyobj) == pybuiltin("type")(c_expected.circuit.pyobj)
    @test c.circuit.pyobj.get_qubit_count() == c_expected.circuit.pyobj.get_qubit_count()
    # Kyulacsのテクニックを使えば、get_qubit_count()などの関数を自動でラップできる。
    @test c.thetas == c_expected.thetas
    @test c.theta_offsets == c_expected.theta_offsets
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
    println(" num_pauli(circuit, num_thetas)=", num_pauli(c, num_theta(c))) #2
    println("ioff=", theta_offset(c, num_theta(c)) + num_pauli(c, num_theta(c))) #2
    println("generator.pyobj=", generator.pyobj)
    println("generator.pyobj.terms=", generator.pyobj.terms)
end

@testset "ucc.add_parametric_circuit_using_generator_gen_t2_kucj_2!" begin
    n_qubit = 2
    n_electron = 1
    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    # a_2^{dag} a_2 a_1{dag} a_1
    generator = gen_t2_kucj_2(2, 1)
    generator_rm = rm_identity(generator)
    println("terms_dict(generator_rm)=", terms_dict(generator_rm))
    println("generator_rm.pyobj=", generator_rm.pyobj)
    println("generator_rm.pyobj.terms=", generator_rm.pyobj.terms)

    add_parametric_circuit_using_generator!(c, generator_rm, 1.0)
    println(" get_term_count(generator_rm)=", get_term_count(generator_rm)) #2
    #println(" terms_dict(generator)=", terms_dict(generator))  
    #terms_dict(generator)=Dict{Any, Any}(() => 0.0 + 0.25im, ((2, "Z"),) => 0.0 - 0.25im, 
    #((1, "Z"), (2, "Z")) => 0.0 + 0.25im, ((1, "Z"),) => 0.0 - 0.25im)
    println("theta_offset(c, num_theta(c))= ", theta_offset(c, num_theta(c))) #0
    println(" num_pauli(circuit, num_thetas)=", num_pauli(c, num_theta(c))) #4
    println("ioff=", theta_offset(c, num_theta(c)) + num_pauli(c, num_theta(c))) #4
    @test num_pauli(c, num_theta(c)) == 3
    #println("circuit.theta_offsets[1][3][1]=", c.theta_offsets[1][3][1])
    #println("circuit.theta_offsets[1][3][1]=", c.theta_offsets[1][3][2])
    #println("circuit.theta_offsets[1][3][1]=", c.theta_offsets[1][3][3])
    #println("circuit.theta_offsets[1][3][4]=", c.theta_offsets[1][3][4])
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
    @test vec ≈ [0.5 * theta, 1.0]
end

@testset "ucc.uccgsd" begin
    #Random.seed!(1)
    scipy_opt = pyimport("scipy.optimize")
    nsite = 2
    n_qubit = 2 * nsite
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
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron)

    enes_ed = eigvals(sparse_mat.toarray())

    ham_jw = jordan_wigner(ham)

    circuit = uccgsd(n_qubit, orbital_rot=true)
    #circuit = kucj2(n_qubit)
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
    @test abs(EigVal_min - cost_history[end]) < 1e-3
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
    vec ≈ [0.5 * theta, 1.0]
end


#@testset "ucc.gen_t2_kucj" begin
#Random.seed!(1)
#    println("gen_t2_kucj(1,1)=", gen_t2_kucj(1, 1).pyobj.terms)
#@test gen_t2_kucj(1, 1).pyobj.terms == Dict{Any,Any}()
#    println("gen_t2_kucj(2,1)=", gen_t2_kucj(2, 1).pyobj.terms)
#@test gen_t2_kucj(2, 1).pyobj.terms == Dict{Any,Any}(() => 0.5 + 0.0im, ((0, "Z"),) => -0.5 + 0.0im, ((0, "Z"), (1, "Z")) => 0.5 + 0.0im, ((1, "Z"),) => -0.5 + 0.0im)
#end

@testset "ucc.gen_t2_kucj_2" begin
    #Random.seed!(1)
    println("gen_t2_kucj_2(1,1,1,1)=", gen_t2_kucj_2(1, 1).pyobj.terms)
    #@test gen_t2_kucj_2(1, 1).pyobj.terms == Dict{Any,Any}()
    println("gen_t2_kucj_2(2,2,1,1)=", gen_t2_kucj_2(2, 1).pyobj.terms)
    #gen_t2_kucj_2(2,2,1,1)=Dict{Any, Any}(() => 0.0 + 0.25im, ((0, "Z"),) => 0.0 - 0.25im, 
    #((0, "Z"), (1, "Z")) => 0.0 + 0.25im, ((1, "Z"),) => 0.0 - 0.25im)
    #@test gen_t2_kucj_2(2, 1).pyobj.terms == Dict{Any,Any}(() => 0.5 + 0.0im, ((0, "Z"),) => -0.5 + 0.0im, ((0, "Z"), (1, "Z")) => 0.5 + 0.0im, ((1, "Z"),) => -0.5 + 0.0im)
end

@testset "ucc.gen_t2_kucj_2" begin
    n_qubit = 4
    n_electron = 1
    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t2_kucj_2(2, 1)
    generator += gen_t2_kucj_2(4, 2)
    add_parametric_circuit_using_generator!(c, generator, 1.0)
    println("num_pauli(c, 1)=", num_pauli(c, 1))
    #@test num_pauli(c, 1) == 2 #１番目のパラメータに関するパウリ行列の個数
end

@testset "ucc.num_theta" begin
    n_qubit = 4
    n_qubit_ = 2

    c = UCCQuantumCircuit(n_qubit)
    c_ = UCCQuantumCircuit(n_qubit_)
    c_test = UCCQuantumCircuit(n_qubit_)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_p_t2(1, 2, 3, 4)
    #generator = rm_identity(generator), c.theta_offsets
    println("rm_identity=", generator)
    add_parametric_circuit_using_generator!(c, generator, 1.0)
    println("num_pauli(c, 1)=", num_pauli(c, 1))
    println("thetaoffset=", c.theta_offsets)
    @test num_pauli(c, 1) == 8 #１番目のパラメータに関するパウリ行列の個数

    generator_test = OFQubitOperator("Z1", 1.0)
    generator_test += OFQubitOperator("Z2", 1.0)
    println("generator_test=", generator_test)
    add_parametric_circuit_using_generator!(c_test, generator_test, 1.0)
    println("num_pauli(c_test, 1)=", num_pauli(c_test, 1))
    println("thetaoffset_c_test=", c_test.theta_offsets)

    generator_ = gen_t2_kucj_2(2, 1)
    generator_ = rm_identity(generator_)
    #generator = rm_identity(generator), c.theta_offsets
    println("rm_identity_=", generator_)
    add_parametric_circuit_using_generator!(c_, generator_, 1.0)
    println("num_pauli(c_, 1)=", num_pauli(c_, 1))
    println("thetaoffset=", c_.theta_offsets)
    @test num_pauli(c_, 1) == 3 #１番目のパラメータに関するパウリ行列の個数
end


@testset "ucc.kucj" begin
    #Random.seed!(1)
    scipy_opt = pyimport("scipy.optimize")
    nsite = 2
    n_qubit = 2 * nsite
    U = 1.0
    t = -0.01
    μ = 0.0

    ham = generate_ham_1d_hubbard(t, U, nsite, μ)

    n_electron = 2
    @assert mod(n_electron, 2) == 0
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron)
    enes_ed = eigvals(sparse_mat.toarray())

    ham_jw = jordan_wigner(ham)
    state0 = create_hf_state(n_qubit, n_electron)
    circuit, parameterinfo = kucj(n_qubit, k=1, sparse=false)
    #println("num_theta(circuit)=", num_theta(circuit))
    pinfo = QCMaterial.ParamInfo(parameterinfo)
    theta_init = rand(pinfo.nparam)
    cost_history, thetas_opt =
    QCMaterial.solve_gs_kucj(ham_jw, circuit, state0, parameterinfo, theta_init=theta_init, verbose=true,
        comm=QCMaterial.MPI_COMM_WORLD
    )
    @test abs(minimum(enes_ed) - cost_history[end]) < 1e-3
end