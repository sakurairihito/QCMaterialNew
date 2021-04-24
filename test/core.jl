using Test
using QCMaterial

#=
@testset "core.circuit_param" begin
    nsite = 2
    n_electron = nsite　
    n_qubit = 2*nsite 
    U = 1.0

    ham = FermionOperator()
    for i in 1:nsite
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^ $(down)^ $(up) $(down)", -U) 
    end
    println(ham)

    ham_obs = create_observable(jordan_wigner(ham), n_qubit)
    println(ham_obs)
    circuit = uccgsd(n_qubit, true)
    thetas = 1. * collect(1:num_theta(circuit))
    update_circuit_param!(circuit, thetas)
    thetas_rec = get_circuit_param(circuit)
end
=#

@testset "core.of_qubit_operator" begin
    op = OFQubitOperator("X0 X5", 1.0) + OFQubitOperator("Y1 Y2", 2.0)
    @test get_n_qubit(op) == 6
    @test get_term_count(op) == 2
    @test terms_dict(op)[((0,"X"), (5,"X"))] == 1.0
    @test terms_dict(op)[((1,"Y"), (2,"Y"))] == 2.0
    #a, i = 0, 1
    #crr, ann = 1, 0
    #generator = jordan_wigner(FermionOperator([(a, crr), (i, ann)], 1.0))
    ##println(generator.pyobj.__class__)
    #println(get_term_count(generator))
    #println(generator)
end

@testset "core.add_parametric_circuit_using_generator" begin
    n_qubit = 4
    a, i = 0, 0
    crr, ann = 1, 0
    @assert 2*crr < n_qubit && 2*ann < n_qubit
    generator = jordan_wigner(FermionOperator([(a, crr), (i, ann)], 1.0))

    circuit = QulacsParametricQuantumCircuit(n_qubit)
    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
    add_parametric_circuit_using_generator!(circuit, generator, 1.0)
    @test circuit.theta_offsets == [
        (2, 0, [0.0]),
        (2, 2, [0.0])
    ]
end

@testset "core.qubit_operator" begin
    op1 = OFQubitOperator("Z0", 1.0)
    op2 = OFQubitOperator("Z0", 1.0im)
    op3 = OFQubitOperator("X0", 1.0)

    state0 = create_hf_state(1, 0) # Create |0>
    state1 = create_hf_state(1, 1) # Create |1>

    @test is_hermitian(op1)
    @test !is_hermitian(op2)
    @test get_expectation_value(op1, state1) == -1.0
    @test get_transition_amplitude(op3, state1, state0) == 1.0
end