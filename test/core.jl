using Test
using QCMaterial

@testset "core.fermionopertor" begin
    op1 = FermionOperator("1^")
    op2 = FermionOperator("1")

    @test op1 == op1
    @test op1 != op2
    @test op1 * op2 == FermionOperator("1^ 1")

    @test FermionOperator("", 0.0) == FermionOperator()
end

@testset "core.of_qubit_operator" begin
    op = OFQubitOperator("X1 X6", 1.0) + OFQubitOperator("Y2 Y3", 2.0)
    @test get_n_qubit(op) == 6
    @test get_term_count(op) == 2
    @test terms_dict(op)[((1,"X"), (6,"X"))] == 1.0
    @test terms_dict(op)[((2,"Y"), (3,"Y"))] == 2.0
end


@testset "core.qulacs_quantum_circuit" begin
    n_qubit = 2
    state = QulacsQuantumState(n_qubit, 0b00)
    # Initial state |00>
    #  (1) H gate to 1-th qubit
    #  (2) CNOT gate (control 1, target 2)
    #  (3) X gate to 1-th qubit
    # Final state (|01> + |10>)/sqrt(2)
    circuit = QulacsQuantumCircuit(n_qubit)
    add_H_gate!(circuit, 1)
    add_CNOT_gate!(circuit, 1, 2)
    add_X_gate!(circuit, 1)
    update_quantum_state!(circuit, state)

    @test get_vector(state) ≈ [0, 1, 1, 0]/sqrt(2)
end


@testset "core.qulacs_quantum_state" begin
    n_qubit = 2

    # Initialize to 0b0
    state = QulacsQuantumState(n_qubit)
    state2 = QulacsQuantumState(n_qubit, 0b0)
    @test get_vector(state) == get_vector(state2)

    # Initialize to 0b01
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b01)
    state2 = QulacsQuantumState(n_qubit, 0b01)
    @test get_vector(state) == get_vector(state2)

    # Test inner_product
    state = QulacsQuantumState(n_qubit, 0b01)
    @test inner_product(state, state) ≈ 1.0
end

@testset "core.add_parametric_circuit_using_generator" begin
    n_qubit = 4
    a, i = 1, 1
    crr, ann = 1, 0
    @assert 1 <= 2*a <= n_qubit && 1 <= 2*i <= n_qubit
    generator = jordan_wigner(FermionOperator([(a, crr), (i, ann)], 1.0))

    circuit = UCCQuantumCircuit(n_qubit)
    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
    add_parametric_circuit_using_generator!(circuit, generator, 1.0)
    @test circuit.theta_offsets == [
        (2, 0, [0.0]),
        (2, 2, [0.0])
    ]
end

@testset "core.qubit_operator" begin
    op1 = OFQubitOperator("Z1", 1.0)
    op2 = OFQubitOperator("Z1", 1.0im)
    op3 = OFQubitOperator("X1", 1.0)

    state0 = create_hf_state(1, 0) # Create |0>
    state1 = create_hf_state(1, 1) # Create |1>

    @test is_hermitian(op1)
    @test !is_hermitian(op2)
    @test get_expectation_value(op1, state1) == -1.0
    @test get_transition_amplitude(op3, state1, state0) == 1.0

    @test OFQubitOperator("X1") * OFQubitOperator("Y1") == OFQubitOperator("Z1", 1.0im)
    @test OFQubitOperator("X1 Y2") * OFQubitOperator("Y1 Z2") == OFQubitOperator("Z1 X2", -1.0)
end