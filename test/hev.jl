using Test
using QCMaterial

@testset "create_A_gate" begin
    n_qubit = 2
    # Prepare |Psi> = |00> 
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b01) #X1|00> = |01>
    theta, phi = 0.0, 0.0
    target_two_qubits = [1, 2]
    circuit = QulacsParametricQuantumCircuit(n_qubit)
    circuit_A = create_A_gate(circuit, theta, phi, target_two_qubits) 
    #量子回路を作るときは、QulacsParametricQuantumCircuitにする。
    circuit = QulacsVariationalQuantumCircuit(circuit_A) #非常にややこしい。。
    # update_quantum_stateをする前にQVQCに変換する
    update_quantum_state!(circuit, state)
    state_res = QulacsQuantumState(n_qubit)
    set_computational_basis!(state_res, 0b10) #X1|00> = |01>
    @test isapprox(-1.0*get_vector(state_res), get_vector(state)) 
end



