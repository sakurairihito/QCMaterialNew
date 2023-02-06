using Revise
using QCMaterial
using LinearAlgebra

function makecircuit1(nqubits)
    circuit = QulacsParametricQuantumCircuit(nqubits)
    #circuit.add_H_gate(0)
    add_parametric_RX_gate!(circuit, 1, pi/3)
    return circuit
end
qc1 = makecircuit1(1)

function add_control_qubit_for_circuit(circuit, control_index, total_num_qubits)
    num_gate = get_gate_count(circuit)
    control_circuit = QulacsQuantumCircuit(total_num_qubits)
    for i in 1:num_gate
        gate_tmp = get_gate(circuit, i)
        gate_tmp = to_matrix_gate(gate_tmp)
        add_control_qubit!(gate_tmp, control_index, i)
        add_gate!(control_circuit,  gate_tmp)
    end
    return control_circuit
end

control_index = 2 #
add_control_qubit_for_circuit(qc1, [2], 2)

function mycu1()
    nqubits = 1
    circuit = makecircuit1(nqubits)
    total_num_qubits = nqubits + 1
    control_index = [2]
    control_circuit1 = add_control_qubit_for_circuit(circuit, control_index, total_num_qubits)
    #circuit_drawer(control_circuit1, "mpl") 
    state1 = QulacsQuantumState(total_num_qubits)
    gatex = X(1)
    update_quantum_state!(gatex, state1)
    #control_circuit1.update_quantum_state(state1)
    update_quantum_state!(control_circuit1, state1)
    return print(get_vector(state1))
end
@show mycu1()


function makecircuit2(nqubits)
    circuit2 = QulacsParametricQuantumCircuit(nqubits)
    add_parametric_RY_gate!(circuit2, 1, pi/3)
    add_H_gate!(circuit2, 1)
    return circuit2
end

function mycu2()
    nqubits = 1
    circuit = makecircuit2(nqubits)
    total_num_qubits = nqubits + 1
    control_index = [2]
    control_circuit2 = add_control_qubit_for_circuit(circuit, control_index,total_num_qubits)
    state1 = QulacsQuantumState(total_num_qubits)
    xgate = X(2)
    update_quantum_state!(xgate, state1)
    update_quantum_state!(control_circuit2, state1)
    return print(get_vector(state1))
end

@show mycu2()


# overall
function measure_by_sampling(phi)
    nqubits = 1
    #phi = 0.0
    total_num_qubits = nqubits + 1
    state = QulacsQuantumState(total_num_qubits)
    H_anci = H(total_num_qubits)
    update_quantum_state!(H_anci, state)
    Rz_anci = RZ(total_num_qubits, phi)
    update_quantum_state!(Rz_anci, state)
    X_anci = X(total_num_qubits)
    update_quantum_state!(X_anci, state)
    circuit = makecircuit1(nqubits)
    control_index = [total_num_qubits]
    control_circuit1 = add_control_qubit_for_circuit(circuit, control_index, total_num_qubits)
    update_quantum_state!(control_circuit1, state)
    
    circuit2 = makecircuit2(nqubits)
    control_circuit2 = add_control_qubit_for_circuit(circuit2, control_index, total_num_qubits)
    update_quantum_state!(control_circuit2, state)
    #update_quantum_state

    #circuit3 = makecircuit3(nqubits)
    #control_circuit3 = add_control_qubit_for_circuit(circuit3, control_index, total_num_qubits)
    #update_quantum_state!(control_circuit3, state)
    update_quantum_state!(H_anci, state)

    nshots = 2^10
    samples = state_sampling(state, nshots)
    estimated_amp = 0
    #mask = Int("1" + "0" * (nqubits), 2)
    mask = 2^nqubits
    for s in samples
        bitcount = count_ones(s & mask) #1のビット数を数える。
        estimated_amp += (-1)^bitcount #Z|0>->1|0>, Z|1>->-1|1>
    end
    return estimated_amp/nshots 
end

println("before measure")
phi = 0.0
real = measure_by_sampling(phi)
@show real
