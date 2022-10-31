export hev, create_A_gate, hea_preserve_particle

function hev(n_qubit, depth)
    circuit = QulacsParametricQuantumCircuit(n_qubit)
    for d in 1:depth
        for i in 1:n_qubit
            add_parametric_RY_gate!(circuit, i, 0.0)
            add_parametric_RZ_gate!(circuit, i, 0.0)
        end
        for i in 1:n_qubit÷2
            add_CNOT_gate!(circuit, 2 * i - 1, 2 * i)
        end
        for i in 1:n_qubit÷2-1
            add_CNOT_gate!(circuit, 2 * i, 2 * i + 1)
        end
    end
    for i in 1:n_qubit
        add_parametric_RY_gate!(circuit, i, 0.0)
        add_parametric_RZ_gate!(circuit, i, 0.0)
    end
    circuit = QulacsVariationalQuantumCircuit(circuit)
    circuit
end

function create_A_gate(circuit, theta, phi, target_two_qubits)
    first = target_two_qubits[1]
    second = target_two_qubits[2]
    #theta1 = two_theta[1]
    #theta1 = two_theta[2]
    add_CNOT_gate!(circuit, second, first)
    add_parametric_RY_gate!(circuit, second, theta+π/2) # theta, phi
    add_parametric_RZ_gate!(circuit, second, phi+π) 
    add_CNOT_gate!(circuit, first, second)
    add_parametric_RY_gate!(circuit, second, -(theta+π/2)) #same theta & phi
    add_parametric_RZ_gate!(circuit, second, -(phi+π)) 
    add_CNOT_gate!(circuit, second, first) 
    return circuit 
end


function hea_preserve_particle(n_qubit, n_repeat)
    #
    if n_qubit <= 0 || n_qubit % 2 != 0
        error("Invalid n_qubit: $(n_qubit)")
    end
    circuit = QulacsParametricQuantumCircuit(n_qubit)
    # X_gate?
    # Whici posiotion ?
    # init state is all zero |0000000>
    for i in 1:n_qubit
        if i % 2 == 1
            continue
        end
        add_X_gate!(circuit, i) 
    end

    for j in 1:n_repeat
        for i in 1:n_qubit÷2 
            #idx = i*2
            create_A_gate(circuit, 0.0, 0.0, [2*i-1, 2*i]) #theta
        end
        for i in 1:n_qubit÷2-1
            #idx = i*2
            create_A_gate(circuit, 0.0, 0.0, [2*i, 2*i+1]) 
        end
    end
    circuit = QulacsVariationalQuantumCircuit(circuit) 
    return circuit
end



function create_A_gate_compact(circuit, theta, phi, target_two_qubits)
    first = target_two_qubits[1]
    second = target_two_qubits[2]
    #theta1 = two_theta[1]
    #theta1 = two_theta[2]
    add_CNOT_gate!(circuit, second, first)
    add_parametric_RY_gate!(circuit, second, theta+π/2) # theta, phi
    add_parametric_RZ_gate!(circuit, second, phi+π) 
    add_CNOT_gate!(circuit, first, second)
    add_parametric_RY_gate!(circuit, second, -(theta+π/2)) #same theta & phi
    add_parametric_RZ_gate!(circuit, second, -(phi+π)) 
    add_CNOT_gate!(circuit, second, first) 
    return circuit 
end



function hea_preserve_particle_compact(n_qubit, n_repeat)
    #
    if n_qubit <= 0 || n_qubit % 2 != 0
        error("Invalid n_qubit: $(n_qubit)")
    end
    circuit = QulacsParametricQuantumCircuit(n_qubit)
    # X_gate?
    # Whici posiotion ?
    # init state is all zero |0000000>
    for i in 1:n_qubit
        if i % 2 == 1
            continue
        end
        add_X_gate!(circuit, i) 
    end

    for j in 1:n_repeat
        for i in 1:n_qubit÷2 
            #idx = i*2
            create_A_gate(circuit, 0.0, 0.0, [2*i-1, 2*i]) #theta
        end
        for i in 1:n_qubit÷2-1
            #idx = i*2
            create_A_gate(circuit, 0.0, 0.0, [2*i, 2*i+1]) 
        end
    end
    circuit = QulacsVariationalQuantumCircuit(circuit) 
    return circuit
end