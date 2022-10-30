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
    add_parametric_RY_gate!(circuit, second, theta+π/2) 
    add_parametric_RZ_gate!(circuit, second, phi+π) 
    add_CNOT_gate!(circuit, first, second)
    add_parametric_RY_gate!(circuit, second, -(theta+π/2))
    add_parametric_RZ_gate!(circuit, second, -(phi+π)) 
    add_CNOT_gate!(circuit, second, first) 
    return circuit 
end


function hea_preserve_particle(n_qubit, n_repeat, thetas, phi, target_two_qubits)
    #
    if n_qubit <= 0 || n_qubit % 2 != 0
        error("Invalid n_qubit: $(n_qubit)")
    end

    circuit = QulacsParametricQuantumCircuit(n_qubit)
    # X_gate?
    # Whici posiotion ?
    # init state is all zero
    
    #for j in 1:n_repeat
    for i in 1:n_qubit÷2 
        #idx = i*2
        create_A_gate(circuit, thetas[idx-1], thetas[idx], [2*i-1, 2*i]) #奇妙だな。。
        create_A_gate(circuit, thetas[idx+1], thetas[idx+2], [2, 4])
        create_A_gate(circuit, thetas[idx+3], thetas[idx+4], [2, 3])   
    end
    #end
    # 4量子ビットの例
    #create_A_gate(circuit, thetas[1], thetas[2], [1, 3])
    #create_A_gate(circuit, thetas[3], thetas[4], [2, 4]) 
    #create_A_gate(circuit, thetas[5], thetas[6], [2, 3])  
    return circuit
end