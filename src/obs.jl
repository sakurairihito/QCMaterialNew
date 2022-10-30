export num_op, dabron, dabron_exact,num_op_up,num_op_down

## computing bogoron
function num_op(site)
    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)
    op = FermionOperator()
    op += FermionOperator("$(up_index(site))^ $(up_index(site)) $(down_index(site))^  $(down_index(site))", 1.0) #n_1up x n_1down
    return op
end


function num_op_up(site)
    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)
    op = FermionOperator()
    op += FermionOperator("$(up_index(site))^ $(up_index(site))", 1.0) #n_1up x n_1down
    return op
end

function num_op_down(site)
    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)
    op = FermionOperator()
    op += FermionOperator(" $(down_index(site))^  $(down_index(site))", 1.0) #n_1up x n_1down
    return op
end

function dabron(vc, theta_list, numop, state0)
    update_circuit_param!(vc, theta_list)
    state = copy(state0)
    update_quantum_state!(vc, state)
    return get_expectation_value(jordan_wigner(numop), state)
end

# testをする。loadをする
function dabron_exact(numop, n_qubit, vec)
    state_exact = QulacsQuantumState(n_qubit) 
    state_load!(state_exact, vec)
    return get_expectation_value(jordan_wigner(numop), state_exact)
end