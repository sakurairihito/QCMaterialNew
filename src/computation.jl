export apply_qubit_op, get_transition_amplitude_with_obs

"""
Divide a qubit operator into the hermite and antihermite parts.
"""
divide_real_imag(op::QubitOperator) = 
    (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im

"""
Apply a qubit operator op to |state_ket> and fit the result with
circuit * |state_bra>.
The circuit object will be updated on exit.
The squared norm of op * |state_ket> will be returned.
state0_bra will not be modified.
"""
function apply_qubit_op(
    op::QubitOperator,
    state_ket::QuantumState,
    circuit::UCCQuantumCircuit, state0_bra::QuantumState,
    minimizer=mk_scipy_minimize()
    )
    n_qubit = get_n_qubit(state0_bra)
    her, antiher = divide_real_imag(op)

    function cost(thetas::Vector{Float64})
        update_circuit_param!(circuit, thetas)
        re_ = get_transition_amplitude_with_obs(circuit, state0_bra, her, state_ket)
        im_ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
        -abs2(re_ + im_ * im)
    end

    opt_thetas = minimizer(cost, copy(circuit.thetas))
    -cost(opt_thetas)
end

"""
Compute <state_bra| circuit^+ obs |state_ket>, where obs is a hermite observable.
"""
function get_transition_amplitude_with_obs(
    circuit::VariationalQuantumCircuit, state0_bra::QuantumState,
    op::QubitOperator,
    state_ket::QuantumState)
    state_bra = copy(state0_bra)
    update_quantum_state!(circuit, state_bra)
    get_transition_amplitude(op, state_bra, state_ket)
end