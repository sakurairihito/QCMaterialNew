export apply_qubit_op!, get_transition_amplitude_with_obs

"""
Divide a qubit operator into the hermite and antihermite parts.
"""
divide_real_imag(op::QubitOperator) = 
    (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im

"""
Apply a qubit operator op to |state_ket> and fit the result with
circuit * |state_bra>.
The circuit object will be updated on exit.
The squared norm of op * |state_ket>  will be returned.
state0_bra will not be modified.
"""
function apply_qubit_op!(
    op::QubitOperator,
    state_ket::QuantumState,
    circuit::VariationalQuantumCircuit, state0_bra::QuantumState,
    minimizer=mk_scipy_minimize()
    )
    her, antiher = divide_real_imag(op)

    function cost(thetas::Vector{Float64})
        update_circuit_param!(circuit, thetas)
        re_ = get_transition_amplitude_with_obs(circuit, state0_bra, her, state_ket)
        im_ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
        #println("abs2 ", thetas, " ", -abs2(re_ + im_ * im))
        #println("abs2 ", -abs2(re_ + im_ * im))
        #-abs2(re_ + im_ * im)
        abs2(1.0 - (re_ + im_ * im))
    end

    #println("circuit: ", circuit)

    #1 parameter random
    #theta_init_vcex = rand(num_theta(circuit))
    #init_theta_vcex_list = theta_init_vcex
    #println("cost_param_random_before_opt=",cost(init_theta_vcex_list))
    #opt_thetas = minimizer(cost, init_theta_vcex_list)
    #println("opt_thetas=", opt_thetas)
    #println("cost_param_rand_after_opt=", cost(opt_thetas))

    #2 parameter 0
    #println("cost_param_0_before_opt=",cost(get_thetas(circuit)))
    #println("init_parameter=", get_thetas(circuit))
    opt_thetas = minimizer(cost, get_thetas(circuit))
    #println("cost_param_0_after_opt=", cost(opt_thetas))

    norm_right = sqrt(get_expectation_value(hermitian_conjugated(op) * op, state_ket))
    println("norm_right",norm_right)

    update_circuit_param!(circuit, opt_thetas)
    re__ = get_transition_amplitude_with_obs(circuit, state0_bra, her, state_ket)
    im__ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
    z = re__ + im__ * im
    println("Match in apply_qubit_op!: ", z/norm_right)
    return z
    #-cost(opt_thetas)
end

"""
Compute <state_bra| circuit^+ obs |state_ket>, where obs is a hermite observable.
"""
function get_transition_amplitude_with_obs(
    circuit::VariationalQuantumCircuit, state0_bra::QuantumState,
    op::QubitOperator,
    state_ket::QuantumState)
    #debug
    #println("")
    #println("state_ket(Not applied circuit)=", get_vector(state_ket)) 
    #end debug 
    state_bra = copy(state0_bra)
    #println("thetas ", get_thetas(circuit))
    update_quantum_state!(circuit, state_bra)
    #debug
    #println("state_bra(applied circuit)=", get_vector(state_bra))
    #end debug
    #println("trans_amp ", get_transition_amplitude(op, state_bra, state_ket))
    #println("")
    return get_transition_amplitude(op, state_bra, state_ket)
    #debug
    #println("get_transition_amplitude=",get_transition_amplitude(op, state_bra, state_ket))
    #end debug
end