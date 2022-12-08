export apply_qubit_op!, get_transition_amplitude_with_obs, apply_ham!, apply_qubit_ham!

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
    circuit::VariationalQuantumCircuit, state0_bra::QuantumState;
    minimizer=mk_scipy_minimize(),
    verbose=true,
    comm=MPI_COMM_WORLD
    )
    her, antiher = divide_real_imag(op)

    function cost(thetas::Vector{Float64})
        update_circuit_param!(circuit, thetas)
        re_ = get_transition_amplitude_with_obs(circuit, state0_bra, her, state_ket)
        im_ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
        abs2(1.0 - (re_ + im_ * im))
    end
   
    thetas_init = get_thetas(circuit)
    if comm !== nothing
        MPI.Bcast!(thetas_init, 0, comm)
    end
    opt_thetas = minimizer(cost, thetas_init)
    norm_right = sqrt(get_expectation_value(hermitian_conjugated(op) * op, state_ket))
    if verbose
       println("norm_right",norm_right)
    end

    update_circuit_param!(circuit, opt_thetas)
    re__ = get_transition_amplitude_with_obs(circuit, state0_bra, her, state_ket)
    im__ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
    z = re__ + im__ * im
    if verbose
        println("Match in apply_qubit_op!: ", z/norm_right)
    end
    return z
end

function apply_qubit_ham!(
    op::QubitOperator,
    state_ket::QuantumState,
    circuit::VariationalQuantumCircuit, state0_bra::QuantumState;
    minimizer=mk_scipy_minimize(),
    verbose=true,
    comm=MPI_COMM_WORLD
    )
    #her, antiher = divide_real_imag(op)

    function cost(thetas::Vector{Float64})
        update_circuit_param!(circuit, thetas)
        re_ = get_transition_amplitude_with_obs(circuit, state0_bra, op, state_ket)
        #im_ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
        println("transition_H=", -abs(re_))
        - abs((re_ ))
    end
   
    thetas_init = get_thetas(circuit)
    if comm !== nothing
        MPI.Bcast!(thetas_init, 0, comm)
    end
    opt_thetas = minimizer(cost, thetas_init)
    norm_right = sqrt(get_expectation_value(hermitian_conjugated(op) * op, state_ket))
    if verbose
       println("norm_right",norm_right)
    end

    update_circuit_param!(circuit, opt_thetas)
    re__ = get_transition_amplitude_with_obs(circuit, state0_bra, op, state_ket)
    #im__ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
    #z = re__ + im__ * im
    z = re__
    if verbose
        println("Match in apply_qubit_op!: ", z/norm_right)
    end
    return z
end


"""
Compute <state_bra| circuit^+ obs |state_ket>, where obs is a hermite observable.
"""
function get_transition_amplitude_with_obs(
    circuit::VariationalQuantumCircuit, 
    state0_bra::QuantumState,
    op::QubitOperator,
    state_ket::QuantumState)
    state_bra = copy(state0_bra)
    update_quantum_state!(circuit, state_bra)
    return get_transition_amplitude(op, state_bra, state_ket)
end


"""
Apply a Hamiltonian to |state_ket> and fit the result with
circuit * |state_bra>.
The circuit object will be updated on exit.
The squared norm of op * |state_ket>  will be returned.
state0_bra will not be modified.
"""
function apply_ham!(
    op::QubitOperator,
    state_ket::QuantumState,
    circuit::VariationalQuantumCircuit, state0_bra::QuantumState;
    minimizer=mk_scipy_minimize(),
    verbose=true,
    comm=MPI_COMM_WORLD
    )
    #her, antiher = divide_real_imag(op)

    #options = Dict("disp" => verbose, "maxiter" => 300, "gtol" => 1e-8)

    function cost(thetas::Vector{Float64})
        update_circuit_param!(circuit, thetas)
        # amplitude = <bra(\theta)| H |ket>
        amplitude = get_transition_amplitude_with_obs(circuit, state0_bra, op, state_ket)
        abs2(1.0 - (amplitude))
    end
   
    thetas_init = get_thetas(circuit)
    if comm !== nothing
        MPI.Bcast!(thetas_init, 0, comm)
    end
    opt_thetas = minimizer(cost, thetas_init)
    # sqrt(<ket| H H |ket>)
    norm_right = sqrt(get_expectation_value(hermitian_conjugated(op) * op, state_ket))
    if verbose
       println("norm_of_H*|ket>=", norm_right)
    end

    update_circuit_param!(circuit, opt_thetas)

    square_norm = get_transition_amplitude_with_obs(circuit, state0_bra, op, state_ket)
    #im__ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
    #z = re__ + im__ * im
    if verbose
        println("Match in apply_qubit_op!: ", square_norm/norm_right)
    end
    return square_norm
end

