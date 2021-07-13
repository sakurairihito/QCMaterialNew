using LinearAlgebra
#Aの計算
"""
Compute <phi (theta_bra) | phi(theta_ket)>
"""
function overlap(vc::VariationalQuantumCircuit, state0::QulacsQuantumState,
    thetas_left::Vector{Float64}, thetas_right::Vector{Float64})

    circ_tmp = copy(vc)

    # Compute state_left
    update_circuit_param!(circ_tmp, thetas_left)
    state_left = copy(state0)
    update_quantum_state!(circ_tmp, state_left)

    # Compute state_right
    update_circuit_param!(circ_tmp, thetas_right)
    state_right = copy(state0)
    update_quantum_state!(circ_tmp, state_right)

    res = inner_product(state_left, state_right)
    res
end

function compute_A(vc::VariationalQuantumCircuit, state0::QulacsQuantumState, delta_theta=1e-8;
    comm=MPI_COMM_WORLD)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end

    num_thetas = num_theta(vc)
    thetas = get_thetas(vc)

    A = zeros(Complex{Float64}, num_thetas, num_thetas)
    j_start, j_local_size = distribute(num_thetas, MPI_size, MPI_rank)
    for j in j_start:j_start+j_local_size-1
        thetas_j = copy(thetas)
        thetas_j[j] += delta_theta
        for i in 1:num_thetas
            thetas_i = copy(thetas)
            thetas_i[i] += delta_theta
            A[i, j] = real(
                      overlap(vc, state0, thetas_i, thetas_j)
                    - overlap(vc, state0, thetas_i, thetas, )
                    - overlap(vc, state0, thetas,   thetas_j)
                    + overlap(vc, state0, thetas,   thetas, )
                )/delta_theta^2
        end
    end
    if comm === nothing
        return A
    else
        return Allreduce(A, MPI.SUM, comm)
    end
end

#Cの計算
"""
Compute <phi (theta_bra) |H| phi(theta_ket)>
"""
function transition(op::OFQubitOperator, vc::VariationalQuantumCircuit, state0::QulacsQuantumState,
    thetas_left::Vector{Float64}, thetas_right::Vector{Float64})

    circ_tmp = copy(vc)

    # Compute state_left
    update_circuit_param!(circ_tmp, thetas_left)
    state_left = copy(state0)
    update_quantum_state!(circ_tmp, state_left)

    # Compute state_right
    update_circuit_param!(circ_tmp, thetas_right)
    state_right = copy(state0)
    update_quantum_state!(circ_tmp, state_right)
    
    # Compute <state_right|H|state_right>
    res = get_transition_amplitude(op, state_left, state_right)
    res
end


function compute_C(op::OFQubitOperator, vc::VariationalQuantumCircuit,
    state0::QulacsQuantumState, delta_theta=1e-8)
    num_thetas = num_theta(vc)
    thetas = get_thetas(vc)

    C = zeros(Complex{Float64}, num_thetas)
    for i in 1:num_thetas
        thetas_i = copy(thetas)
        thetas_i[i] += delta_theta

        thetas_i2 = copy(thetas)
        thetas_i2[i] -= delta_theta

        C[i] = -real(
            transition(op, vc, state0, thetas, thetas_i)
            -transition(op, vc, state0, thetas, thetas_i2)
        )/(2*delta_theta)
    end
    C
end



#theta(tau)の微分の計算
"""
Compute thetadot = A^(-1) C
"""

function compute_thetadot(op::OFQubitOperator, vc::VariationalQuantumCircuit,
    state0::QulacsQuantumState,delta_theta=1e-8; comm=MPI_COMM_WORLD)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end
    A = compute_A(vc, state0, delta_theta; comm=comm)
    C = compute_C(op, vc, state0, delta_theta)
    thetadot, r = LinearAlgebra.LAPACK.gelsy!(A, C)
    thetadot
end


"""
Perform imaginary-time evolution.

ham_op:
    Hamiltonian
vc:
    Variational circuit. The current value of variational parameters are
    used as the initial value of the imaginary-time evolution.
state0:
    The initial state to which the Variational circuit is applied to
taus:
    list of imaginary times in ascending order
    The first element must be 0.0. 
return:
    list of variational parameters at the given imaginary times.
"""
function imag_time_evolve(ham_op::OFQubitOperator, vc::VariationalQuantumCircuit, state0::QulacsQuantumState,
    taus::Vector{Float64}, delta_theta=1e-8;
    comm=MPI_COMM_WORLD
    )::Tuple{Vector{Vector{Float64}}, Vector{Float64}}
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end
    if taus[1] != 0.0
        error("The first element of taus must be 0!")
    end
    thetas_tau = [copy(get_thetas(vc))]
    log_norm_tau = zeros(Float64, length(taus))  #エルミートの期待値だから必ず実数

    for i in 1:length(taus)-1
        vc_ = copy(vc)
        update_circuit_param!(vc_, thetas_tau[i])
        #compute expectation value
        state0_ = copy(state0)
        update_quantum_state!(vc_, state0_)

        # Compute theta 
        thetas_dot_ = compute_thetadot(ham_op, vc_, state0, delta_theta, comm=comm)
        if taus[i+1] <= taus[i]
            error("taus must be in strictly asecnding order!")
        end
        thetas_next_ = thetas_tau[i] + (taus[i+1] - taus[i]) * thetas_dot_
        push!(thetas_tau, thetas_next_)

        # Compute norm
        log_norm_tau[i+1] = log_norm_tau[i] - get_expectation_value(ham_op, state0_) * (taus[i+1] - taus[i])

    end
    thetas_tau, log_norm_tau
end


function _create_quantum_state(c, state0)
    state0_ = copy(state0)
    update_quantum_state!(c, state0_)
    state0_
end

function _create_quantum_state(c, theta::Vector{Float64}, state0::QuantumState)
    c_ = copy(c)
    state0_ = copy(state0)
    update_circuit_param!(c_, theta)
    _create_quantum_state(c_, state0_)
end



"""
Calculate green function based on imaginary-time evolution.

ham_op:
    Hamiltonian
c_op:
    annihilation operator 
cdagg_op:
    creation operator 
vc:
    Variational circuit. The current value of variational parameters are
    used as the initial value of the imaginary-time evolution.
state_gs:
    The state is the ground state of the hamiltonian
state0_ex:
    The excited state to which the creation operator  is applied to
taus:
    list of imaginary times in ascending order
    The first element must be 0.0. The last element must be beta.
return:
    The list of Green function at each tau 
"""

function compute_gtau(
    ham_op::OFQubitOperator,
    left_op::OFQubitOperator,
    right_op::OFQubitOperator,
    vc_ex::VariationalQuantumCircuit,
    state_gs::QulacsQuantumState,　
    state0_ex::QulacsQuantumState,
    taus::Vector{Float64}, delta_theta=1e-8;
    comm=MPI_COMM_WORLD
    )
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end

    if taus[1] != 0.0
        error("The first element of taus must be 0!")
    end

    if !all(taus[2:end] .> taus[1:end-1])
       error("taus must in strictly asecnding order!")
    end

    circuit_right_ex = copy(vc_ex) 
    right_squared_norm = apply_qubit_op!(right_op, state_gs, circuit_right_ex, state0_ex)
    state_right_ex = copy(state0_ex)
    update_quantum_state!(circuit_right_ex, state_right_ex)

    # exp(-tau H)c^{dag}_j|g.s>
    thetas_tau_right = imag_time_evolve(ham_op, circuit_right_ex, state0_ex, taus, delta_theta, comm=comm)[1]
    log_norm_tau_right = imag_time_evolve(ham_op, circuit_right_ex, state0_ex, taus, delta_theta, comm=comm)[2]
    
    Gfunc_ij_list = Complex{Float64}[]
    E_gs = get_expectation_value(ham_op, state_gs)

    for t in eachindex(taus)
        state_right = _create_quantum_state(vc_ex, thetas_tau_right[t], state0_ex)
        state_left = copy(state_gs)
        # Divide the qubit operator of c_i into its real and imaginary parts.
        op_re, op_im = divide_real_imag(left_op)
        g_re = get_transition_amplitude(op_re, state_left, state_right)
        g_im = get_transition_amplitude(op_im, state_left, state_right)
        push!(Gfunc_ij_list, -(g_re + im * g_im) * right_squared_norm * exp(log_norm_tau_right[t] + E_gs *  taus[t]))
    end
    Gfunc_ij_list
end

