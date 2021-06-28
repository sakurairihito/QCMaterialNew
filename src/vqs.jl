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

function compute_A(vc::VariationalQuantumCircuit, state0::QulacsQuantumState, delta_theta=1e-8)
    num_thetas = num_theta(vc)
    thetas = get_thetas(vc)

    A = zeros(Complex{Float64}, num_thetas, num_thetas)
    for j in 1:num_thetas
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
    A
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
    state0::QulacsQuantumState,delta_theta=1e-8)
    #compute A
    A = compute_A(vc, state0, delta_theta)
    #compute inverse of A
    InvA = inv(A)
    #compute C
    C = compute_C(op, vc, state0, delta_theta)
    #compute AC
    thetadot = InvA * C
    #return thetadot
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
    taus::Vector{Float64}, delta_theta=1e-8)::Tuple{Vector{Vector{Float64}}, Vector{Float64}}
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
        thetas_dot_ = compute_thetadot(ham_op, vc_, state0, delta_theta)
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






#ノルムを考慮した場合
function compute_gtau_before(
    ham_op::OFQubitOperator,
    c_op::OFQubitOperator,
    cdagg_op::OFQubitOperator,
    vc::VariationalQuantumCircuit, 
    state_gs::QulacsQuantumState,　#VQEを実行した後の基底状態
    state0_ex::QulacsQuantumState,
    taus::Vector{Float64}, delta_theta=1e-8)

    if taus[1] != 0.0
        error("The first element of taus must be 0!")
    end

    if !all(taus[2:end] .> taus[1:end-1])
       error("taus must in strictly asecnding order!")
    end

    # Inverse temperature
    beta = taus[end]

    # state_right_ex: A c^{dag}_j|g.s> (A is a normalization factor)
    # TODO: function apply_qubit_opに!を付けて書き換える。

    ##state_right = copy(state_gs)ではないか？
    state_right = _create_quantum_state(vc, state_gs)
    circuit_right_ex = copy(vc) #opt_thetasでcircuitが変わるので、新しくcircuit_exを定義する.

    #!TODO:名前変える。
    right_squared_norm = apply_qubit_op!(cdagg_op, state_right, circuit_right_ex, state0_ex)
    state_right_ex = copy(state0_ex)
    update_quantum_state!(circuit_right_ex, state_right_ex)

    # exp(-tau H)c^{dag}_j|g.s>
    thetas_tau_right = imag_time_evolve(ham_op, circuit_right_ex, state0_ex, taus, delta_theta)[1]
    log_norm_tau_right = imag_time_evolve(ham_op, circuit_right_ex, state0_ex, taus, delta_theta)[2]


    # Compute exp(-(beta-tau) H)|g.s> on the tau mesh from the left
    beta_taus = reverse(beta .- taus)
    
    #state_left = copy(state_gs)?
    #state_left = _create_quantum_state(vc, state_gs)
    # FIXME: THIS IS TRIVIAL
    #thetas_tau_left = imag_time_evolve(ham_op, vc, state_gs, beta_taus, delta_theta)[1]
    log_norm_tau_left = imag_time_evolve(ham_op, vc, state_gs, beta_taus, delta_theta)[2]

    Gfunc_ij_list = Complex{Float64}[]
    ntaus = length(taus)
    E_gs = get_expectation_value(ham_op, state_gs)



    for t in eachindex(taus)
        # exp(-tau H)c^{dag}_j|g.s>
        state_right = _create_quantum_state(vc, thetas_tau_right[t], state0_ex)
        # circicut for exp(-(beta-tau) H) |g.s>
        #vc_left = copy(vc)
        #update_circuit_param!(vc_left, thetas_tau_left[ntaus-t+1])
        state_gs_debug = copy(state_gs)
        #update_quantum_state!(vc_left, state_gs_debug)
        #println("compute_norm_left_ref=", - E_gs * (beta - t))
        #println("compute_norm_left=", log_norm_tau_left[t])
        # Divide the qubit operator of c_i into its real and imaginary parts.
        op_re, op_im = divide_real_imag(c_op)
        g_re = get_transition_amplitude(op_re, state_gs_debug, state_right)
        g_im = get_transition_amplitude(op_im, state_gs_debug, state_right)
        #g_re = get_transition_amplitude_with_obs(vc_left, state_gs, op_re, state_right)
        #g_im = get_transition_amplitude_with_obs(vc_left, state_gs, op_im, state_right)
        push!(Gfunc_ij_list, -(g_re + im * g_im) * right_squared_norm * exp(log_norm_tau_right[t] +　log_norm_tau_left[ntaus-t+1]　+ beta * E_gs))
        #push!(Gfunc_ij_list, -(g_re + im * g_im) * right_squared_norm * exp(log_norm_tau_right[t] - E_gs * (beta - t)　+ beta * E_gs))
    end

    Gfunc_ij_list
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
state0_gs:
    The initial state is the ground state of the hamiltonian
state0_ex:
    The excited state to which the creation operator  is applied to
taus:
    list of imaginary times in ascending order
    The first element must be 0.0. The last element must be beta.
return:
    The list of Green function at each tau 
"""

function compute_gtau_norm(
    ham_op::OFQubitOperator,
    c_op::OFQubitOperator,
    cdagg_op::OFQubitOperator,
    vc_ex::VariationalQuantumCircuit,
    state_gs::QulacsQuantumState,　
    state0_ex::QulacsQuantumState,
    taus::Vector{Float64}, delta_theta=1e-8)

    if taus[1] != 0.0
        error("The first element of taus must be 0!")
    end

    if !all(taus[2:end] .> taus[1:end-1])
       error("taus must in strictly asecnding order!")
    end

    # Inverse temperature
    beta = taus[end]

    circuit_right_ex = copy(vc_ex) 
    right_squared_norm = apply_qubit_op!(cdagg_op, state_gs, circuit_right_ex, state0_ex)
    state_right_ex = copy(state0_ex)
    update_quantum_state!(circuit_right_ex, state_right_ex)

    #debug
    println("right_squared_norm=", right_squared_norm)
    #debug
    println("state_right_ex=", get_vector(state_right_ex))
    
    #debug2
    norm_cdag_gs = inner_product(state_right_ex, state_right_ex)
    println(" norm_c^dag_gs=",  norm_cdag_gs) 

    # exp(-tau H)c^{dag}_j|g.s>
    thetas_tau_right = imag_time_evolve(ham_op, circuit_right_ex, state0_ex, taus, delta_theta)[1]
    log_norm_tau_right = imag_time_evolve(ham_op, circuit_right_ex, state0_ex, taus, delta_theta)[2]
    println("log_norm_tau_right=", log_norm_tau_right)
    
    Gfunc_ij_list = Complex{Float64}[]
    ntaus = length(taus)
    E_gs = get_expectation_value(ham_op, state_gs)

    beta_taus = reverse(beta .- taus)
    #log_norm_tau_left = imag_time_evolve(ham_op, vc_ex, state_gs, beta_taus, delta_theta)[2]
    

    for t in eachindex(taus)
        state_right = _create_quantum_state(vc_ex, thetas_tau_right[t], state0_ex)
        state_left = copy(state_gs)
        # Divide the qubit operator of c_i into its real and imaginary parts.
        op_re, op_im = divide_real_imag(c_op)
        g_re = get_transition_amplitude(op_re, state_left, state_right)
        g_im = get_transition_amplitude(op_im, state_left, state_right)
        println("log_norm_tau_right[t] =", log_norm_tau_right[t] )
        println("E_gs *  t =",E_gs *  taus[t] )
        #push!(Gfunc_ij_list, -(g_re + im * g_im) * right_squared_norm * exp(log_norm_tau_right[t] + log_norm_tau_left[ntaus-t+1] + beta * E_gs))
        push!(Gfunc_ij_list, -(g_re + im * g_im) * right_squared_norm * exp(log_norm_tau_right[t] + E_gs *  taus[t]))
    end
    Gfunc_ij_list
end




