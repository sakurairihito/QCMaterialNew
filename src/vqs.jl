export compute_A
export compute_C
export compute_thetadot

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
    taus::Vector{Float64}, delta_theta=1e-8)::Vector{Vector{Float64}}
    if taus[1] != 0.0
        error("The first element of taus must be 0!")
    end
    thetas_tau = [copy(get_thetas(vc))]
    for i in 1:length(taus)-1
        vc_ = copy(vc)
        update_circuit_param!(vc_, thetas_tau[i])
        thetas_dot_ = compute_thetadot(ham_op, vc_, state0, delta_theta)
        if taus[i+1] <= taus[i]
            error("taus must be in strictly asecnding order!")
        end
        thetas_next_ = thetas_tau[i] + (taus[i+1] - taus[i]) * thetas_dot_
        push!(thetas_tau, thetas_next_)
    end
    thetas_tau
end


function _create_quantum_state(c, state0)
    state0_ = copy(state0)
    update_quantum_state!(c, state0_)
    state0_
end




"""
Calculate green function based on imaginary-time evolution.

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
    G[i,j]
"""

function imag_time_evo_green_func(
    ham_op::OFQubitOperator, op::Vector{OFQubitOperator},
    vc::VariationalQuantumCircuit,
    state0_gs::QulacsQuantumState,
    state0_ex::QulacsQuantumState,
    taus::Vector{Float64}, beta::Float64, delta_theta=1e-8)

    n_qubit = get_n_qubit(state0)
    thetas = get_thetas(vc)
    beta_taus = copy(taus)
    beta_taus = beta_taus .- beta
    d_theta = 0.01

    #prepare the ground state|g.s>
    state_right = copy(state0_gs)
    update_quantum_state!(vc, state_right)   
    #state_right = _create_quantum_state(vc, state0_gs)
    
    #function apply_qubit_opに!を付けて書き換える。
    # state_right_ex: A c^{dag}_j|g.s> (A is a normalization factor)
    state_right_ex = copy(state0_ex)
    #opt_thetasでcircuitが変わるので、新しくcircuit_exを定義する
    circuit_right_ex = copy(vc)
     _ = apply_qubit_op(op[j], state_right, circuit_right_ex, state0_ex)
    update_quantum_state!(circuit_right_ex, state_right_ex)
    #state_right_ex = _create_quantum_state(circuit_right_ex, state0_ex)と書ける

    #exp(-tau H)c^{dag}_j|g.s>
    thetas_tau = imag_time_evolve(ham_op, vc, state_right_ex, taus, d_theta)
    state_right_list = []
    for t in eachindex(taus)
        c_ = copy(vc)
        update_circuit_param!(c_, thetas_tau[t])
        push!(state_right_list, _create_quantum_state(c_, state0_ex))
    end

    #left side
    #prepare the ground state|g.s>
    state_left = copy(state0_gs)
    update_quantum_state!(vc, state_left) 
    #state_left = _create_quantum_state(vc, state0_gs)

    #exp(-(beta-tau) H)|g.s>
    thetas_tau = imag_time_evolve(ham_op, vc, state_left, beta_taus, d_theta)
    state_left_list = []
    for t in eachindex(taus)
        c__ = copy(vc)
        update_circuit_param!(c__, thetas_tau[t])
        state_left_fin = create_quantum_state(c__, state0_gs)
        #B c^{dag}_i exp(-(beta-tau) H)|g.s>(B is a normalization factor)
        # exp(-(beta-tau) H)|g.s>の各虚時間tauごとのリストに対して、c^{dag}_iをかける。
        #opt_thetasでcircuitが変わるので、新しくcircuit_exを定義する
        circuit_left_ex = copy(vc)
        state_left_ex = copy(state0_ex)
        _ = apply_qubit_op(op[i], state_left_fin, circuit_left_ex, state_left_ex)
        update_quantum_state!(circuit_left_ex, state_left_ex)
        #state_left_ex = _create_quantum_state(circuit_left_e, state0_ex)と書ける
        push!(state_left_list, state_left_ex) 
    end
    

    #Compute G[i, j](taus)
    #各虚時間tauで定義されているパラメータを含んだ状態<state(tau)| state(tau)>の内積のリストを作る？
    #G[i, j]のi,jはそれぞれ、op[i],op[j]の添字
    Gfunc_ij_list = []
    for t in eachindex(taus)
        push!(Gfunc_ij_list, inner_product(state_left_list[t],state_right_list[t] ))
    end
    Gfunc_ij_list
end
