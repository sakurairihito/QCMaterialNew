export compute_A
export compute_C
export compute_thetadot

#Aの計算
"""
Compute <phi (theta_bra) | phi(theta_ket)>
"""
function overlap(ucccirc::UCCQuantumCircuit, state0::QulacsQuantumState,
    thetas_left::Vector{Float64}, thetas_right::Vector{Float64})

    circ_tmp = copy(ucccirc)

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

function compute_A(ucccirc::UCCQuantumCircuit, state0::QulacsQuantumState, delta_theta=1e-8)
    num_thetas = num_theta(ucccirc)
    thetas = ucccirc.thetas

    A = zeros(Complex{Float64}, num_thetas, num_thetas)
    for j in 1:num_thetas
        thetas_j = copy(ucccirc.thetas)
        thetas_j[j] += delta_theta
        for i in 1:num_thetas
            thetas_i = copy(ucccirc.thetas)
            thetas_i[i] += delta_theta
            A[i, j] = real(
                      overlap(ucccirc, state0, thetas_i, thetas_j)
                    - overlap(ucccirc, state0, thetas_i, thetas, )
                    - overlap(ucccirc, state0, thetas,   thetas_j)
                    + overlap(ucccirc, state0, thetas,   thetas, )
                )/delta_theta^2
        end
    end
    A
end

#Cの計算
"""
Compute <phi (theta_bra) |H| phi(theta_ket)>
"""
function transition(op::OFQubitOperator, ucccirc::UCCQuantumCircuit, state0::QulacsQuantumState,
    thetas_left::Vector{Float64}, thetas_right::Vector{Float64})

    circ_tmp = copy(ucccirc)

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


function compute_C(op::OFQubitOperator, ucccirc::UCCQuantumCircuit,state0::QulacsQuantumState, delta_theta=1e-8)
    num_thetas = num_theta(ucccirc)
    thetas = ucccirc.thetas

    C = zeros(Complex{Float64}, num_thetas)
    for i in 1:num_thetas
        thetas_i = copy(ucccirc.thetas)
        thetas_i[i] += delta_theta
        C[i] = -real(
            transition(op, ucccirc, state0, thetas, thetas_i)
            -transition(op, ucccirc, state0, thetas, thetas)
        )/delta_theta
    end
    C
end



#theta(tau)の微分の計算
"""
Compute thetadot = A^(-1) C
"""

function compute_thetadot(op::OFQubitOperator, ucccirc::UCCQuantumCircuit,state0::QulacsQuantumState,delta_theta=1e-8)
    #compute A
    A = compute_A(ucccirc, state0, delta_theta)
    #compute inverse of A
    InvA = inv(A)
    #compute C
    C = compute_C(op, ucccirc, state0, delta_theta)
    #compute AC
    thetadot = InvA * C
    #return thetadot
    thetadot
end


"""
Perform imaginary-time evolution.

taus:
    list of imaginary times in ascending order
    The first element must be 0.0. 
return:
    list of variational parameters at the given imaginary times.
"""
function imag_time_evolve(ham_op::OFQubitOperator, ucccirc::UCCQuantumCircuit, state0::QulacsQuantumState,
    taus::Vector{Float64}, delta_theta=1e-8)::Vector{Vector{Float64}}
    # Implement!
    #for i in 1:taus[end]
    #for i in 1:length(taus)
    #for i in eachindex(taus)
    for (i, tau) in enumerate(taus)
        theta[i] = compute_thetadot(ham_op,ucccirc,state0,delta_theta)
        taus += 1.0
    end
end


  
    #tauの範囲[0,beta]
    #次にメッシュを区切る.メッシュ点の番号[1,N].区間の数=N-1. delta tau = beta/(N-1).
    #各メッシュ点での初期パラメータを与える。
    #パラメータのtau微分を計算する。
    #パラメータのt
