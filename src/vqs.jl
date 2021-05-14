export compute_A
export compute_C

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