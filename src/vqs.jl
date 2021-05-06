#A_ijの計算
"""
Compute <phi (theta_bra) | phi(theta_ket)>
"""
function overlap(ucccirc::UCCQuantumCircuit, state0::QulacsQuantumState, theta_bra, theta_ket, delta_theta)
    circ_tmp = copy(ucccirc)
    # Do something
end

function compute_Aij(ucccirc::UCCQuantumCircuit, state0::QulacsQuantumState, delta_theta=1e-8)
    num_thetas = num_theta(ucccirc)
    theta = ucccirc.theta

    A = zeros(Complex{Float64}, num_thetas, num_thetas)
    for j in 1:num_thetas
        theta_j = copy(ucc.theta)
        theta_j[j] += delta_theta
        for i in 1:num_thetas
            theta_i = copy(ucc.theta)
            theta_i[i] += delta_theta
            A[i, j] = real(
                      overlap(ucccirc, state0, theta_i, theta_j, delta_theta)
                    - overlap(ucccirc, state0, theta_i, theta,   delta_theta)
                    - overlap(ucccirc, state0, theta,   theta_j, delta_theta)
                    + overlap(ucccirc, state0, theta,   theta,   delta_theta)
                )/delta_theta^2
        end
    end

end