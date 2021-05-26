using Test
using LinearAlgebra
using QCMaterial

@testset "vqs.A" begin
    n_qubit = 2
    n_electron = 1

    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 0.0)

    state0 = QulacsQuantumState(n_qubit, 0b01)

    new_thetas = copy(c.thetas)

    new_thetas .+= 1e-8
    c_new = copy(c)
    update_circuit_param!(c_new, new_thetas)
    A = compute_A(c, state0, 1e-2)
    @test A ≈ [0.25] atol=1e-5
end

@testset "vqs.C" begin
    n_qubit = 2
    n_electron = 1

    ham = OFQubitOperator("X1 X2", 1.0)

    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 0.0)

    state0 = QulacsQuantumState(n_qubit, 0b01)

    C = compute_C(ham, c, state0, 1e-2)
    @test C ≈ [0.5] atol=1e-5
end

@testset "vqs.thetadot" begin
    n_qubit = 2
    n_electron = 1
    ham = OFQubitOperator("X1 X2", 1.0)
    c = UCCQuantumCircuit(n_qubit)
    
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 0.0)

    state0 = QulacsQuantumState(n_qubit,0b01)
    thetadot = compute_thetadot(ham, c, state0, 1e-2) 
    @test thetadot ≈ [2] atol=1e-5
end

@testset "vqs.imag_time_evolve_single_qubit" begin
    """
    Hamiltonian H = Z1
    No time evolution occurs.
    exp(- tau * H) |1> = A |1>,
    where A is a normalization constant.
    """
    n_qubit = 1
    d_theta = 0.01

    ham = OFQubitOperator("Z1", 1.0)
    state0 = QulacsQuantumState(n_qubit, 0b1)

    c = QulacsParametricQuantumCircuit(n_qubit)
    add_parametric_RY_gate!(c, 1, 0.0)
    vc = QulacsVariationalQuantumCircuit(c)

    taus = [0.0, 1.0]
    thetas_tau = imag_time_evolve(ham, vc, state0, taus, d_theta)

    for i in eachindex(taus)
        @test all(thetas_tau[i] .≈ [0.0])
    end
end

@testset "vqs.imag_time_evolve_single_qubit2" begin
    """
    Hamiltonian H = Z1
    No time evolution occurs.
    exp(- tau * H) (|0> + |1>)/sqrt(2) = R_Y(theta(tau)) |0>,
    where A is a normalization constant and
    theta(tau) = acos(exp(-tau)/sqrt(exp(-2tau) + exp(2*tau))).
    theta = pi/2 (tau=0) => pi (tau=+infty)
    """
    n_qubit = 1
    d_theta = 0.01

    ham = OFQubitOperator("Z1", 1.0)
    state0 = QulacsQuantumState(n_qubit, 0b0)

    c = QulacsParametricQuantumCircuit(n_qubit)
    add_parametric_RY_gate!(c, 1, 0.5*pi)
    vc = QulacsVariationalQuantumCircuit(c)

    taus = collect(range(0.0, 1, length=100))
    thetas_tau = imag_time_evolve(ham, vc, state0, taus, d_theta)

    theta_extact(τ) = 2*acos(exp(-τ)/sqrt(exp(-2*τ) + exp(2*τ)))

    thetas_ref = theta_extact.(taus)
    thetas_res = [thetas_tau[i][1] for i in eachindex(taus)]
    @test isapprox(thetas_ref, thetas_res, rtol=0.01)
end


@testset "vqs.imag_time_evolve_double_qubit" begin
    """
    1site hubbard under magnetic field
    Hamiltonian H = Un_(1up)n_(2dn) - hn_(1up) + hn_(2dn) 
    No time evolution occurs.
    exp(- tau * H) (|1up> + |1dn>)/sqrt(2) = exp(theta/2*(c_1up^(dag)c_1dn-c_1dn^(dag)c_1up)) |1up>,
    Coefficient of |1up> is exp(h*tau)/sqrt(exp(2*h*tau) + exp(-2*h*tau))).
    Coefficient of |1dn> is exp(-h*tau)/sqrt(exp(2*h*tau) + exp(-2*h*tau))).

    we check the coefficients of |1up> and |1up> instead of theta.
    """
    n_qubit = 2
    d_theta = 0.01
    U = 1
    h = 1

    ham = FermionOperator()
    up = up_index(1)
    down = down_index(1)
    ham += FermionOperator("$(up)^ $(down)^ $(up) $(down)", -U)
    ham += FermionOperator("$(up)^  $(up) ", -h)
    ham += FermionOperator("$(down)^ $(down)", h)
    ham = jordan_wigner(ham)

    # Prepare |Psi> = |01>  (# Apply c^dagger_1up to |00>, which will yield |01>.)
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b01)

    state_1up = QulacsQuantumState(n_qubit)
    set_computational_basis!(state_1up, 0b01)

    state_1dn = QulacsQuantumState(n_qubit)
    set_computational_basis!(state_1dn, 0b10)

    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2^dagger a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 0.5*pi)

    #update_quantum_state!(c, state)??

    taus = collect(range(0.0, 1, length=1000))
    println("taus", taus)
    thetas_tau = imag_time_evolve(ham, c, state, taus, d_theta)

    coeff_1up(τ) = exp(h*τ)/sqrt(exp(2*h*τ) + exp(-2*h*τ))
    coeff_1up(τ) = exp(-h*τ)/sqrt(exp(2*h*τ) + exp(-2*h*τ))

    coeff_1up_res = Complex{Float64}[]
    for t in eachindex(taus)
    #for t in [1, 2, 3]
        update_circuit_param!(c, thetas_tau[t])
        state = QulacsQuantumState(n_qubit)
        set_computational_basis!(state, 0b01)
        update_quantum_state!(c, state)
        println("debug", thetas_tau[t])
        push!(coeff_1up_res, inner_product(state_1up, state))
    end
    @test isapprox(coeff_1up.(taus), coeff_1up_res, rtol=0.01)



end


