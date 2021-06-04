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

    """

    n_qubit = 2
    d_theta = 0.01
    
    U = 1.0
    h = 1.0

    ham = FermionOperator()
    up = up_index(1)
    down = down_index(1)
    ham += FermionOperator("$(up)^ $(down)^ $(up) $(down)", -U)
    ham += FermionOperator("$(up)^  $(up) ", -h)
    ham += FermionOperator("$(down)^ $(down)", h)
    ham = jordan_wigner(ham)

    state0 = QulacsQuantumState(n_qubit, 0b01)

    c = QulacsParametricQuantumCircuit(n_qubit)
    # add_parametric_RYY_gate! -> add_parametric_multi_Pauli_rotation_gate!
    target = [1,2] 
    #X->pauli_X, Y->pauli_Y, Z->pauli_Z
    pauli_ids = [pauli_Y, pauli_Y] 
    add_parametric_multi_Pauli_rotation_gate!(c, target, pauli_ids, 0.5*pi)
    #QulacsVariationalQuantumCircuitに対してadd_S_gate!を定義する。
    #QulacsVariationalQuantumCircuitとQulacsQuantumCircuitは両方ともQuantumCircuitを親に持つが、兄弟間同士で継承(多重継承)できないことに注意する。
    add_S_gate!(c, 2)
    vc = QulacsVariationalQuantumCircuit(c)

    taus = collect(range(0.0, 1, length=100))
    thetas_tau = imag_time_evolve(ham, vc, state0, taus, d_theta)

    theta_extact(τ) = 2*acos(exp(h*τ)/sqrt(exp(2*h*τ) + exp(-2*h*τ)))

    #generate vector whose elements are theta_exact(τ) 
    thetas_ref = theta_extact.(taus)
    #i is tau and num of parameter is 1
    thetas_res = [thetas_tau[i][1] for i in eachindex(taus)]
    @test isapprox(thetas_ref, thetas_res, rtol=0.01)

end


@testset "vqs.compute_gtau" begin
    """
    """
    n_qubit　= 2
    nsite = 1
    t = 1.0
    U = 1.0
    µ = 1.0
    up = up_index(1)
    down = down_index(1)
    d_theta = 0.01

    ham_op = generate_ham_1d_hubbard(t, U, nsite, μ)
    ham_op = jordan_wigner(ham_op)
    
    c_op = FermionOperator("$(up) ", 1.0)
    c_op = jordan_wigner(c_op)
    cdagg_op = FermionOperator("$(up)^ ", 1.0)
    cdagg_op = jordan_wigner(cdagg_op)

    c_ = QulacsParametricQuantumCircuit(n_qubit)
    target = [1,2] 
    pauli_ids = [pauli_Y, pauli_Y] 
    add_parametric_multi_Pauli_rotation_gate!(c_, target, pauli_ids, 0.5*pi)
    add_S_gate!(c_, 2)
    vc = QulacsVariationalQuantumCircuit(c_)



    #c = UCCQuantumCircuit(n_qubit)
    #a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    #generator = gen_t1(1, 2)
    #add_parametric_circuit_using_generator!(c, generator, 0.0)

    #c = UCCQuantumCircuit(c_)

    state0_gs = QulacsQuantumState(n_qubit)
    #Perform VQE to compute ground state
    
    #function cost(thetas)
    #    update_circuit_param!(vc, thetas)
    #    get_expectation_value(ham_op, state0_gs) 
    #end
    state0_ex = QulacsQuantumState(n_qubit)
    
    taus = collect(range(0.0, 1.0, length=10))
    #println(taus)
    Gfunc_ij_list = compute_gtau(ham_op, c_op, cdagg_op, vc,  state0_gs, state0_ex, taus, d_theta)

end
