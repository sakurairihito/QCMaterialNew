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

@testset "vqs.imag_time_evolve" begin
    """
    Hamiltonian = Z1
    No time evolution occurs.
    """
    n_qubit = 1
    tau_max = 1.0
    ntau = 2
    d_theta = 0.01

    ham = OFQubitOperator("Z1", 1.0)
    state0 = QulacsQuantumState(n_qubit, 0b1)

    c = QulacsParametricQuantumCircuit(n_qubit)
    add_parametric_RY_gate!(c, 1, 0.0)

    taus = collect(range(0.0, tau_max, length=ntau))

    vc = QulacsVariationalQuantumCircuit(c)
    thetas_tau = imag_time_evolve(ham, vc, state0, taus, d_theta)

    for i in 1:ntau
        @test all(thetas_tau[i] .≈ [0.0])
    end
end