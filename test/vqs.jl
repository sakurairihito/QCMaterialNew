using Test
using LinearAlgebra
using QCMaterial

@testset "vqs.A" begin
    n_qubit = 2
    n_electron = 1

    c = UCCQuantumCircuit(n_qubit)
    # a_0^dagger a_1 - a^1 a_0 -> 0.5i (X0 Y1 - X1 Y0)
    generator = gen_t1(0, 1)
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

    ham = OFQubitOperator("X0 X1", 1.0)

    c = UCCQuantumCircuit(n_qubit)
    # a_0^dagger a_1 - a^1 a_0 -> 0.5i (X0 Y1 - X1 Y0)
    generator = gen_t1(0, 1)
    add_parametric_circuit_using_generator!(c, generator, 0.0)

    state0 = QulacsQuantumState(n_qubit, 0b01)

    new_thetas = copy(c.thetas)

    new_thetas .+= 1e-8
    c_new = copy(c)
    update_circuit_param!(c_new, new_thetas)
    C = compute_C(ham, c, state0, 1e-2)
    @test C ≈ [0.5] atol=1e-5
end