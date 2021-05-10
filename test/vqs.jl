using Test
using LinearAlgebra
using QCMaterial

@testset "vqs.Aij" begin
    n_qubit = 2
    n_electron = 1

    c = UCCQuantumCircuit(n_qubit)
    # a_0^dagger a_1 - a^1 a_0 -> i (X0 Y1 - X1 Y0)
    generator = gen_t1(0, 1)
    add_parametric_circuit_using_generator!(c, generator, 0.0)

    state0 = QulacsQuantumState(n_qubit, 0b01)

    #new_thetas = copy(c.thetas)
    #new_thetas .+= 0.1
    #c_new = copy(c)
    #update_circuit_param!(c_new, new_thetas)
    #println("debug", QCMaterial.overlap(c_new, state0, new_thetas, c.thetas))

    A = compute_Aij(c, state0, 1e-2)
    println(A)
end