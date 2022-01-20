using Test
using LinearAlgebra
using QCMaterial

import PyCall: pyimport
import Random

@testset "ucc.uccgsd" begin
    #Random.seed!(1)
    scipy_opt = pyimport("scipy.optimize")

    nsite = 2 
    n_qubit = 2*nsite 
    U = 1.0
    t = -0.01
    μ = 0

    ham = FermionOperator()
    for i in 1:nsite
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^ $(down)^ $(up) $(down)", -U) 
    end

    for i in 1:nsite-1
        ham += FermionOperator("$(up_index(i+1))^ $(up_index(i))", t) 
        ham += FermionOperator("$(up_index(i))^ $(up_index(i+1))", t) 
        ham += FermionOperator("$(down_index(i+1))^ $(down_index(i))", t) 
        ham += FermionOperator("$(down_index(i))^ $(down_index(i+1))", t) 
    end

    for i in 1:nsite
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^  $(up) ", -μ) 
        ham += FermionOperator("$(down)^ $(down)", -μ)
    end

    n_electron = 2　
    @assert mod(n_electron, 2) == 0
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron);　

    enes_ed = eigvals(sparse_mat.toarray());　

    ham_jw = jordan_wigner(ham)

    circuit = uccgsd(n_qubit, orbital_rot=true)

    function cost(theta_list)
        state = create_hf_state(n_qubit, n_electron)
        update_circuit_param!(circuit, theta_list) 
        update_quantum_state!(circuit, state) 
        get_expectation_value(ham_jw, state) 
    end

    theta_init = rand(num_theta(circuit))
    cost_history = Float64[] 
    init_theta_list = theta_init
    push!(cost_history, cost(init_theta_list))
    

    method = "BFGS"
    options = Dict("disp" => true, "maxiter" => 200, "gtol" => 1e-5)
    callback(x) = push!(cost_history, cost(x))
    opt = scipy_opt.minimize(cost, init_theta_list, method=method, callback=callback)
    println("Eigval_vqe=", cost_history[end])
    EigVal_min = minimum(enes_ed)
    println("EigVal_min=", EigVal_min)
    @test abs(EigVal_min-cost_history[end]) < 1e-6 
end


@testset "ucc.UCCQuantumCircuit" begin
    n_qubit = 4
    c = uccgsd(n_qubit, orbital_rot=true)
    c.thetas .= 1.0
    c_copy = copy(c)
    @test all(c_copy.thetas == c.thetas)
end

@testset "ucc.one_rotation_gate" begin
    using LinearAlgebra
    n_qubit = 1
    theta = 1e-5

    c = UCCQuantumCircuit(n_qubit)
    add_parametric_multi_Pauli_rotation_gate!(
            c.circuit, [1], [pauli_Y], theta)

    state = QulacsQuantumState(n_qubit, 0b1)
    update_quantum_state!(c, state)
    vec = get_vector(state)
    vec ≈ [0.5*theta, 1.0]
end