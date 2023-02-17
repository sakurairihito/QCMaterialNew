using Test
using LinearAlgebra
using QCMaterial

import PyCall: pyimport
import Random

@testset "vqe.solve_gs" begin
    Random.seed!(100)
    nsite = 2
    n_qubit = 2 * nsite
    U = 1.0
    t = -0.01
    μ = 0

    ham = FermionOperator()
    for i = 1:nsite
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^ $(down)^ $(up) $(down)", -U)
    end

    for i = 1:nsite-1
        ham += FermionOperator("$(up_index(i+1))^ $(up_index(i))", t)
        ham += FermionOperator("$(up_index(i))^ $(up_index(i+1))", t)
        ham += FermionOperator("$(down_index(i+1))^ $(down_index(i))", t)
        ham += FermionOperator("$(down_index(i))^ $(down_index(i+1))", t)
    end

    for i = 1:nsite
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^  $(up) ", -μ)
        ham += FermionOperator("$(down)^ $(down)", -μ)
    end

    # Compute exact ground-state energy
    n_electron = 2
    @assert mod(n_electron, 2) == 0
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron)
    EigVal_min = minimum(eigvals(sparse_mat.toarray()))

    # Ansatz
    state0 = create_hf_state(n_qubit, n_electron)
    circuit = uccgsd(n_qubit, orbital_rot = true)
    theta_init = rand(num_theta(circuit))

    # VQE
    cost_history, thetas_opt = QCMaterial.solve_gs(
        jordan_wigner(ham),
        circuit,
        state0,
        theta_init = theta_init,
        verbose = true,
        comm = QCMaterial.MPI_COMM_WORLD,
    )
    @show abs(EigVal_min - cost_history[end])
    @test abs(EigVal_min - cost_history[end]) < 1e-6
end


@testset "get_expected_value_sampling" begin
    U = 4.0
    V = 1.0
    μ = 1.0
    ε = [1.0, 0.0]
    nsite = 2
    n_qubit = nsite * 2
    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite)
    #@show ham 
    n_electron = 2
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron) #行列の取得 
    using LinearAlgebra
    enes_ed = eigvals(sparse_mat.toarray()) #対角化を行う
    EigVal_min = minimum(enes_ed)
    #@show EigVal_min 
    ham_q = jordan_wigner(ham)
    nshots = 2^17
    n_electron_gs = 2
    n_qubit = 4
    state2 = create_hf_state(n_qubit, n_electron_gs)
    depth = 4
    circuit2 = hev(n_qubit, depth)
    Random.seed!(90)
    theta_init = rand(num_theta(circuit2))
    update_circuit_param!(circuit2, theta_init)
    update_quantum_state!(circuit2, state2)
    res2 = get_expected_value_sampling(ham_q, state2, nshots=nshots)
    exact2 = get_expectation_value(ham_q, state2)
    @show abs(res2 - exact2)
    #@test res ≈ exact 
    isapprox(res2, exact2, atol = 1e-2)
    isapprox(res2, exact2, rtol = 1e-2)
    @test abs(res2 - exact2) < 1e-2
end

@testset "solve_gs_sampling" begin
    U = 4.0
    V = 1.0
    μ = 1.0
    ε = [1.0, 0.0]
    nsite = 2
    n_qubit = nsite * 2
    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite)
    #@show ham 
    n_electron = 2
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron) #行列の取得 
    using LinearAlgebra
    enes_ed = eigvals(sparse_mat.toarray()) #対角化を行う
    EigVal_min = minimum(enes_ed)
    #@show EigVal_min 
    ham_q = jordan_wigner(ham)
    #@show ham_q 
    # values

    n_qubit = 4
    n_electron_gs = 2
    state2 = create_hf_state(n_qubit, n_electron_gs)
    depth = 4
    circuit2 = uccgsd(n_qubit)
    Random.seed!(90)
    theta_init = rand(num_theta(circuit2))
    update_circuit_param!(circuit2, theta_init)
    update_quantum_state!(circuit2, state2)

    cost_history, _ = solve_gs_sampling(ham_q, circuit2, state2, nshots=2^16)
    @show cost_history
    @show cost_history[end]
    @show EigVal_min
    @show abs(EigVal_min - cost_history[end])

    @test abs(EigVal_min - cost_history[end]) < 1e-2
end


@testset "solve_gs_sampling2" begin
    U = 0.0
    V = 0.0
    μ = 1.0
    ε = [1.0, 0.0]
    nsite = 2
    n_qubit = nsite * 2
    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite)
    #@show ham 
    n_electron = 2
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron) #行列の取得 
    using LinearAlgebra
    enes_ed = eigvals(sparse_mat.toarray()) #対角化を行う
    EigVal_min = minimum(enes_ed)
    #@show EigVal_min 
    ham_q = jordan_wigner(ham)
    #@show ham_q 
    # values

    n_qubit = 4
    n_electron_gs = 2
    state2 = create_hf_state(n_qubit, n_electron_gs)
    depth = 4
    circuit2 = hev(n_qubit, depth)
    Random.seed!(90)
    theta_init = rand(num_theta(circuit2))
    update_circuit_param!(circuit2, theta_init)
    update_quantum_state!(circuit2, state2)

    cost_history, _ = solve_gs_sampling(ham_q, circuit2, state2, nshots=2^16)
    @show cost_history
    @show cost_history[end]
    @show EigVal_min
    @show abs(EigVal_min - cost_history[end])

    @test abs(EigVal_min - cost_history[end]) < 1e-1
end
