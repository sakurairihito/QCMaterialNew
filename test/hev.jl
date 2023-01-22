using Test
using QCMaterial
using Random
using LinearAlgebra

@testset "create_H_gate" begin
    n_qubit = 1
    state = QulacsQuantumState(n_qubit)
    circuit = QulacsParametricQuantumCircuit(n_qubit)
    add_H_gate!(circuit, 1)
    update_quantum_state!(circuit, state)
    @show get_vector(state)
    @test get_vector(state) == ComplexF64[0.7071067811865475 + 0.0im, 0.7071067811865475 + 0.0im]
end

@testset "create_A_gate" begin
    n_qubit = 2
    # Prepare |Psi> = |00> 
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b01) #X1|00> = |01> 
    theta, phi = 0.0, 0.0
    target_two_qubits = [1, 2]
    circuit = QulacsParametricQuantumCircuit(n_qubit)
    circuit_A = create_A_gate(circuit, theta, phi, target_two_qubits)

    #量子回路を作るときは、QulacsParametricQuantumCircuitにする。
    circuit = QulacsVariationalQuantumCircuit(circuit_A) #非常にややこしい。。
    @test num_theta(circuit) == 4 # 独立なパラメータは4つのうち,2つになるはず
    #@show 
    @show get_thetas(circuit)

    # update_quantum_stateをする前にQVQCに変換する
    update_quantum_state!(circuit, state)
    state_res = QulacsQuantumState(n_qubit)
    set_computational_basis!(state_res, 0b10) #X1|00> = |01>
    @test isapprox(-1.0 * get_vector(state_res), get_vector(state))
    thetas_init = rand(num_theta(circuit))
    update_circuit_param!(circuit, thetas_init)
    @show get_thetas(circuit)
end

@testset "hea_preserve_particle" begin
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


    state = QulacsQuantumState(n_qubit)
    #circuit = QulacsParametricQuantumCircuit(n_qubit) 
    depth = 4
    circuit = hea_preserve_particle(n_qubit, depth)
    theta_init = rand(num_theta(circuit))
    @show get_thetas(circuit)

    #Perform VQE
    cost_history, thetas_opt = QCMaterial.solve_gs(
        jordan_wigner(ham),
        circuit,
        state,
        theta_init = theta_init,
        verbose = true,
        comm = QCMaterial.MPI_COMM_WORLD,
    )
    @test abs(EigVal_min - cost_history[end]) < 1e-6

    #@show thetas_opt
    #VQE using ADAM
    m = zeros(length(theta_init))
    v = zeros(length(theta_init))
    cost_history_, thetas_opt_ = QCMaterial.solve_gs_Adam(
        jordan_wigner(ham),
        circuit,
        m,
        v,
        state,
        theta_init = theta_init,
        eta = 0.3,
        beta = [0.9, 0.999],
        n_steps = 300,
        verbose = true,
        comm = QCMaterial.MPI_COMM_WORLD,
    )
    @test abs(EigVal_min - cost_history_[end]) < 1e-3 #7.94934802622895e-5 
    @show thetas_opt_
    # current: depth * 4(# of params in A gates) * 3(# of A gates) = 4 * 4 * 3 = 48
    # true: compact form : depth * 2 * 3 (compact form)
    @test num_theta(circuit) == depth * 4 * 3
end

@testset "create_A_gate_compact" begin
    n_qubit = 2
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b01) #X1|00> = |01> 
    theta, phi = 0.0, 0.0
    target_two_qubits = [1, 2]
    i = target_two_qubits[1]
    j = target_two_qubits[2]
    circuit = QulacsParametricQuantumCircuit(n_qubit)
    parameterinfo = []
    circuit_A = create_A_gate_compact(
        circuit,
        theta,
        phi,
        target_two_qubits,
        parameterinfo,
        i,
        j,
        1,
    )
    #量子回路を作るときは、QulacsParametricQuantumCircuitにする。
    circuit = QulacsVariationalQuantumCircuit(circuit_A) #非常にややこしい。。
    @test num_theta(circuit) == 4 # 独立なパラメータは4つのうち,2つになるはず
    #@show 
    @show get_thetas(circuit)

    # update_quantum_stateをする前にQVQCに変換する
    update_quantum_state!(circuit, state)
    state_res = QulacsQuantumState(n_qubit)
    set_computational_basis!(state_res, 0b10) #X1|00> = |01>
    @test isapprox(-1.0 * get_vector(state_res), get_vector(state))
    thetas_init = rand(num_theta(circuit))
    update_circuit_param!(circuit, thetas_init)
    @show get_thetas(circuit)
    @show parameterinfo

    # same params with parameterinfo
    pinfo = ParamInfo(parameterinfo)
    #pinfo.nparam
    thetas_init_compact = rand(pinfo.nparam)
    @show thetas_init_compact
    theta_expand = expand(pinfo, thetas_init_compact)
    @show theta_expand
    update_circuit_param!(circuit, theta_expand)
    @show get_thetas(circuit)
end


@testset "hea_preserve_particle_compact" begin
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
    state0 = QulacsQuantumState(n_qubit)
    #circuit = QulacsParametricQuantumCircuit(n_qubit) 
    depth = 4
    circuit, keys = hea_preserve_particle_compact(n_qubit, depth)

    pinfo = QCMaterial.ParamInfo(keys)
    println("θ_unique length=", pinfo.nparam)
    println("θ_long length=", pinfo.nparamlong)
    println("θunique = rand(pinfo.nparam)", rand(pinfo.nparam))
    theta_init = rand(pinfo.nparam)

    #Perform VQE
    cost_history, thetas_opt = QCMaterial.solve_gs_kucj(
        jordan_wigner(ham),
        circuit,
        state0,
        theta_init = theta_init,
        keys,
        verbose = true,
        comm = QCMaterial.MPI_COMM_WORLD,
    )
    @test abs(EigVal_min - cost_history[end]) < 1e-4

    #@show thetas_opt
    #VQE using ADAM
    #m = zeros(length(theta_init))
    #v = zeros(length(theta_init)) 
    #cost_history_, thetas_opt_ =
    #QCMaterial.solve_gs_Adam(jordan_wigner(ham), circuit, m, v, state, theta_init=theta_init,eta = 0.3, beta = [0.9, 0.999], n_steps=300, verbose=true,
    #    comm=QCMaterial.MPI_COMM_WORLD
    #)
    #@test abs(EigVal_min-cost_history_[end]) < 1e-3 #7.94934802622895e-5 
    #@show thetas_opt_
    # current: depth * 4(# of params in A gates) * 3(# of A gates) = 4 * 4 * 3 = 48
    # true: compact form : depth * 2 * 3 (compact form)
    #@test num_theta(circuit) == depth * 4 * 3 
end