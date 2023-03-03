using Test
using QCMaterial
import Random
using LinearAlgebra
using MPI
@testset "prediction" begin
    @test predict(1.0, 2.0, 3.0) == 5.0
end

@testset "SGDupdate!" begin
    # training data sets
    ndata =32
    X = range(-π, stop=π, length=ndata)
    y_clean = X
    y = y_clean + 0.1randn(ndata) # 0.1を正規分布に持つノイズを入れている。
    theta_init = [0.4, 2.0]
    #theta_sgd = copy(theta_init)
    eta = 0.001
    n_steps = 10000
    sgd = QCMaterial.SGD(theta_init, eta)
    trainer_sgd = QCMaterial.Trainer(n_steps, mse, X, y)

    for i in 1:trainer_sgd.n_steps
        update!(sgd, trainer_sgd.mse, trainer_sgd.X, trainer_sgd.y)
    end

    opt_mse_sgd = mse(X, y, sgd.tmp_theta[1], sgd.tmp_theta[2])
    @test opt_mse_sgd < 0.03
end

@testset "MOMupdate!" begin
    # training data sets
    ndata =32
    X = range(-π, stop=π, length=ndata)
    y_clean = X
    y = y_clean + 0.1randn(ndata) # 0.1を正規分布に持つノイズを入れている。
    theta_mom = [0.4, 2.0]
    eta = 0.0001
    n_steps = 1000
    v = [0.0, 0.0]
    momentum =  0.0001
    mom = QCMaterial.Momentum(theta_mom, v, eta, momentum)
    trainer_mom = QCMaterial.Trainer(n_steps, mse, X, y)

    for i in 1:trainer_mom.n_steps
        update!(mom, trainer_mom.mse, trainer_mom.X, trainer_mom.y)
    end
    opt_mse_mom = mse(X, y, mom.tmp_theta[1], mom.tmp_theta[2])
    @show opt_mse_mom
    @test opt_mse_mom < 0.09
end

@testset "ADAGupdate!" begin
    # training data sets
    ndata =32
    X = range(-π, stop=π, length=ndata)
    y_clean = X
    y = y_clean + 0.1randn(ndata) # 0.1を正規分布に持つノイズを入れている。
    theta_adag = [0.4, 2.0]
    s = [0.0, 0.0]
    eta = 0.1
    eps = 1e-7
    n_steps = 1000
    adag = QCMaterial.AdaGrad(theta_adag, s, eta, eps)
    trainer_ada = QCMaterial.Trainer(n_steps, mse, X, y)

    for i in 1:trainer_ada.n_steps
        update!(adag, trainer_ada.mse, trainer_ada.X, trainer_ada.y)
    end
    opt_mse_ada = mse(X, y, adag.tmp_theta[1], adag.tmp_theta[2])
    @show opt_mse_ada
    @test opt_mse_ada < 0.03
end

@testset "RMSpupdate!" begin
    # training data sets
    ndata =32
    X = range(-π, stop=π, length=ndata)
    y_clean = X
    y = y_clean + 0.1randn(ndata) # 0.1を正規分布に持つノイズを入れている。
    theta_rmsp = [0.4, 2.0]
    s = [0.0, 0.0]
    eta = 0.2
    eps = 1e-7
    decay_rate = 0.999
    rmsp = QCMaterial.RMSprop(theta_rmsp, s, eta, eps, decay_rate)
    n_steps = 1000
    trainer_rmsp = QCMaterial.Trainer(n_steps, mse, X, y)

    for i in 1:trainer_rmsp.n_steps
        update!(rmsp, trainer_rmsp.mse, trainer_rmsp.X, trainer_rmsp.y)
    end
    opt_mse_rmsp = mse(X, y, rmsp.tmp_theta[1], rmsp.tmp_theta[2])
    @show opt_mse_rmsp
    @test opt_mse_rmsp < 0.03
end

@testset "ADAMupdate!" begin
    # training data sets
    ndata =32
    X = range(-π, stop=π, length=ndata)
    y_clean = X
    y = y_clean + 0.1randn(ndata) # 0.1を正規分布に持つノイズを入れている。
    theta_adam = [0.4, 2.0]
    m = [0.0, 0.0] # fill!(0, length(theta_rmsprop))
    v = [0.0, 0.0]
    eta = 0.2
    eps = 1e-7
    beta = [0.7, 0.777]
    adam = QCMaterial.Adam(theta_adam, m, v, eta, eps, beta)
    n_steps = 1000
    trainer_adam = QCMaterial.Trainer(n_steps, mse, X, y)
    update!(adam, trainer_adam.mse, trainer_adam.X, trainer_adam.y, n_steps)
    opt_mse_adam = mse(X, y, adam.tmp_theta[1], adam.tmp_theta[2])
    #opt_mse_adam2 = mse(X, y, adam.tmp_theta) koredemo OK!
    @show opt_mse_adam
    @test opt_mse_adam < 0.03
end



#quantum circuit
#init_param # variational parameters
#Expectation Value = Cost function

@testset "vqe.solve_gs" begin
    Random.seed!(100)
    nsite = 2 
    n_qubit = 2*nsite 
    U = 1.0
    t = -0.01
    μ = 1.0
        
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
        # Compute exact ground-state energy
    n_electron = 2
    @assert mod(n_electron, 2) == 0
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron);　
    EigVal_min = minimum(eigvals(sparse_mat.toarray()));
    
    # Ansatz
    state0 = create_hf_state(n_qubit, n_electron)
    circuit = uccgsd(n_qubit, orbital_rot=true)
    theta_init = rand(num_theta(circuit))
    
    # VQE
    m = zeros(length(theta_init))
    v = zeros(length(theta_init)) 
     cost_history, thetas_opt = 
       QCMaterial.solve_gs_Adam(jordan_wigner(ham), circuit, m, v, state0, theta_init=theta_init,eta = 0.01, beta = [0.9, 0.999], n_steps=100, verbose=true,
           comm=QCMaterial.MPI_COMM_WORLD, dx=1e-3
       )
    @show abs(EigVal_min-cost_history[end]) 
    @test abs(EigVal_min-cost_history[end]) < 1e-3
end

@testset "vqe.solve_gs_kucj_adam" begin
    Random.seed!(100)
    nsite = 2 
    n_qubit = 2*nsite 
    U = 1.0
    t = -0.01
    μ = 1.0
        
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
        # Compute exact ground-state energy
    n_electron = 2
    @assert mod(n_electron, 2) == 0
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron);　
    EigVal_min = minimum(eigvals(sparse_mat.toarray()));
    
    # Ansatz
    state0 = create_hf_state(n_qubit, n_electron)
    #circuit = kucj(n_qubit, orbital_rot=true)
    circuit, parameterinfo = kucj(n_qubit, k=2, sparse=false, oneimp=true, twoimp=false)
    theta_init = rand(num_theta(circuit))
    
    # VQE
    m = zeros(length(theta_init))
    v = zeros(length(theta_init)) 
     cost_history, thetas_opt = 
       QCMaterial.solve_gs_kucj_Adam(jordan_wigner(ham), circuit, m, v, state0, parameterinfo,theta_init=theta_init,eta = 0.01, beta = [0.9, 0.999], n_steps=100, verbose=true,
           comm=QCMaterial.MPI_COMM_WORLD
       )
    @show abs(EigVal_min-cost_history[end]) 
    @test abs(EigVal_min-cost_history[end]) < 1e-3
end


@testset "vqe.solve_gs_sampling" begin
    Random.seed!(100)
    nsite = 2 
    n_qubit = 2*nsite 
    U = 1.0
    t = -0.01
    μ = 1.0
        
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
        # Compute exact ground-state energy
    n_electron = 2
    @assert mod(n_electron, 2) == 0
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron);　
    EigVal_min = minimum(eigvals(sparse_mat.toarray()));
    
    # Ansatz
    state0 = create_hf_state(n_qubit, n_electron)
    circuit = uccgsd(n_qubit, orbital_rot=true)
    theta_init = rand(num_theta(circuit))
    
    # VQE
    m = zeros(length(theta_init))
    v = zeros(length(theta_init)) 
     cost_history, thetas_opt = 
       QCMaterial.solve_gs_sampling_Adam(jordan_wigner(ham), circuit, m, v, state0, theta_init=theta_init,eta = 0.01, beta = [0.9, 0.999], n_steps=60, verbose=true,
           comm=QCMaterial.MPI_COMM_WORLD, dx=1e-1, nshots=2^13
       )
    @show abs(EigVal_min-cost_history[end]) 
    @test abs(EigVal_min-cost_history[end]) < 1e-2
end




@testset "computation.apply_qubit_ham_sampling_Adam" begin
    MPI_COMM_WORLD = MPI.COMM_WORLD
    U = 1.0
    V = 1.0
    μ = 1.0
    ε = [1.0, 0.0]
    nsite = 2
    nqubits = nsite * 2
    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite) 
    ham_q = jordan_wigner(ham)  

    circuit_bra = uccgsd(nqubits)
    Random.seed!(120)
    #θ = rand(num_theta(circuit_bra))
    #θs = rand(num_theta)
    #⃗
    #θ⃗
    theta_init = rand(num_theta(circuit_bra))
    update_circuit_param!(circuit_bra, theta_init) 
    circuit_ket = uccgsd(nqubits)
    Random.seed!(90)
    theta_init2 = rand(num_theta(circuit_ket))
    update_circuit_param!(circuit_ket, theta_init2) 
    state_augmented = QulacsQuantumState(nqubits+1)
    set_computational_basis!(state_augmented, 0b00011)

    #cdag = jordan_wigner(FermionOperator("1^"))
    #fitting_ham = apply_qubit_ham_sampling!(ham_q, state_augmented, circuit_bra, circuit_ket, nshots=2^20, dx=1e-1)
    m = zeros(length(theta_init))
    v = zeros(length(theta_init)) 
    fitting_ham = QCMaterial.apply_qubit_ham_sampling_Adam!(
        ham_q, state_augmented, circuit_bra, circuit_ket, m, v, theta_init=theta_init, eta = 0.1, beta = [0.9, 0.99], n_steps=70, verbose=true,
           comm=QCMaterial.MPI_COMM_WORLD, dx=1e-1, nshots=2^14
    )
    @show fitting_ham

    #exact 
    circuit_bra = uccgsd(nqubits)
    Random.seed!(120)
    theta_init = rand(num_theta(circuit_bra))
    update_circuit_param!(circuit_bra, theta_init) 
    state0_bra = QulacsQuantumState(nqubits)
    set_computational_basis!(state0_bra, 0b0011)

    circuit_ket = uccgsd(nqubits)
    Random.seed!(90)
    theta_init2 = rand(num_theta(circuit_ket))
    update_circuit_param!(circuit_ket, theta_init2) 
    state_ket = QulacsQuantumState(nqubits)
    set_computational_basis!(state_ket, 0b0011)
    update_quantum_state!(circuit_ket, state_ket)

    #exact = get_transition_amplitude_with_obs(circuit_bra, state0_bra, ham_q, state_ket)
    verbose = true
    maxiter = 500
    gtol = 1e-8
    squared_norm1 = apply_qubit_ham!(ham_q, state0_bra, circuit_bra, state_ket, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
        verbose=verbose)
        )
    println("squared_norm1=", squared_norm1)

    #@show (fitting_norm_sampling) 
    #@show real(fitting_norm_sampling) 
    @show squared_norm1
    err_abs = abs(real(fitting_ham) - squared_norm1)

    @show err_abs
    err_rel = err_abs/abs(squared_norm1)
    @show err_rel
    #@test isapprox(real(fitting_ham), squared_norm1, rtol=1e-1)   
end



@testset "computation.apply_qubit_op_sampling_Adam" begin
    MPI_COMM_WORLD = MPI.COMM_WORLD
    U = 1.0
    V = 1.0
    μ = 1.0
    ε = [1.0, 0.0]
    nsite = 2
    nqubits = nsite * 2
    #ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite) 
    #ham_q = jordan_wigner(ham)  

    circuit_bra = uccgsd(nqubits)
    Random.seed!(120)
    theta_init = rand(num_theta(circuit_bra))
    update_circuit_param!(circuit_bra, theta_init) 
    circuit_ket = uccgsd(nqubits)
    Random.seed!(90)
    theta_init2 = rand(num_theta(circuit_ket))
    update_circuit_param!(circuit_ket, theta_init2) 
    state_augmented = QulacsQuantumState(nqubits+1)
    set_computational_basis!(state_augmented, 0b00011)

    cdag = jordan_wigner(FermionOperator("1^"))
    m = zeros(length(theta_init))
    v = zeros(length(theta_init)) 
    #fitting_ham = apply_qubit_ham_sampling!(ham_q, state_augmented, circuit_bra, circuit_ket, nshots=2^20, dx=1e-1)
    fitting_ham = QCMaterial.apply_qubit_op_sampling_Adam!(
        cdag, state_augmented, circuit_bra, circuit_ket, m, v, theta_init=theta_init, eta = 0.1, beta = [0.9, 0.999], n_steps=100, verbose=true,
           comm=QCMaterial.MPI_COMM_WORLD, dx=1e-1, nshots=2^13
    )
    @show fitting_ham

    #exact 
    circuit_bra = uccgsd(nqubits)
    Random.seed!(120)
    theta_init = rand(num_theta(circuit_bra))
    update_circuit_param!(circuit_bra, theta_init) 
    state0_bra = QulacsQuantumState(nqubits)
    set_computational_basis!(state0_bra, 0b0011)

    circuit_ket = uccgsd(nqubits)
    Random.seed!(90)
    theta_init2 = rand(num_theta(circuit_ket))
    update_circuit_param!(circuit_ket, theta_init2) 
    state_ket = QulacsQuantumState(nqubits)
    set_computational_basis!(state_ket, 0b0011)
    update_quantum_state!(circuit_ket, state_ket)

    #exact = get_transition_amplitude_with_obs(circuit_bra, state0_bra, ham_q, state_ket)
    verbose = true
    maxiter = 500
    gtol = 1e-8
    squared_norm1 = apply_qubit_op!(cdag, state0_bra, circuit_bra, state_ket, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
        verbose=verbose)
        )
    println("squared_norm1=", squared_norm1)

    #@show (fitting_norm_sampling) 
    #@show real(fitting_norm_sampling) 
    @show squared_norm1
    err_abs = abs(real(fitting_ham) - squared_norm1)

    @show err_abs
    err_rel = err_abs/abs(squared_norm1)
    @show err_rel
    @test isapprox(real(fitting_ham), squared_norm1, rtol=1e-1)   
end