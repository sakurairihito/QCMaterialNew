using Test
using QCMaterial
import Random
using LinearAlgebra

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
           comm=QCMaterial.MPI_COMM_WORLD
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