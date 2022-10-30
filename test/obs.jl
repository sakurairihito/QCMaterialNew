using Test
using QCMaterial
using LinearAlgebra
using Random

@testset "num_op" begin
    op1 = num_op(1)
    op2 = num_op_up(1)
    op3 = num_op_down(1)
    @test op1 == op2 * op3
end

@testset "dabron" begin
    Random.seed!(100)
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
    cost_history, thetas_opt = 
       QCMaterial.solve_gs(jordan_wigner(ham), circuit, state0, theta_init=theta_init, verbose=true,
           comm=QCMaterial.MPI_COMM_WORLD
       )
    
    op1 = num_op(1) #number op at 1 site
    #dabron(vc, theta_list, numop, state0)
    expecval_op1 = dabron(circuit, thetas_opt, op1, state0)
    vec = eigvecs(sparse_mat.toarray())[:, 1] #最低固有ベクトル
    exact_expecval_op1 = dabron_exact(op1, n_qubit, vec)
    @test abs(expecval_op1 - exact_expecval_op1) < 1e-3
end