using Test
using LinearAlgebra
using QCMaterial

import PyCall: pyimport
import Random

@testset "vqe.solve_gs" begin
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

    @test abs(EigVal_min-cost_history[end]) < 1e-6 
end