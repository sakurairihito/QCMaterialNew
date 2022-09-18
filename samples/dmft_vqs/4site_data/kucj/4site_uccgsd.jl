using QCMaterial
using PyCall
using LinearAlgebra
using HDF5
import Random
import PyCall: pyimport
using Test


function generate_impurity_ham_with_1imp_multibath(U::Float64, V::Float64, μ::Float64, ε::Vector{Float64}, nsite::Integer)
    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)
    ham = FermionOperator()

    #Coulomb   
    ham += FermionOperator("$(up_index(1))^ $(down_index(1))^ $(up_index(1)) $(down_index(1))", -U)

    for ispin in [1, 2]
        for i in 2:nsite
            ham += FermionOperator("$(so_idx(1, ispin))^ $(so_idx(i, ispin))", -V)
            ham += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(1, ispin))", -V)
        end
    end

    #chemical potential
    for ispin in [1, 2]
        ham += FermionOperator("$(so_idx(1, ispin))^ $(so_idx(1, ispin))", -μ)
    end

    for ispin in [1, 2]
        for i in 2:nsite
            ham += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(i, ispin))", ε[i])
        end
    end
    ham
end

#beta = 1000 (T=0.001)
nsite = 4
n_qubit = 2 * nsite
U = 4.0
μ = U / 2
V = 1.0
ε = [0.0, -1.0, 0.0, 1.0]
d_theta = 1e-5
verbose = QCMaterial.MPI_rank == 0
Random.seed!(90)


#Hamiltonian
#ham_op1 = generate_impurity_ham_with_1imp_3bath_dmft(U, μ, nsite)
ham_op1 = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite)
n_electron_gs = 4
@assert mod(n_electron_gs, 2) == 0
sparse_mat = get_number_preserving_sparse_operator(ham_op1, n_qubit, n_electron_gs);
enes_ed = eigvals(sparse_mat.toarray());
EigVal_min = minimum(enes_ed)
println("Ground energy=", EigVal_min)

#ansatz
state0 = create_hf_state(n_qubit, n_electron_gs)
vc = uccgsd(n_qubit, nocc=2, orbital_rot=true, uccgsd=true, p_uccgsd=false)
#depth = n_qubit*2
#depth = n_qubit*2
#vc = hev(n_qubit, depth)
theta_init = rand(num_theta(vc))

#Perform VQE
cost_history, thetas_opt =
    QCMaterial.solve_gs(jordan_wigner(ham_op1), vc, state0, theta_init=theta_init, verbose=true,
        comm=QCMaterial.MPI_COMM_WORLD
    )

#debug
println("Ground energy_VQE=", cost_history[end])
println("Ground energy=", EigVal_min)
@test abs(EigVal_min - cost_history[end]) < 1e-3