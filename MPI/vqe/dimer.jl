using  MPI
MPI.Init()

using QCMaterial
using PyCall
using LinearAlgebra
using HDF5
import Random
import PyCall: pyimport
using Test



# Make Hamiltonian
U = 1.0
V = 1.0
μ = U/2
ε = [1.0, 1.0]
nsite = 2
n_qubit = nsite * 2
ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite) 
#@show ham 
n_electron = 2 
sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron) #行列の取得 
enes_ed = eigvals(sparse_mat.toarray()) #対角化を行う
EigVal_min = minimum(enes_ed) 
#@show EigVal_min 
#ham_q = jordan_wigner(ham) 
#@show ham_q 

#n_electron_gs = 2
#state = create_hf_state(n_qubit, n_electron_gs)
state = QulacsQuantumState(n_qubit)
circuit_gs = uccgsd(n_qubit, nx=2)
Random.seed!(90)
theta_init = rand(num_theta(circuit_gs));
update_circuit_param!(circuit_gs, theta_init)


#cost_history ,thetas_opt = solve_gs_sampling(ham_q, circuit_gs, state)\
nshots=2^15 
cost_history ,thetas_opt = solve_gs_sampling(jordan_wigner(ham), circuit_gs, state, nshots=nshots, verbose=true)
#cost_history, thetas_opt = 
#QCMaterial.solve_gs(jordan_wigner(ham), circuit_gs, state, theta_init=theta_init, verbose=true,
#    comm=QCMaterial.MPI_COMM_WORLD
#)
@show cost_history[end] 
