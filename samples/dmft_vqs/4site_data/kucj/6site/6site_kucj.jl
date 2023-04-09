using QCMaterial
using PyCall
using LinearAlgebra
using HDF5
import Random
import PyCall: pyimport
using Test


#beta = 1000 (T=0.001)
nsite = 6
n_qubit = 2 * nsite
U = 4.0
μ = U / 2
V = 0.0
ε = [0.0, -2.0, -1.0, 0.0, 1.0, 2.0]
d_theta = 1e-5
verbose = QCMaterial.MPI_rank == 0
Random.seed!(90) #first(90), second(120), thritd

#Hamiltonian
ham_op1 = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite)
n_electron_gs = 6
@assert mod(n_electron_gs, 2) == 0
sparse_mat = get_number_preserving_sparse_operator(ham_op1, n_qubit, n_electron_gs);
enes_ed = eigvals(sparse_mat.toarray());
EigVal_min = minimum(enes_ed)
count_eigvalmin = count(enes_ed.==EigVal_min)
println("Count_Ground_energy=", count_eigvalmin)
println("Ground energy=", EigVal_min)

#ansatz
state0 = create_hf_state(n_qubit, n_electron_gs)
vc, parameterinfo = kucj(n_qubit, k=1, sparse=true)
pinfo = QCMaterial.ParamInfo(parameterinfo)
@show pinfo.nparam
@show pinfo.nparamlong


##theta_opt for k-1 if k=0, nothing
theta_init = []
theta_init = read_and_parse_float(ARGS[1]) 
# Generate a normally-distributed random number of type T with mean 0 and standard deviation 1.
theta_rand = rand(pinfo.nparam-length(theta_init))
for i in 1:(pinfo.nparam-length(theta_init))
    push!(theta_init, theta_rand[i])
end
## theta_initにk=1で最適化したパラメータを突っ込む。
#Perform VQE
cost_history, thetas_opt =
    QCMaterial.solve_gs_kucj(jordan_wigner(ham_op1), vc, state0, parameterinfo, theta_init=theta_init, verbose=true,
        comm=QCMaterial.MPI_COMM_WORLD
    )
#debug
println("Ground energy_VQE=", cost_history[end])
println("Ground energy=", EigVal_min)
#@test abs(EigVal_min - cost_history[end]) < 1e-3
println("difference bet exact and VQE=", abs(EigVal_min - cost_history[end]))
# thetas_optを別ファイルに

compact_theta_opt  = make_compact_params(thetas_opt, parameterinfo ) 
#make_compact_paraminfo = compact_paraminfo(parameterinfo)
pushfirst!(compact_theta_opt, Int(pinfo.nparam))
write_to_txt_1("opt_params_kucjsparse_k2_seed90.txt", compact_theta_opt)

@test abs(EigVal_min - cost_history[end]) < 1e-3