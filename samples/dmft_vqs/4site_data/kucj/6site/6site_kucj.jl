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
nsite = 6
n_qubit = 2 * nsite
U = 4.0
#U = 0.0
μ = U / 2
V = 0.5
ε = [0.0, -2.0, -1.0, 0.0, 1.0, 2.0]
d_theta = 1e-5
verbose = QCMaterial.MPI_rank == 0
Random.seed!(90) #first(90), second(120), thritd

#Hamiltonian
#ham_op1 = generate_impurity_ham_with_1imp_3bath_dmft(U, μ, nsite)
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
#vc, parameterinfo = kucj(n_qubit, ucj=false)
vc, parameterinfo = kucj(n_qubit, k=3, sparse=true)

pinfo = QCMaterial.ParamInfo(parameterinfo)
#println("θ_unique length=", pinfo.nparam)
#println("θ_long length=", pinfo.nparamlong)
#println("θunique = rand(pinfo.nparam)", rand(pinfo.nparam))
theta_init = rand(pinfo.nparam)



## theta_initにk=1で最適化したパラメータを突っ込む。

#Perform VQE
cost_history, thetas_opt =
    QCMaterial.solve_gs_kucj(jordan_wigner(ham_op1), vc, state0, parameterinfo, theta_init=theta_init, verbose=true,
        comm=QCMaterial.MPI_COMM_WORLD
    )
#debug
println("Ground energy_VQE=", cost_history[end])
println("Ground energy=", EigVal_min)
@test abs(EigVal_min - cost_history[end]) < 1e-3

# thetas_optを別ファイルに

#==
## k=2
state0_k2 = create_hf_state(n_qubit, n_electron_gs)
#vc, parameterinfo = kucj(n_qubit, ucj=false)
vc_k2, parameterinfo_k2 = kucj(n_qubit, k=2, sparse=true)
pinfo_k2 = QCMaterial.ParamInfo(parameterinfo)
#println("θ_unique length=", pinfo.nparam)
#println("θ_long length=", pinfo.nparamlong)
#println("θunique = rand(pinfo.nparam)", rand(pinfo.nparam))
theta_init_k2 = rand(pinfo.nparam)

thetas_opt_k1
==#