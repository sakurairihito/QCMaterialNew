using QCMaterial
using PyCall
using LinearAlgebra

import Random
import PyCall: pyimport

"tau plus"
nsite = 2
n_qubit　= 2*nsite
U = 1.0
V = 1.0
μ = 1.0
ε = 1.0
d_theta = 1e-5

#Hamiltonian
ham_op = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite)
    
n_electron_gs = 2
@assert mod(n_electron_gs, 2) == 0
sparse_mat = get_number_preserving_sparse_operator(ham_op, n_qubit, n_electron_gs);
enes_ed = eigvals(sparse_mat.toarray());

#debug
println("Ground energy_ED=",minimum(enes_ed))

#ansatz
state0 = create_hf_state(n_qubit, n_electron_gs)
vc = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=true)
theta_init = rand(num_theta(vc))

#Perform VQE
cost_history, thetas_opt = 
QCMaterial.solve_gs(jordan_wigner(ham_op), vc, state0, theta_init=theta_init, verbose=true,
    comm=QCMaterial.MPI_COMM_WORLD
)

#debug
println("Ground energy_VQE=",cost_history[end])

#c^{dag},c
up1 = up_index(1)
down1 = down_index(1)
up2 = up_index(2)
down2 = down_index(2)
#right_op = FermionOperator("$(up1)^ ", 1.0)
#right_op = jordan_wigner(right_op)
#left_op = FermionOperator("$(up1) ", 1.0)
#left_op = jordan_wigner(left_op)

#1down
right_op = FermionOperator("$(down1)^ ", 1.0)
right_op = jordan_wigner(right_op)
left_op = FermionOperator("$(down1) ", 1.0)
left_op = jordan_wigner(left_op)

##Ansatz -> apply_qubit_op & imag_time_evolve
vc_ex = uccgsd(n_qubit, orbital_rot=true)

#state_gs = QulacsQuantumState(n_qubit,0b0000)
state_gs = create_hf_state(n_qubit, n_electron_gs)
update_circuit_param!(vc, thetas_opt)
update_quantum_state!(vc, state_gs)

n_electron_ex = 3
state0_ex = create_hf_state(n_qubit, n_electron_ex)

taus = collect(range(0.0, 4.0, length=3000))
beta = taus[end]

Gfunc_ij_list = compute_gtau(jordan_wigner(ham_op), left_op, right_op, vc_ex,  state_gs, state0_ex, taus, d_theta)
println("Gfunc_ij_list=", Gfunc_ij_list)

function write_to_txt(file_name, x, y)
    open(file_name,"w") do fp
        for i in 1:length(x)
            println(fp, x[i], " ", real(y[i]))
        end
    end
end

write_to_txt("g_of_tau_dimer_buntenn_3000.txt", taus, Gfunc_ij_list)



  
