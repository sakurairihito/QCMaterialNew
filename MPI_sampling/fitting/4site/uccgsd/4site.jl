

using QCMaterial
using PyCall
using LinearAlgebra
using HDF5
import Random
import PyCall: pyimport
using Test
import MPI


##### 0次のモーメント
Random.seed!(100)
nsite = 4
n_qubit = 2 * nsite 
U = 4.0
μ = U/2

ham = generate_impurity_ham_with_1imp_3bath_dmft(U, μ, nsite)
#@show ham 
n_electron = 4 
sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron);
EigVal_min = minimum(eigvals(sparse_mat.toarray()));
println("sparse_mat.toarray())=", size(sparse_mat.toarray()))
#ham = jordan_wigner(ham_op)
#@show op
#state0 = create_hf_state(n_qubit, n_electron)
state = QulacsQuantumState(n_qubit)
circuit_gs = uccgsd(n_qubit, orbital_rot=true, nx=4)
theta_init = rand(num_theta(circuit_gs))
#println("θ_unique length=", pinfo.nparam)
#println("θ_long length=", pinfo.nparamlong)
#cost_history ,thetas_opt = solve_gs_sampling(ham_q, circuit_gs, state)\
nshots=10000
cost_history ,thetas_opt = solve_gs_sampling(jordan_wigner(ham), circuit_gs, state, nshots=nshots)
#cost_history, thetas_opt = 
#QCMaterial.solve_gs(jordan_wigner(ham), circuit_gs, state, theta_init=theta_init, verbose=true,
#    comm=QCMaterial.MPI_COMM_WORLD
#)
@show cost_history 
@show cost_history[end] 
@show EigVal_min 
@show abs(EigVal_min - cost_history[end])  
#ß@test abs(EigVal_min - cost_history[end]) < 1e-1 


ntest = 1
norm_list = []
rela_err = []

circuit_cdag = uccgsd(n_qubit , nx=5) 
Random.seed!(90)
theta_init = rand(num_theta(circuit_cdag))
update_circuit_param!(circuit_cdag, theta_init) 
#circuit_ket = uccgsd(n_qubit , nx=2)
#update_circuit_param!(circuit_ket, thetas_opt);
state_augmented = QulacsQuantumState(n_qubit+1) 
#set_computational_basis!(state_augmented, 0b00000) 
op_q = FermionOperator(" 1^", 1.0) 
op_q = jordan_wigner(op_q) 

#nshots = 50
println("before fitting op")
fitting_fop = apply_qubit_op_sampling_vqelike!(op_q, state_augmented, circuit_cdag, circuit_gs, nshots=nshots, dx=0.1)
@show fitting_fop

c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
#set_computational_basis!(state_augmented, 0b00000)

op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag)
#op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, her, circuit_cdag)
#op_re_ = op_re_re + im * op_re_im
#op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, antiher, circuit_cdag)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag)
#op_im_ = op_im_re + im * op_im_im 
mom0 = op_re_re + op_im_im * im * im
mom0 = mom0 * fitting_fop

mom0_exact_2=0.4999999999999979
err_rel = abs(real(mom0) - mom0_exact_2) / real(mom0_exact_2)

comm=QCMaterial.MPI_COMM_WORLD
if comm !== nothing
    err_rel = MPI.Allreduce(err_rel, MPI.SUM, comm)
    err_rel = err_rel/MPI.Comm_size(comm)
end

@show err_rel





# mom1_exact_2=1.0705354590372629
# mom2_exact_2=3.5972258098000203 + 0.0im
# mom3_exact_2=14.835226120399732 + 0.0im
ham_op_mom = generate_impurity_ham_with_1imp_3bath_dmft(U, μ, nsite)
E0  = FermionOperator(" ", cost_history[end])
ham_op_mom = (ham_op_mom - E0)

circuit_cdag_h = uccgsd(n_qubit , nx=5) 
Random.seed!(90)
theta_init = rand(num_theta(circuit_cdag_h))
update_circuit_param!(circuit_cdag_h, theta_init) 
state_augmented = QulacsQuantumState(n_qubit+1) 
#set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
#nshots = 1000
fitting_h = apply_qubit_ham_sampling_vqelike!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h, circuit_cdag, nshots=nshots, dx=0.1)
@show fitting_h


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
#set_computational_basis!(state_augmented, 0b00000)

op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h)
mom1 = op_re_re + op_im_im * im * im
@show mom1
mom1 = mom1 * fitting_fop * fitting_h
@show mom1

#=
circuit_cdag_h2 = uccgsd(n_qubit , nx=5) 
Random.seed!(90)
theta_init = rand(num_theta(circuit_cdag_h2))
update_circuit_param!(circuit_cdag_h, theta_init) 
state_augmented = QulacsQuantumState(n_qubit+1) 
#set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
nshots = 2^15
fitting_h2 = apply_qubit_ham_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h2, circuit_cdag_h, nshots=nshots, dx=0.1)
@show fitting_h2


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
#set_computational_basis!(state_augmented, 0b00000)

op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h2)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h2)
mom2 = op_re_re + op_im_im * im * im
@show mom1
mom2 = mom2 * fitting_fop * fitting_h * fitting_h2
@show mom2


circuit_cdag_h3 = uccgsd(n_qubit , nx=5) 
Random.seed!(90)
theta_init = rand(num_theta(circuit_cdag_h3))
update_circuit_param!(circuit_cdag_h, theta_init) 
state_augmented = QulacsQuantumState(n_qubit+1) 
#set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
nshots = 2^15
fitting_h3 = apply_qubit_ham_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h3, circuit_cdag_h2, nshots=nshots, dx=0.1)
@show fitting_h3


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
#set_computational_basis!(state_augmented, 0b00000)

op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h3)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h3)
mom3 = op_re_re + op_im_im * im * im
@show mom3
mom3 = mom3 * fitting_fop * fitting_h * fitting_h2 * fitting_h3
@show mom3

##
circuit_cdag_h4 = uccgsd(n_qubit , nx=5) 
Random.seed!(90)
theta_init = rand(num_theta(circuit_cdag_h4))
update_circuit_param!(circuit_cdag_h, theta_init) 
state_augmented = QulacsQuantumState(n_qubit+1) 
#set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
nshots = 2^15
fitting_h4 = apply_qubit_ham_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h4, circuit_cdag_h3, nshots=nshots, dx=0.1)
@show fitting_h4


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
#set_computational_basis!(state_augmented, 0b00000)

op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h4)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h4)
mom4 = op_re_re + op_im_im * im * im
@show mom4
mom4 = mom4 * fitting_fop * fitting_h * fitting_h2 * fitting_h3 * fitting_h4
@show mom4

##
circuit_cdag_h5 = uccgsd(n_qubit , nx=5) 
Random.seed!(90)
theta_init = rand(num_theta(circuit_cdag_h5))
update_circuit_param!(circuit_cdag_h, theta_init) 
state_augmented = QulacsQuantumState(n_qubit+1) 
#set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
nshots = 2^15
fitting_h5 = apply_qubit_ham_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h5, circuit_cdag_h4, nshots=nshots, dx=0.1)
@show fitting_h5


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
#set_computational_basis!(state_augmented, 0b00000)

op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h5)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h5)
mom5 = op_re_re + op_im_im * im * im
@show mom5
mom5 = mom5 * fitting_fop * fitting_h * fitting_h2 * fitting_h3 * fitting_h4 * fitting_h5
@show mom5


@show mom0
@show mom1
@show mom2
@show mom3
@show mom4
@show mom5

=#