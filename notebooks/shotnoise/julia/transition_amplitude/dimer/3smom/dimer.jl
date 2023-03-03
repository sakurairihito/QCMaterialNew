using QCMaterial
using PyCall
using LinearAlgebra
using HDF5
import Random
import PyCall: pyimport
using Test
import MPI


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
cost_history ,thetas_opt = solve_gs_sampling(jordan_wigner(ham), circuit_gs, state, nshots=nshots)
#cost_history, thetas_opt = 
#QCMaterial.solve_gs(jordan_wigner(ham), circuit_gs, state, theta_init=theta_init, verbose=true,
#    comm=QCMaterial.MPI_COMM_WORLD
#)
@show cost_history 
@show cost_history[end] 
@show EigVal_min 
@show abs(EigVal_min - cost_history[end])  
@test abs(EigVal_min - cost_history[end]) < 1e-1 


ntest = 1
norm_list = []
rela_err = []

circuit_cdag = uccgsd(n_qubit , nx=3) 
Random.seed!(90)
theta_init = rand(num_theta(circuit_cdag))
update_circuit_param!(circuit_cdag, theta_init) 
#circuit_ket = uccgsd(n_qubit , nx=2)
#update_circuit_param!(circuit_ket, thetas_opt);
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 
op_q = FermionOperator(" 1^", 1.0) 
op_q = jordan_wigner(op_q) 

nshots = 2^15
fitting_fop = apply_qubit_op_sampling!(op_q, state_augmented, circuit_cdag, circuit_gs, nshots=nshots, dx=0.1)
@show fitting_fop
push!(norm_list, fitting_fop)

squared_norm = 0.5620610332874679
#@show abs(fitting_fop + squared_norm)/squared_norm
push!(rela_err, abs(fitting_fop - squared_norm)/squared_norm )

@show norm_list
@show rela_err


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag)
#op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, her, circuit_cdag)
#op_re_ = op_re_re + im * op_re_im
#op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, antiher, circuit_cdag)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag)
#op_im_ = op_im_re + im * op_im_im 
mom0 = op_re_re + op_im_im * im * im
mom0 = mom0 * fitting_fop

mom0_exact_2 = 0.3159126138504742 + 0.0im
@show mom0
@show mom0_exact_2 
err_ab = abs(real(mom0) - mom0_exact_2)
@show err_ab
err_rel = err_ab / real(mom0_exact_2)
@show err_rel



ham_op_mom = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite) 
E0  = FermionOperator(" ", cost_history[end])
ham_op_mom = (ham_op_mom - E0)

circuit_cdag_h = uccgsd(n_qubit , nx=3) 
Random.seed!(90)
theta_init = rand(num_theta(circuit_cdag_h))
update_circuit_param!(circuit_cdag_h, theta_init) 
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
nshots = 2^15
fitting_h = apply_qubit_ham_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h, circuit_cdag, nshots=nshots, dx=0.1)
@show fitting_h


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h)
mom1 = op_re_re + op_im_im * im * im
@show mom1
mom1 = mom1 * fitting_fop * fitting_h
@show mom1


mom1_exact_2=0.541013366603164 + 0.0im
@show abs(mom1 - mom1_exact_2)/abs(mom1_exact_2)


circuit_cdag_h2 = uccgsd(n_qubit , nx=3) 
Random.seed!(90)
theta_init = rand(num_theta(circuit_cdag_h2))
update_circuit_param!(circuit_cdag_h, theta_init) 
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
nshots = 2^15
fitting_h2 = apply_qubit_ham_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h2, circuit_cdag_h, nshots=nshots, dx=0.1)
@show fitting_h2


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h2)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h2)
mom2 = op_re_re + op_im_im * im * im
@show mom1
mom2 = mom2 * fitting_fop * fitting_h * fitting_h2
@show mom2


circuit_cdag_h3 = uccgsd(n_qubit , nx=3) 
Random.seed!(90)
theta_init = rand(num_theta(circuit_cdag_h3))
update_circuit_param!(circuit_cdag_h, theta_init) 
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
nshots = 2^15
fitting_h3 = apply_qubit_ham_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h3, circuit_cdag_h2, nshots=nshots, dx=0.1)
@show fitting_h3


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h3)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h3)
mom3 = op_re_re + op_im_im * im * im
@show mom3
mom3 = mom3 * fitting_fop * fitting_h * fitting_h2 * fitting_h3
@show mom3

##
circuit_cdag_h4 = uccgsd(n_qubit , nx=3) 
Random.seed!(90)
theta_init = rand(num_theta(circuit_cdag_h4))
update_circuit_param!(circuit_cdag_h, theta_init) 
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
nshots = 2^15
fitting_h4 = apply_qubit_ham_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h4, circuit_cdag_h3, nshots=nshots, dx=0.1)
@show fitting_h4


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h4)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h4)
mom4 = op_re_re + op_im_im * im * im
@show mom4
mom4 = mom4 * fitting_fop * fitting_h * fitting_h2 * fitting_h3 * fitting_h4
@show mom4

##
circuit_cdag_h5 = uccgsd(n_qubit , nx=3) 
Random.seed!(90)
theta_init = rand(num_theta(circuit_cdag_h5))
update_circuit_param!(circuit_cdag_h, theta_init) 
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
nshots = 2^15
fitting_h5 = apply_qubit_ham_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h5, circuit_cdag_h4, nshots=nshots, dx=0.1)
@show fitting_h5


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

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