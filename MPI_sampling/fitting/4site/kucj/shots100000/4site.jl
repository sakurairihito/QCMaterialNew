using QCMaterial
using PyCall
using LinearAlgebra
using HDF5
import Random
import PyCall: pyimport
using Test
import MPI

# Make Hamiltonian

Random.seed!(100)
nsite = 4
n_qubit = 2 * nsite 
U = 4.0
μ = U/2
n_electron = 4 
ham = generate_impurity_ham_with_1imp_3bath_dmft(U, μ, nsite)

sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron) #行列の取得 
enes_ed = eigvals(sparse_mat.toarray()) #対角化を行う
EigVal_min = minimum(enes_ed) 
state = QulacsQuantumState(n_qubit)
circuit_gs, parameterinfo = kucj(n_qubit, k=5, sparse=false, nx =4)
pinfo = QCMaterial.ParamInfo(parameterinfo)
theta_init = rand(pinfo.nparam)

nshots=10000
cost_history ,thetas_opt = solve_gs_kucj_sampling(jordan_wigner(ham), circuit_gs, state, parameterinfo, theta_init=theta_init, nshots=nshots)
#cost_history ,thetas_opt = QCMaterial.solve_gs_kucj(jordan_wigner(ham), circuit_gs, state, parameterinfo, theta_init=theta_init, nshots=nshots)
#cost_history, thetas_opt =
#QCMaterial.solve_gs_kucj(jordan_wigner(ham), circuit_gs, state, parameterinfo, theta_init=theta_init, verbose=true,
#    comm=QCMaterial.MPI_COMM_WORLD
#    )

@show cost_history 
@show cost_history[end] 
@show EigVal_min 
@show abs(EigVal_min - cost_history[end])  
#@test abs(EigVal_min - cost_history[end]) < 1e-1 

ntest = 1
norm_list = []
rela_err = []

circuit_cdag, parameterinfo = kucj(n_qubit, k=5, sparse=false, nx=5)
pinfo = QCMaterial.ParamInfo(parameterinfo)
theta_init = rand(pinfo.nparam)

#circuit_ket = uccgsd(n_qubit , nx=2)
#update_circuit_param!(circuit_ket, thetas_opt);
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 
op_q = FermionOperator(" 1^", 1.0) 
op_q = jordan_wigner(op_q) 

@show "before apply_qubit_op_sampling"
#nshots = 1000
#fitting_fop = apply_qubit_op_sampling!(op_q, state_augmented, circuit_cdag, circuit_gs, nshots=nshots, dx=0.1)
fitting_fop = apply_qubit_op_kucj_sampling!(op_q, state_augmented, circuit_cdag, circuit_gs, parameterinfo,theta_init=theta_init, nshots=nshots, dx=0.1)
@show fitting_fop
#push!(norm_list, fitting_fop)

#squared_norm = 0.5620610332874679
#@show abs(fitting_fop + squared_norm)/squared_norm
#push!(rela_err, abs(fitting_fop - squared_norm)/squared_norm )

#@show norm_list
#@show rela_err


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag)
op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, her, circuit_cdag)
op_re_ = op_re_re + im * op_re_im
op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, antiher, circuit_cdag)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag)
op_im_ = op_im_re + im * op_im_im 
mom0 = op_re_re + op_im_im * im * im

#=
if comm !== nothing
    mom0 = MPI.Allreduce(mom0, MPI.SUM, comm)
    mom0 = mom0/MPI.Comm_size(comm)
end
=#



mom0 = mom0 * fitting_fop

comm=QCMaterial.MPI_COMM_WORLD
if comm !== nothing
    mom0 = MPI.Allreduce(mom0, MPI.SUM, comm)
    mom0 = mom0/MPI.Comm_size(comm)
end
@show mom0
#@show mom0
mom0_exact_2 = 0.4999999999999979
#err_ab = abs(real(mom0) - mom0_exact_2)
err_rel = abs(real(mom0) - mom0_exact_2) / real(mom0_exact_2)
@show err_rel



@show err_rel


# mom1
ham_op_mom = generate_impurity_ham_with_1imp_3bath_dmft(U, μ, nsite)
E0  = FermionOperator(" ", cost_history[end])
ham_op_mom = (ham_op_mom - E0)

#circuit_cdag_h = uccgsd(n_qubit , nx=3) 
#Random.seed!(90)
#theta_init = rand(num_theta(circuit_cdag_h))
circuit_cdag_h, parameterinfo = kucj(n_qubit, k=5, sparse=false, nx=5)
pinfo = QCMaterial.ParamInfo(parameterinfo)
theta_init = rand(pinfo.nparam)

#update_circuit_param!(circuit_cdag_h, theta_init) 
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
#nshots = 1000
fitting_h = apply_qubit_ham_kucj_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h, circuit_cdag, parameterinfo,theta_init=theta_init, nshots=nshots, dx=0.1)
#@show fitting_h


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

p_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h)
op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, her, circuit_cdag_h)
op_re_ = op_re_re + im * op_re_im
op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, antiher, circuit_cdag_h)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h)
op_im_ = op_im_re + im * op_im_im
mom1 = op_re_ + op_im_ * im
#@show mom1
mom1 = mom1 * fitting_fop * fitting_h
#@show mom1
if comm !== nothing
    mom1 = MPI.Allreduce(mom1, MPI.SUM, comm)
    mom1 = mom1/MPI.Comm_size(comm)
end
@show mom1
mom1_exact_2=1.0705354590372629
err_rel_mom1 =  abs(mom1 - mom1_exact_2)/abs(mom1_exact_2)



@show err_rel_mom1
# print result




# mom2

circuit_cdag_h2, parameterinfo = kucj(n_qubit, k=5, sparse=false, nx=5)
pinfo = QCMaterial.ParamInfo(parameterinfo)
theta_init = rand(pinfo.nparam)
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
#nshots = 1000
fitting_h2 = apply_qubit_ham_kucj_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h2, circuit_cdag_h, parameterinfo,theta_init=theta_init, nshots=nshots, dx=0.1)
#apply_qubit_ham_kucj_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h, circuit_cdag, parameterinfo,theta_init=theta_init, nshots=nshots, dx=0.1)
#@show fitting_h2


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

p_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h2)
op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, her, circuit_cdag_h2)
op_re_ = op_re_re + im * op_re_im
op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, antiher, circuit_cdag_h2)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h2)
op_im_ = op_im_re + im * op_im_im
mom2 = op_re_ + op_im_ * im
mom2 = mom2 * fitting_fop * fitting_h * fitting_h2
#@show mom2

if comm !== nothing
    mom2 = MPI.Allreduce(mom2, MPI.SUM, comm)
    mom2 = mom2/MPI.Comm_size(comm)
end
@show mom2
mom2_exact_2=3.5972258098000203
err_rel_mom2 =  abs(mom2 - mom2_exact_2)/abs(mom2_exact_2)



@show err_rel_mom2


circuit_cdag_h3, parameterinfo = kucj(n_qubit, k=5, sparse=false, nx=5)
pinfo = QCMaterial.ParamInfo(parameterinfo)
theta_init = rand(pinfo.nparam)
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
#nshots = 1000
#fitting_h3 = apply_qubit_ham_sampling_vqelike!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h3, circuit_cdag_h2, nshots=nshots, dx=0.1)
fitting_h3 = apply_qubit_ham_kucj_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h3, circuit_cdag_h2, parameterinfo,theta_init=theta_init, nshots=nshots, dx=0.1)

@show fitting_h3


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

p_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h3)
op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, her, circuit_cdag_h3)
op_re_ = op_re_re + im * op_re_im
op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, antiher, circuit_cdag_h3)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h3)
op_im_ = op_im_re + im * op_im_im
mom3 = op_re_ + op_im_ * im
#@show mom3
mom3 = mom3 * fitting_fop * fitting_h * fitting_h2 * fitting_h3
#@show mom3

if comm !== nothing
    mom3 = MPI.Allreduce(mom3, MPI.SUM, comm)
    mom3 = mom3/MPI.Comm_size(comm)
end
@show mom3
mom3_exact_2=14.835226120399728
err_rel_mom3 =  abs(mom3 - mom3_exact_2)/abs(mom3_exact_2)



@show err_rel_mom3
println("hello world")



circuit_cdag_h4, parameterinfo = kucj(n_qubit, k=5, sparse=false, nx=5)
pinfo = QCMaterial.ParamInfo(parameterinfo)
theta_init = rand(pinfo.nparam)
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
#nshots = 1000
#fitting_h3 = apply_qubit_ham_sampling_vqelike!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h3, circuit_cdag_h2, nshots=nshots, dx=0.1)
fitting_h4 = apply_qubit_ham_kucj_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h4, circuit_cdag_h3, parameterinfo,theta_init=theta_init, nshots=nshots, dx=0.1)

@show fitting_h4


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

p_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h4)
op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, her, circuit_cdag_h4)
op_re_ = op_re_re + im * op_re_im
op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, antiher, circuit_cdag_h4)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h4)
op_im_ = op_im_re + im * op_im_im
mom4 = op_re_ + op_im_ * im
#@show mom3
mom4 = mom4 * fitting_fop * fitting_h * fitting_h2 * fitting_h3 * fitting_h4
#

if comm !== nothing
    mom4 = MPI.Allreduce(mom4, MPI.SUM, comm)
    mom4 = mom4/MPI.Comm_size(comm)
end

@show mom4
mom4_exact_2=71.32641573302384
err_rel_mom4 =  abs(mom4 - mom4_exact_2)/abs(mom4_exact_2)



@show err_rel_mom4
println("hello world")


circuit_cdag_h5, parameterinfo = kucj(n_qubit, k=5, sparse=false, nx=5)
pinfo = QCMaterial.ParamInfo(parameterinfo)
theta_init = rand(pinfo.nparam)
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
#nshots = 1000
#fitting_h3 = apply_qubit_ham_sampling_vqelike!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h3, circuit_cdag_h2, nshots=nshots, dx=0.1)
fitting_h5 = apply_qubit_ham_kucj_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h5, circuit_cdag_h4, parameterinfo,theta_init=theta_init, nshots=nshots, dx=0.1)

#@show fitting_h4


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

p_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h5)
op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, her, circuit_cdag_h5)
op_re_ = op_re_re + im * op_re_im
op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, antiher, circuit_cdag_h5)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h5)
op_im_ = op_im_re + im * op_im_im
mom5 = op_re_ + op_im_ * im
#@show mom3
mom5 = mom5 * fitting_fop * fitting_h * fitting_h2 * fitting_h3 * fitting_h4 * fitting_h5

if comm !== nothing
    mom5 = MPI.Allreduce(mom5, MPI.SUM, comm)
    mom5 = mom5/MPI.Comm_size(comm)
end

@show mom5

mom5_exact_2=381.8400134363475
err_rel_mom5 =  abs(mom5 - mom5_exact_2)/abs(mom5_exact_2)



@show err_rel_mom5
println("hello world")



circuit_cdag_h6, parameterinfo = kucj(n_qubit, k=5, sparse=false, nx=5)
pinfo = QCMaterial.ParamInfo(parameterinfo)
theta_init = rand(pinfo.nparam)
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 

#nshots = 2^16
#nshots = 1000
#fitting_h3 = apply_qubit_ham_sampling_vqelike!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h3, circuit_cdag_h2, nshots=nshots, dx=0.1)
fitting_h6 = apply_qubit_ham_kucj_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h6, circuit_cdag_h5, parameterinfo,theta_init=theta_init, nshots=nshots, dx=0.1)

#@show fitting_h4


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

p_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h6)
op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, her, circuit_cdag_h6)
op_re_ = op_re_re + im * op_re_im
op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, antiher, circuit_cdag_h6)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h6d)
op_im_ = op_im_re + im * op_im_im
mom6 = op_re_ + op_im_ * im
#@show mom3
mom6 = mom6 * fitting_fop * fitting_h * fitting_h2 * fitting_h3 * fitting_h4 * fitting_h5* fitting_h6
if comm !== nothing
    mom6 = MPI.Allreduce(mom6, MPI.SUM, comm)
    mom6 = mom6/MPI.Comm_size(comm)
end
@show mom6

#mom5_exact_2=381.8400134363475
#err_rel_mom5 =  abs(mom5 - mom5_exact_2)/abs(mom5_exact_2)



#@show err_rel_mom5
@show mom6
println("hello world")




circuit_cdag_h7, parameterinfo = kucj(n_qubit, k=5, sparse=false, nx=5)
pinfo = QCMaterial.ParamInfo(parameterinfo)
theta_init = rand(pinfo.nparam)
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 
fitting_h7 = apply_qubit_ham_kucj_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h7, circuit_cdag_h6, parameterinfo,theta_init=theta_init, nshots=nshots, dx=0.1)


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

p_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h7)
op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, her, circuit_cdag_h7)
op_re_ = op_re_re + im * op_re_im
op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, antiher, circuit_cdag_h7)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h7)
op_im_ = op_im_re + im * op_im_im
mom7 = op_re_ + op_im_ * im
#@show mom3
mom7 = mom7 * fitting_fop * fitting_h * fitting_h2 * fitting_h3 * fitting_h4 * fitting_h5 * fitting_h6 * fitting_h7
if comm !== nothing
    mom7 = MPI.Allreduce(mom7, MPI.SUM, comm)
    mom7 = mom7/MPI.Comm_size(comm)
end
@show mom7


circuit_cdag_h8, parameterinfo = kucj(n_qubit, k=5, sparse=false, nx=5)
pinfo = QCMaterial.ParamInfo(parameterinfo)
theta_init = rand(pinfo.nparam)
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 
fitting_h8 = apply_qubit_ham_kucj_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h8, circuit_cdag_h7, parameterinfo,theta_init=theta_init, nshots=nshots, dx=0.1)


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

p_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h8)
op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, her, circuit_cdag_h8)
op_re_ = op_re_re + im * op_re_im
op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, antiher, circuit_cdag_h8)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h8)
op_im_ = op_im_re + im * op_im_im
mom8 = op_re_ + op_im_ * im
#@show mom3
mom8 = mom8 * fitting_fop * fitting_h * fitting_h2 * fitting_h3 * fitting_h4 * fitting_h5 * fitting_h6 * fitting_h7 * fitting_h8
if comm !== nothing
    mom8 = MPI.Allreduce(mom8, MPI.SUM, comm)
    mom8 = mom8/MPI.Comm_size(comm)
end
@show mom8


circuit_cdag_h9, parameterinfo = kucj(n_qubit, k=5, sparse=false, nx=5)
pinfo = QCMaterial.ParamInfo(parameterinfo)
theta_init = rand(pinfo.nparam)
state_augmented = QulacsQuantumState(n_qubit+1) 
set_computational_basis!(state_augmented, 0b00000) 
fitting_h9 = apply_qubit_ham_kucj_sampling!(jordan_wigner(ham_op_mom), state_augmented, circuit_cdag_h9, circuit_cdag_h8, parameterinfo,theta_init=theta_init, nshots=nshots, dx=0.1)


c1 = FermionOperator(" 1", 1.0) 
c1 = jordan_wigner(c1)
her, antiher = divide_real_imag(c1)
state_augmented = QulacsQuantumState(n_qubit+1)
set_computational_basis!(state_augmented, 0b00000)

p_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, her, circuit_cdag_h9)
op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, her, circuit_cdag_h9)
op_re_ = op_re_re + im * op_re_im
op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_gs, antiher, circuit_cdag_h9)
op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_gs, antiher, circuit_cdag_h9)
op_im_ = op_im_re + im * op_im_im
mom9 = op_re_ + op_im_ * im
#@show mom3
mom9 = mom9 * fitting_fop * fitting_h * fitting_h2 * fitting_h3 * fitting_h4 * fitting_h5 * fitting_h6 * fitting_h7 * fitting_h8 * fitting_h9
if comm !== nothing
    mom9 = MPI.Allreduce(mom9, MPI.SUM, comm)
    mom9 = mom9/MPI.Comm_size(comm)
end
@show mom9