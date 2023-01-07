using QCMaterial
using PyCall
using LinearAlgebra
using HDF5
import Random
import PyCall: pyimport

##### 0次のモーメント
Random.seed!(100)
nsite = 2
n_qubit = 2*nsite 
U = 1.0
μ = U/2
ε1 = [1.0, 1.0] 
V= 1.0
#ham = generate_ham_1d_hubbard(t, U, nsite, μ)
ham_op = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite)
#@show ham
n_electron = 2
sparse_mat = get_number_preserving_sparse_operator(ham_op, n_qubit, n_electron);
EigVal_min = minimum(eigvals(sparse_mat.toarray()));
println("sparse_mat.toarray())=", size(sparse_mat.toarray()))

# computer ground state
state_gs_exact = QulacsQuantumState(n_qubit)
sparse_mat2 = get_sparse_operator(ham_op, n_qubit)
vec_gs_exact2 = eigvecs((sparse_mat2.toarray()))[:, 1]
#e_gs_exact = eigvals((sparse_mat.toarray()))
#println("e_gs_exact=", eigvals(sparse_mat2.toarray()))
println("e_gs_exact_numparticles_conserved=", eigvecs((sparse_mat.toarray()))[:, 1])
println("e_gs_exact=", vec_gs_exact2)
state_load!(state_gs_exact, vec_gs_exact2) #exact_ground_state
println("state_gs_exact_after_load=", get_vector(state_gs_exact))
println("state_load QK")
ham = jordan_wigner(ham_op)

#@show op
# perform VQE for generating reference state (state)
state0 = create_hf_state(n_qubit, n_electron)
circuit = uccgsd(n_qubit, orbital_rot=true)
theta_init = rand(num_theta(circuit))


### test_eigen ###
F = eigen(sparse_mat2.toarray())
println("eigen_=", reverse(F.vectors[:, 1]))

### test_c


#### test
op_test_cdag  = FermionOperator("1^")
sparse_mat_cdag = get_sparse_operator(op_test_cdag, n_qubit)
#println("cdag_mat=", sparse_mat_op.toarray())
op_test_c  = FermionOperator("1")
sparse_mat_c = get_sparse_operator(op_test_c, n_qubit)
####


### c^dag |gs> mom0###
cdag_gs = sparse_mat_cdag * vec_gs_exact2 
println(size(cdag_gs))
c_cdag_gs = sparse_mat_c * cdag_gs
println(size(c_cdag_gs))
println(size(vec_gs_exact2'))
#mom0_exact_2 = vec_gs_exact2' * c_cdag_gs
mom0_exact_2 = dot(vec_gs_exact2, c_cdag_gs )
println("mom0_exact_2=", mom0_exact_2) 

#### mom1 ham
ham_op_mom = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite) 
identity = FermionOperator(" ", EigVal_min)
ham_op_mom = (ham_op_mom - identity)



### c^dag |gs> mom1###
#ham_op_mom1 = (ham_op_mom - identity)
sparse_mat_mom1 = get_sparse_operator(ham_op_mom, n_qubit)
#vec_gs_exact3 = eigvecs((sparse_mat2.toarray()))[:, 1]


cdag_gs = sparse_mat_cdag * vec_gs_exact2 
println(size(cdag_gs))
h_cdag_gs = sparse_mat_mom1 * cdag_gs
c_h_cdag_gs = sparse_mat_c * h_cdag_gs
println(size(c_cdag_gs))
println(size(vec_gs_exact2'))
#mom0_exact_2 = vec_gs_exact2' * c_cdag_gs
mom1_exact_2 = dot(vec_gs_exact2, c_h_cdag_gs )
println("mom0_exact_2=", mom1_exact_2) 




# VQE
cost_history, thetas_opt = 
   QCMaterial.solve_gs(ham, circuit, state0, theta_init=theta_init, verbose=true,
       comm=QCMaterial.MPI_COMM_WORLD
   )


update_circuit_param!(circuit, thetas_opt) #set opt params
update_quantum_state!(circuit, state0) # approx ground state
vec_gs = get_vector(state0)
println("vec_gs_vqe=", vec_gs)

expec_val_g = get_expectation_value(ham, state0)
@show expec_val_g
println("dif_ground_energy=", abs(expec_val_g - EigVal_min))
@assert isapprox(expec_val_g,  EigVal_min , rtol=1e-3) # OK

# Apply c^dagger_1 to |Psi>, which will yield |10>.
op = jordan_wigner(FermionOperator("1^"))
# Prepare |phi> = |0011>
n_electron_incremented = 3
state0_bra = create_hf_state(n_qubit, n_electron_incremented)

# Fit <1100|U(theta)^dagger c_2^dagger |GS>
circuit_bra = uccgsd(n_qubit, orbital_rot=true)
update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
#update_circuit_param!(circuit_bra, rand(size(circuit_bra.theta_offset)[1])) 
verbose = true
maxiter = 300
gtol = 1e-8
squared_norm = apply_qubit_op!(op, state0, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
    options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
    verbose=verbose))
println("squared_norm=",squared_norm)

#ここまででc^dag |GS> の近似的状態をcircuit_bra |state0_bra>で表現した。
update_quantum_state!(circuit_bra, state0_bra)
println("fitting_c_dag_gs_VQE=",squared_norm * get_vector(state0_bra))
#c_1に関する基底状態とフィッティングした状態の遷移確率を計算する。
divide_real_imag(op::QubitOperator) = (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im

op2 = jordan_wigner(FermionOperator("1"))
op_re, op_im = divide_real_imag(op2)
op_re = get_transition_amplitude(op_re, state0, state0_bra)
op_im = get_transition_amplitude(op_im, state0, state0_bra)
#res = get_transition_amplitude(op2, state0, state0_bra)
#println("res=", res)
res = op_re + im * op_im
res = res * squared_norm
println("res*coeff=",res)

# 厳密な計算をする。つまり、フィッティングによって得られた値と比較をする。
# <GS| c_1 c^{dag}_1 |GS>
op_mom0 = FermionOperator()
op_mom0 += FermionOperator(" 1  1^", 1.0) #c_1 c^dag_1
op_mom0_re, op_mom0_im = divide_real_imag(jordan_wigner(op_mom0))
#op_mom0_exact_re = get_transition_amplitude(op_mom0_re, state0, state0)
#op_mom0_exact_im = get_transition_amplitude(op_mom0_im, state0, state0)
op_mom0_exact_re = get_transition_amplitude(op_mom0_re, state_gs_exact, state_gs_exact)
op_mom0_exact_im = get_transition_amplitude(op_mom0_im, state_gs_exact, state_gs_exact)
exact_mom0 = op_mom0_exact_re + im * op_mom0_exact_im  
#exact_mom0 = get_expectation_value(jordan_wigner(op_), state0)
println("exact_mom0=", exact_mom0)

#println("dif_mom0=", abs(res - exact_mom0)) # 3.4416913763379853e-15 
println("dif_mom0=", abs(res - mom0_exact_2)) 

@assert isapprox(res, mom0_exact_2, rtol=1e-3)

## state_vector
vec_cdag_gs = get_vector(state0_bra)
println("vec_mom0=", squared_norm .* vec_cdag_gs)

##  1次のモーメント ##
# <GS| c_1 c^{dag}_1 |GS>
# H-E0の定義をする。
ham_op_mom_exact = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite) 
identity_exact = FermionOperator(" ", EigVal_min)
ham_op_mom_exact = ham_op_mom_exact - identity_exact
ham_op_mom1_exact = ham_op_mom_exact^1

## exact moments at m=1
op_c = FermionOperator(" 1", 1.0) #c_1
op_c = op_c * ham_op_mom1_exact #c H
op_c = op_c * FermionOperator(" 1^", 1.0) #c H c^dagger
op_c_re, op_c_im = divide_real_imag(jordan_wigner(op_c))
op_c_exact_re = get_transition_amplitude(op_c_re, state0, state0)
op_c_exact_im = get_transition_amplitude(op_c_im, state0, state0)
exact_mom1 = op_c_exact_re + im * op_c_exact_im  
println("exact_mom1=", exact_mom1)

n_electron_incremented = 3
state1_bra = create_hf_state(n_qubit, n_electron_incremented)
update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))

# (H-E0)^1
ham_op_mom = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite) 
identity = FermionOperator(" ", expec_val_g)
ham_op_mom = (ham_op_mom - identity)
#println("ham_op_mom^1=", ham_op_mom3)
ham_op_mom1 = ham_op_mom^1
#println("ham_op_mom^$m=", ham_op_mom3)
#println("ham_op_mom^$m=", ham_op_mom)
verbose = true
maxiter = 500
gtol = 1e-8
squared_norm1 = apply_qubit_ham!(jordan_wigner(ham_op_mom1), state0_bra, circuit_bra, state1_bra, minimizer=QCMaterial.mk_scipy_minimize(
    options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
    verbose=verbose))
println("squared_norm1=", squared_norm1)

update_quantum_state!(circuit_bra, state1_bra)
#c_1に関する基底状態とフィッティングした状態の遷移確率を計算する。
#divide_real_imag(op::QubitOperator) = (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im
op = jordan_wigner(FermionOperator("1"))
op_re, op_im = divide_real_imag(op)
op_re1 = get_transition_amplitude(op_re, state0, state1_bra)
op_im1 = get_transition_amplitude(op_im, state0, state1_bra)
mom1 = op_re1 + im * op_im1 

#coeeffを考慮する。
mom1 = mom1 * squared_norm1 * squared_norm
println("squared_norm=", squared_norm)
println("squared_norm1=", squared_norm1)
println("mom1_1の計算=", mom1)
println("mom1_exact=", exact_mom1)
println("dif_mom1=", abs(mom1 - exact_mom1))
#@assert isapprox(mom1, exact_mom1, rtol=1e-3) 

@assert isapprox(mom1, mom1_exact_2, rtol=1e-3) 