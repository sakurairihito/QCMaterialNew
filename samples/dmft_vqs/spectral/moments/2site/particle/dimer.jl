using QCMaterial
using PyCall
using LinearAlgebra
using HDF5
import Random
import PyCall: pyimport

##### 0次のモーメント
Random.seed!(100)
nsite = 2
n_qubit = 2 * nsite 
U = 1.0
μ = U/2
#μ = 0.5
ε1 = [1.0, 1.0] 
V= 1.0
#ham = generate_ham_1d_hubbard(t, U, nsite, μ)
ham_op = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite)
#@show ham
n_electron = 2
sparse_mat = get_number_preserving_sparse_operator(ham_op, n_qubit, n_electron);
EigVal_min = minimum(eigvals(sparse_mat.toarray()));
#println("sparse_mat.toarray())=", size(sparse_mat.toarray()))
#computer ground state
#state_gs_exact = QulacsQuantumState(n_qubit)
#vec_gs_exact = eigvecs((sparse_mat.toarray()))[:, 1]
#e_gs_exact = eigvals((sparse_mat.toarray()))
#println("e_gs_exact=", e_gs_exact)
# @show state
# vec = [0, 1, 2, 3]
#println(vec_gs_exact)
#state_load!(state_gs_exact, vec_gs_exact) #exact_ground_state
#println("state_load QK")
ham = jordan_wigner(ham_op)
#@show op
# perform VQE for generating reference state (state)
state0 = create_hf_state(n_qubit, n_electron)
circuit = uccgsd(n_qubit, orbital_rot=true)
theta_init = rand(num_theta(circuit))

# VQE
cost_history, thetas_opt = 
   QCMaterial.solve_gs(ham, circuit, state0, theta_init=theta_init, verbose=true,
       comm=QCMaterial.MPI_COMM_WORLD
   )
    
update_circuit_param!(circuit, thetas_opt) #set opt params
update_quantum_state!(circuit, state0) # approx ground state
#vec_gs = get_vector(state0)
#println("vec_gs=", vec_gs)

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

#c_1に関する基底状態とフィッティングした状態の遷移確率を計算する。
divide_real_imag(op::QubitOperator) = (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im

op2 = jordan_wigner(FermionOperator("1"))
op_re, op_im = divide_real_imag(op2)
op_re = get_transition_amplitude(op_re, state0, state0_bra)
op_im = get_transition_amplitude(op_im, state0, state0_bra)
#res = get_transition_amplitude(op2, state0, state0_bra)
#println("res=", res)
res = op_re + im * op_im
@show res
res = res * squared_norm
println("res*coeff=",res)

# 厳密な計算をする。つまり、フィッティングによって得られた値と比較をする。
# <GS| c_1 c^{dag}_1 |GS>
op_mom0 = FermionOperator()
op_mom0 += FermionOperator(" 1  1^", 1.0) #c_1 c^dag_1
op_mom0_re, op_mom0_im = divide_real_imag(jordan_wigner(op_mom0))
#op_mom0_exact_re = get_transition_amplitude(op_mom0_re, state0, state0)
#op_mom0_exact_im = get_transition_amplitude(op_mom0_im, state0, state0)
op_mom0_exact_re = get_transition_amplitude(op_mom0_re, state0, state0)
op_mom0_exact_im = get_transition_amplitude(op_mom0_im, state0, state0)
exact_mom0 = op_mom0_exact_re + im * op_mom0_exact_im  
#exact_mom0 = get_expectation_value(jordan_wigner(op_), state0)
println("exact_mom0=", exact_mom0)

println("dif_mom0=", abs(res - exact_mom0)) # 3.4416913763379853e-15 
@assert isapprox(res, exact_mom0, rtol=1e-3)

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
@assert isapprox(mom1, exact_mom1, rtol=1e-3) 

##  2次のモーメント ##
# (H-E0)^2の定義をする。
ham_op_mom2_exact = ham_op_mom_exact^2

## exact moments at m=2
op_c = FermionOperator(" 1", 1.0) #c_1
op_c = op_c * ham_op_mom2_exact #c H
op_c = op_c * FermionOperator(" 1^", 1.0) #c H c^dagger
op_c_re, op_c_im = divide_real_imag(jordan_wigner(op_c))
op_c_exact_re = get_transition_amplitude(op_c_re, state0, state0)
op_c_exact_im = get_transition_amplitude(op_c_im, state0, state0)
exact_mom2 = op_c_exact_re + im * op_c_exact_im  
println("exact_mom2=", exact_mom2)

n_electron_incremented = 3
state2_bra = create_hf_state(n_qubit, n_electron_incremented)
update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))

# (H-E0)^2
#ham_op_mom2 = ham_op_mom^2

verbose = true
maxiter = 500
gtol = 1e-8
squared_norm2 = apply_qubit_ham!(jordan_wigner(ham_op_mom1), state1_bra, circuit_bra, state2_bra, minimizer=QCMaterial.mk_scipy_minimize(
    options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
    verbose=verbose))
println("squared_norm2=", squared_norm2)

update_quantum_state!(circuit_bra, state2_bra)
#c_1に関する基底状態とフィッティングした状態の遷移確率を計算する。
#divide_real_imag(op::QubitOperator) = (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im
op = jordan_wigner(FermionOperator("1"))
op_re, op_im = divide_real_imag(op)
op_re2 = get_transition_amplitude(op_re, state0, state2_bra)
op_im2 = get_transition_amplitude(op_im, state0, state2_bra)
mom2 = op_re2 + im * op_im2 

#coeeffを考慮する。
mom2 = mom2 * squared_norm2 *squared_norm1 * squared_norm
println("squared_norm=", squared_norm)
println("squared_norm1=", squared_norm1)
println("mom2の計算=", mom2)
println("mom2_exact=", exact_mom2)
println("dif_mom1=", abs(mom2 - exact_mom2))
@assert isapprox(mom2, exact_mom2, rtol=1e-3) 



##  3次のモーメント ##
# (H-E0)^3の定義をする。
ham_op_mom3_exact = ham_op_mom_exact^3

## exact moments at m=2
op_c = FermionOperator(" 1", 1.0) #c_1
op_c = op_c * ham_op_mom3_exact #c H
op_c = op_c * FermionOperator(" 1^", 1.0) #c H c^dagger
op_c_re, op_c_im = divide_real_imag(jordan_wigner(op_c))
op_c_exact_re = get_transition_amplitude(op_c_re, state0, state0)
op_c_exact_im = get_transition_amplitude(op_c_im, state0, state0)
exact_mom3 = op_c_exact_re + im * op_c_exact_im  
println("exact_mom3=", exact_mom3)

n_electron_incremented = 3
state3_bra = create_hf_state(n_qubit, n_electron_incremented)
update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))

# (H-E0)^2
#ham_op_mom2 = ham_op_mom^2

verbose = true
maxiter = 500
gtol = 1e-8
squared_norm3 = apply_qubit_ham!(jordan_wigner(ham_op_mom1), state2_bra, circuit_bra, state3_bra, minimizer=QCMaterial.mk_scipy_minimize(
    options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
    verbose=verbose))
println("squared_norm3=", squared_norm3)

update_quantum_state!(circuit_bra, state3_bra)
#c_1に関する基底状態とフィッティングした状態の遷移確率を計算する。
#divide_real_imag(op::QubitOperator) = (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im
op = jordan_wigner(FermionOperator("1"))
op_re, op_im = divide_real_imag(op)
op_re3 = get_transition_amplitude(op_re, state0, state3_bra)
op_im3 = get_transition_amplitude(op_im, state0, state3_bra)
mom3 = op_re3 + im * op_im3 

#coeeffを考慮する。
mom3 = mom3 * squared_norm3 * squared_norm2 *squared_norm1 * squared_norm

println("mom3の計算=", mom3)
println("mom3_exact=", exact_mom3)
println("dif_mom3=", abs(mom3 - exact_mom3))
@assert isapprox(mom3, exact_mom3, rtol=1e-3) 

##  4次のモーメント ##

# (H-E0)^4の定義をする。
ham_op_mom4_exact = ham_op_mom_exact^4

## exact moments at m=2
op_c = FermionOperator(" 1", 1.0) #c_1
op_c = op_c * ham_op_mom4_exact #c H
op_c = op_c * FermionOperator(" 1^", 1.0) #c H c^dagger
op_c_re, op_c_im = divide_real_imag(jordan_wigner(op_c))
op_c_exact_re = get_transition_amplitude(op_c_re, state0, state0)
op_c_exact_im = get_transition_amplitude(op_c_im, state0, state0)
exact_mom4 = op_c_exact_re + im * op_c_exact_im  
println("exact_mom4=", exact_mom4)

n_electron_incremented = 3
state4_bra = create_hf_state(n_qubit, n_electron_incremented)
update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))

# (H-E0)^4
#ham_op_mom4 = ham_op_mom

verbose = true
maxiter = 500
gtol = 1e-8
squared_norm4 = apply_qubit_ham!(jordan_wigner(ham_op_mom1), state3_bra, circuit_bra, state4_bra, minimizer=QCMaterial.mk_scipy_minimize(
    options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
    verbose=verbose))
println("squared_norm4=", squared_norm4)

update_quantum_state!(circuit_bra, state4_bra)
#c_1に関する基底状態とフィッティングした状態の遷移確率を計算する。
#divide_real_imag(op::QubitOperator) = (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im
op = jordan_wigner(FermionOperator("1"))
op_re, op_im = divide_real_imag(op)
op_re4 = get_transition_amplitude(op_re, state0, state4_bra)
op_im4 = get_transition_amplitude(op_im, state0, state4_bra)
mom4 = op_re4 + im * op_im4

#coeeffを考慮する。
mom4 = mom4 * squared_norm4 * squared_norm3 * squared_norm2 *squared_norm1 * squared_norm

println("mom4の計算=", mom4)
println("mom3_exact=", exact_mom4)
println("dif_mom3=", abs(mom4 - exact_mom4))
@assert isapprox(mom4, exact_mom4, rtol=1e-3)


##  5次のモーメント ##

# (H-E0)^5の定義をする。
ham_op_mom5_exact = ham_op_mom_exact^5

## exact moments at m=2
op_c = FermionOperator(" 1", 1.0) #c_1
op_c = op_c * ham_op_mom5_exact #c H
op_c = op_c * FermionOperator(" 1^", 1.0) #c H c^dagger
op_c_re, op_c_im = divide_real_imag(jordan_wigner(op_c))
op_c_exact_re = get_transition_amplitude(op_c_re, state0, state0)
op_c_exact_im = get_transition_amplitude(op_c_im, state0, state0)
exact_mom5 = op_c_exact_re + im * op_c_exact_im  
println("exact_mom4=", exact_mom5)

n_electron_incremented = 3
state5_bra = create_hf_state(n_qubit, n_electron_incremented)
update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))

# (H-E0)^4
#ham_op_mom4 = ham_op_mom

verbose = true
maxiter = 500
gtol = 1e-8
squared_norm5 = apply_qubit_ham!(jordan_wigner(ham_op_mom1), state4_bra, circuit_bra, state5_bra, minimizer=QCMaterial.mk_scipy_minimize(
    options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
    verbose=verbose))
println("squared_norm5=", squared_norm5)

update_quantum_state!(circuit_bra, state5_bra)
#c_1に関する基底状態とフィッティングした状態の遷移確率を計算する。
#divide_real_imag(op::QubitOperator) = (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im
op = jordan_wigner(FermionOperator("1"))
op_re, op_im = divide_real_imag(op)
op_re5 = get_transition_amplitude(op_re, state0, state5_bra)
op_im5 = get_transition_amplitude(op_im, state0, state5_bra)
mom5 = op_re5 + im * op_im5

#coeeffを考慮する。
mom5 = mom5 * squared_norm5 * squared_norm4 * squared_norm3 * squared_norm2 *squared_norm1 * squared_norm

println("mom5の計算=", mom5)
println("mom5_exact=", exact_mom5)
println("dif_mom5=", abs(mom5 - exact_mom5))
@assert isapprox(mom5, exact_mom5, rtol=1e-3)

println("U=$(U),vqe")
println("mom0=",res)
println("mom1_1の計算=", mom1)
println("mom2の計算=", mom2)
println("mom3の計算=", mom3)
println("mom4の計算=", mom4)
println("mom5の計算=", mom5)