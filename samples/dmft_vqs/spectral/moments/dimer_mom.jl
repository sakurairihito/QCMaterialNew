using QCMaterial
using PyCall
using LinearAlgebra
using HDF5
import Random
import PyCall: pyimport


##### 0次のモーメント

nsite = 2 
n_qubit = 2*nsite 
U = 0.0
μ = 1.0
ε1 = [1.0,-1.0,1.0] 
V= 1.0
#ham = generate_ham_1d_hubbard(t, U, nsite, μ)
ham_op = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite)
#@show ham
n_electron = 2
sparse_mat = get_number_preserving_sparse_operator(ham_op, n_qubit, n_electron);
EigVal_min = minimum(eigvals(sparse_mat.toarray()));
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
expec_val_g = get_expectation_value(ham, state0)
@show expec_val_g
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
res = res * squared_norm
println("res*coeff=",res)

# 厳密な計算をする。つまり、フィッティングによって得られた値と比較をする。
# <GS| c_1 c^{dag}_1 |GS>
op_ = FermionOperator()
op_ += FermionOperator(" 1  1^", 1.0) #c_1 c^dag_1
exact_op_ = get_expectation_value(jordan_wigner(op_), state0)

println("dif=", abs(res - exact_op_)) # 3.4416913763379853e-15 
@assert isapprox(res, exact_op_, rtol=1e-3)





##  1次のモーメント
mom1_2 = res * expec_val_g #1次のモーメントの第二項

# <GS| c_1 c^{dag}_1 |GS>
op_ = FermionOperator()
op_ += FermionOperator(" 1^", 1.0) #c_1 c^dag_1
op_ *= ham_op
op_ *= FermionOperator(" 1", 1.0) 
exact_mom1_1 = get_expectation_value(jordan_wigner(op_), state0)
println("excat_mom1_1=", exact_mom1_1) 
exact_mom1_2 = exact_op_ * EigVal_min
exact_mom1 = exact_mom1_1 - exact_mom1_2

## mom1_1の計算
#state0_braが coeff * c^dag |GS>
#coeff = squared_norm 
n_electron_incremented = 3
state1_bra = create_hf_state(n_qubit, n_electron_incremented)
update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
# Fit <1100|U(theta)^dagger c_2^dagger |GS>
#circuit_bra = uccgsd(n_qubit, orbital_rot=true)
#update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
#update_circuit_param!(circuit_bra, rand(size(circuit_bra.theta_offset)[1])) 
#ham_re, ham_im = divide_real_imag(ham)
#println("ham_re_analytics=", ham_re) # hemitian
#println("ham_im_analytics=", ham_im) # 0
verbose = true
maxiter = 300
gtol = 1e-16
squared_norm2 = apply_qubit_ham!(ham, state0_bra, circuit_bra, state1_bra, minimizer=QCMaterial.mk_scipy_minimize(
    options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
    verbose=verbose))
println("squared_norm2=",squared_norm2)

update_quantum_state!(circuit_bra, state1_bra)

#c_1に関する基底状態とフィッティングした状態の遷移確率を計算する。
#divide_real_imag(op::QubitOperator) = (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im
op2 = jordan_wigner(FermionOperator("1"))
op_re1, op_im1 = divide_real_imag(op2)
op_re1 = get_transition_amplitude(op_re1, state0, state1_bra)
op_im1 = get_transition_amplitude(op_im1, state0, state1_bra)
#res = get_transition_amplitude(op2, state0, state0_bra)
#println("res=", res)
mom1_1 = op_re1 + im * op_im1
#coeeffを考慮する。
mom1_1 = mom1_1 * squared_norm2 * squared_norm
mom1 = mom1_1 - mom1_2

println("mom1_1=", mom1_1)
println("exact_mom1_1=", exact_mom1_1)
println("dif_mom1_2=", abs(mom1_2 - exact_mom1_2))
@assert isapprox(mom1_2, exact_mom1_2, rtol=1e-3) 
println("dif_mom1_1=", abs(mom1_1 - exact_mom1_1))
@assert isapprox(mom1_1, exact_mom1_1, rtol=1e-3) 
println("dif_mom1=", abs(mom1 - exact_mom1))
@assert isapprox(mom1, exact_mom1, rtol=1e-3) 