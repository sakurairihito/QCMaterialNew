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
#μ = 0.5
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


### cdag |gs> mom0###
c_gs = sparse_mat_c * vec_gs_exact2 
println(size(c_gs))
cdag_c_gs = sparse_mat_cdag * c_gs
#println(size(c_cdag_gs))
#println(size(vec_gs_exact2'))
#mom0_exact_2 = vec_gs_exact2' * c_cdag_gs
mom0_exact_2 = dot(vec_gs_exact2, cdag_c_gs)
println("mom0_exact_2=", mom0_exact_2) 

#### mom1 ham
ham_op_mom = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite) 
E0 = FermionOperator(" ", EigVal_min)
ham_op_mom = (ham_op_mom - E0)

### c^dag |gs> mom1###
#ham_op_mom1 = (ham_op_mom - identity)
sparse_mat_mom1 = get_sparse_operator(ham_op_mom, n_qubit)
#vec_gs_exact3 = eigvecs((sparse_mat2.toarray()))[:, 1]


c_gs = sparse_mat_c * vec_gs_exact2 
#println(size(cdag_gs))
h_c_gs = sparse_mat_mom1 * c_gs
cdag_h_c_gs = sparse_mat_cdag * h_c_gs
#println(size(c_cdag_gs))
#println(size(vec_gs_exact2'))
#mom0_exact_2 = vec_gs_exact2' * c_cdag_gs
mom1_exact_2 = dot(vec_gs_exact2, cdag_h_c_gs )
println("mom1_exact_2=", mom1_exact_2) 



### c^dag |gs> mom2###
#ham_op_mom1 = (ham_op_mom - identity)
sparse_mat_mom2 = get_sparse_operator(ham_op_mom^2, n_qubit)
#vec_gs_exact3 = eigvecs((sparse_mat2.toarray()))[:, 1]


c_gs = sparse_mat_c * vec_gs_exact2 
#println(size(cdag_gs))
h_c_gs = sparse_mat_mom2 * c_gs
cdag_h_c_gs = sparse_mat_cdag * h_c_gs
#println(size(c_cdag_gs))
#println(size(vec_gs_exact2'))
#mom0_exact_2 = vec_gs_exact2' * c_cdag_gs
mom2_exact_2 = dot(vec_gs_exact2, cdag_h_c_gs )
println("mom2_exact_2=", mom2_exact_2)


### c^dag |gs> mom3###
#ham_op_mom1 = (ham_op_mom - identity)
sparse_mat_mom3 = get_sparse_operator(ham_op_mom^3, n_qubit)
#vec_gs_exact3 = eigvecs((sparse_mat2.toarray()))[:, 1]


c_gs = sparse_mat_c * vec_gs_exact2 
#println(size(cdag_gs))
h_c_gs = sparse_mat_mom3 * c_gs
cdag_h_c_gs = sparse_mat_cdag * h_c_gs
#println(size(c_cdag_gs))
#println(size(vec_gs_exact2'))
#mom0_exact_2 = vec_gs_exact2' * c_cdag_gs
mom3_exact_2 = dot(vec_gs_exact2, cdag_h_c_gs )
println("mom3_exact_2=", mom3_exact_2)

### c^dag |gs> mom4###
#ham_op_mom1 = (ham_op_mom - identity)
sparse_mat_mom4 = get_sparse_operator(ham_op_mom^4, n_qubit)
#vec_gs_exact3 = eigvecs((sparse_mat2.toarray()))[:, 1]


c_gs = sparse_mat_c * vec_gs_exact2 
#println(size(cdag_gs))
h_c_gs = sparse_mat_mom4 * c_gs
cdag_h_c_gs = sparse_mat_cdag * h_c_gs
#println(size(c_cdag_gs))
#println(size(vec_gs_exact2'))
#mom0_exact_2 = vec_gs_exact2' * c_cdag_gs
mom4_exact_2 = dot(vec_gs_exact2, cdag_h_c_gs )
println("mom4_exact_2=", mom4_exact_2)




### c^dag |gs> mom5###
#ham_op_mom1 = (ham_op_mom - identity)
sparse_mat_mom5 = get_sparse_operator(ham_op_mom^5, n_qubit)
#vec_gs_exact3 = eigvecs((sparse_mat2.toarray()))[:, 1]


c_gs = sparse_mat_c * vec_gs_exact2 
#println(size(cdag_gs))
h_c_gs = sparse_mat_mom5 * c_gs
cdag_h_c_gs = sparse_mat_cdag * h_c_gs
#println(size(c_cdag_gs))
#println(size(vec_gs_exact2'))
#mom0_exact_2 = vec_gs_exact2' * c_cdag_gs
mom5_exact_2= dot(vec_gs_exact2, cdag_h_c_gs )
println("mom5_exact_2=", mom5_exact_2) 