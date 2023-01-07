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
U = 0.0
#μ = U/2
μ = 0.5
ε1 = [1.0, 1.0] 
V= 1.0
#ham = generate_ham_1d_hubbard(t, U, nsite, μ)
ham_op = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite)
#@show ham
#sparse_mat = get_sparse_operator(ham_op, n_qubit);
n_electron = 2
sparse_mat = get_number_preserving_sparse_operator(ham_op, n_qubit, n_electron);
EigVal_min = minimum(eigvals(sparse_mat.toarray()));
println(EigVal_min)
println("sparse_mat.toarray())=", size(sparse_mat.toarray()))
#computer ground state
#omega = 
#Imat = Matrix{Float64}(I, 2^n_qubit, 2^n_qubit)
#I = [1 0 0 0 0 0 ; 0 1 0 0 0 0; 0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0;0 0 0 0 0 1]
e = eigen(Hermitian(sparse_mat.toarray()))
weight = real.(conj.(e.vectors[1, :]) .* e.vectors[1, :])
println("sum(weight)=", sum(weight))

δ = 0.05
g(x, μ) = 1 / (x - μ)
aomega = zeros(Float64, length(omegas))
for ie in eachindex(e.values)
    aomega_reconst .+= g.(omegas, e.values[ie]) * weight[ie]
end
p = plot(xlim=(-wmax, wmax))
plot!(p, omegas, aomega.(omegas))
#G = (sparse_mat.toarray())
@show rank(G)
#println("G_inv=", inv(Imat - G))

