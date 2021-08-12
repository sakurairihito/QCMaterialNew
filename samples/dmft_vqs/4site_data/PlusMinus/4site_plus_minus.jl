using QCMaterial
using PyCall
using LinearAlgebra
using HDF5
import Random
import PyCall: pyimport


#make taus list file.h5
#length=8
function taus_list(file_name)
    x = Float64[]
    for i in 1:10
        tau = 0.005*(i-1) + exp((0.1*(i-1))) -1
        push!(x,tau)
    end
    fid = h5open(file_name,"w")
    fid["/test/data"] = x
    close(fid)	
end

#read taus list file.h5
function read_taus_list(file_name)
    fid = h5open(file_name,"r")
    x = fid["/test/data"][:]
    close(fid)
    return x
end

#read taus list file.txt
function read_and_parse_float(file_name, n)
    x = zeros(Float64, n)
    open(file_name, "r") do fp
        num_elm = parse(Int64,readline(fp))
        @assert n == num_elm
        for i in 1:n
            x[i] = parse(Float64, readline(fp))
        end
    end
    return x
end


function generate_impurity_ham_with_1imp_3bath_dmft(U::Float64, μ::Float64,  nsite::Integer)
    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)

    ham_op1 = FermionOperator()
    ham_op1 += FermionOperator("$(up_index(1))^ $(down_index(1))^ $(up_index(1)) $(down_index(1))", -U)

    #chemical potential
    for ispin in [1, 2]
        ham_op1 += FermionOperator("$(so_idx(1, ispin))^ $(so_idx(1, ispin))", -μ)
    end

    #bath energy level
    ε = [0.0, 1.11919, 0.00000, -1.11919]
    for ispin in [1, 2]
        for i in 2:nsite
            ham_op1 += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(i, ispin))", ε[i])
        end
    end

    #hybridization
    V = [0.0, -1.26264, 0.07702, -1.26264]

    for ispin in [1, 2]
        for i in 2:nsite
            ham_op1 += FermionOperator("$(so_idx(1, ispin))^ $(so_idx(i, ispin))", V[i])
            ham_op1 += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(1, ispin))", V[i])
        end
    end
    ham_op1
end

function hev(n_qubit, depth)
    circuit = QulacsParametricQuantumCircuit(n_qubit)
    for d in 1:depth
        for i in 1:n_qubit
            add_parametric_RY_gate!(circuit, i, 0.0)
            add_parametric_RZ_gate!(circuit, i, 0.0)
        end
        for i in 1:n_qubit÷2
            add_CNOT_gate!(circuit, 2*i-1, 2*i)
        end
        for i in 1:n_qubit÷2-1
            add_CNOT_gate!(circuit, 2*i, 2*i+1)
        end
    end
    for i in 1:n_qubit
        add_parametric_RY_gate!(circuit, i, 0.0)
        add_parametric_RZ_gate!(circuit, i, 0.0)
    end
    circuit = QulacsVariationalQuantumCircuit(circuit)
    circuit
end



#beta = 1000 (T=0.001)
nsite = 4
n_qubit　= 2*nsite
U = 4.0
μ = U/2
d_theta = 1e-5
verbose = QCMaterial.MPI_rank == 0
Random.seed!(100)
 
#Hamiltonian
ham_op1 = generate_impurity_ham_with_1imp_3bath_dmft(U, μ, nsite)
n_electron_gs = 4
@assert mod(n_electron_gs, 2) == 0
sparse_mat = get_number_preserving_sparse_operator(ham_op1, n_qubit, n_electron_gs);
enes_ed = eigvals(sparse_mat.toarray());

#debug
println("Ground energy_ED=",minimum(enes_ed))

#ansatz
state0 = create_hf_state(n_qubit, n_electron_gs)
vc = uccgsd(n_qubit, nocc = 2, orbital_rot=true, uccgsd = true, p_uccgsd = false)
#depth = n_qubit*2
#vc = hev(n_qubit, depth)
theta_init = rand(num_theta(vc))

#Perform VQE
cost_history, thetas_opt = 
QCMaterial.solve_gs(jordan_wigner(ham_op1), vc, state0, theta_init=theta_init, verbose=true,
    comm=QCMaterial.MPI_COMM_WORLD
)

#debug
println("Ground energy_VQE=",cost_history[end])

#c^{dag},c
up1 = up_index(1)
down1 = down_index(1)
up2 = up_index(2)
down2 = down_index(2)


if ARGS[1] == "plus_true"
    right_op = FermionOperator("$(up1)^ ", 1.0)
    right_op = jordan_wigner(right_op)
    left_op = FermionOperator("$(up1) ", 1.0)
    left_op = jordan_wigner(left_op)

    n_electron_ex = 5

    sign = 1
end


if ARGS[2] == "minus_true"
    right_op = FermionOperator("$(up1) ", 1.0)
    right_op = jordan_wigner(right_op)
    left_op = FermionOperator("$(up1)^ ", 1.0)
    left_op = jordan_wigner(left_op)

    n_electron_ex = 3

    sign = -1
end


vc_ex = uccgsd(n_qubit, nocc = 2, orbital_rot=true, uccgsd = true, p_uccgsd = false)
#vc_ex = hev(n_qubit, depth)
#println(vc_ex)
state_gs = create_hf_state(n_qubit, n_electron_gs)
update_circuit_param!(vc, thetas_opt)
update_quantum_state!(vc, state_gs)

state0_ex = create_hf_state(n_qubit, n_electron_ex)

#taus = collect(range(0.0, 0.02, length = 2))

#make taus list file
#taus_list("dimer_plus.h5")
#generate taus list file
#taus = read_taus_list("dimer_plus.h5")
#println("taus=",taus)

num_taus = 139
taus = read_and_parse_float("sp_tau_plus2_p_139.txt", num_taus)
println("taus=",taus)

Gfunc_ij_list = sign * compute_gtau(
    jordan_wigner(ham_op1),
    left_op,
    right_op,
    vc_ex,
    state_gs,
    state0_ex,
    taus,
    d_theta,
    verbose = verbose,
)
println("Gfunc_ij_list_plus=", Gfunc_ij_list)



function write_to_txt(file_name, x, y)
    open(file_name, "w") do fp
        for i = 1:length(x)
            println(fp, x[i], " ", real(y[i]))
        end
    end
end

write_to_txt("gf_4site_plus_hev_p139.txt", taus, Gfunc_ij_list)
println("done!")