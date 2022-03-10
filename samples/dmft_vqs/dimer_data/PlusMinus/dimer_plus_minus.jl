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
function read_and_parse_float(file_name)
    x = zeros(Float64, 0)
    open(file_name, "r") do fp
        num_elm = parse(Int64,readline(fp))
        for i in 1:num_elm
            push!(x, parse(Float64, readline(fp)))
        end
    end
    return x
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


nsite = 2
n_qubit = 2 * nsite
U = 1.0
V = 1.0
μ = U/2
ε = 1.0
d_theta = 1e-5
verbose = QCMaterial.MPI_rank == 0
Random.seed!(100)


#Hamiltonian
ham_op = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite)

n_electron_gs = 2
@assert mod(n_electron_gs, 2) == 0
sparse_mat = get_number_preserving_sparse_operator(ham_op, n_qubit, n_electron_gs);
enes_ed = eigvals(sparse_mat.toarray());
println("Ground_enegry_ED=", minimum(enes_ed))
#N+1系のground energy
#n_electron_ex = 3
#sparse_mat_ex = get_number_preserving_sparse_operator(ham_op, n_qubit, n_electron_ex);
#enes_ex_ed = eigvals(sparse_mat_ex.toarray());

#debug
#N = particle number
#println("Ground energy_2_ED=",minimum(enes_ed))
#println("Ground_enegry_3_ED=", minimum(enes_ex_ed))


#1 system
#n_electron_1 = 1
#sparse_mat_enum_1 = get_number_preserving_sparse_operator(ham_op, n_qubit, n_electron_1);
#enes_ed_enum_1 = eigvals(sparse_mat_enum_1.toarray());
#debug
#println("Ground energy_ED=",minimum(enes_ed_enum_1))
#println("Ground energy_1_ED=",minimum(enes_ed_enum_1))

#1 system
#n_electron_4 = 4
#sparse_mat_enum_4 = get_number_preserving_sparse_operator(ham_op, n_qubit, n_electron_4);
#enes_ed_enum_4 = eigvals(sparse_mat_enum_4.toarray());
#debug
#println("Ground energy_ED=",minimum(enes_ed_enum_1))
#println("Ground energy_4_ED=",minimum(enes_ed_enum_4))



#ansatz
state0 = create_hf_state(n_qubit, n_electron_gs)
vc = uccgsd(n_qubit, nocc = 1, orbital_rot=true, uccgsd = true, p_uccgsd = false)
#depth = n_qubit 
#vc = hev(n_qubit, depth )
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


if ARGS[1] == "plus_true"
    right_op = FermionOperator("$(up1)^ ", 1.0)
    right_op = jordan_wigner(right_op)
    left_op = FermionOperator("$(up1) ", 1.0)
    left_op = jordan_wigner(left_op)

    n_electron_ex = 3

    sign = 1
end


if ARGS[2] == "minus_true"
    right_op = FermionOperator("$(up1) ", 1.0)
    right_op = jordan_wigner(right_op)
    left_op = FermionOperator("$(up1)^ ", 1.0)
    left_op = jordan_wigner(left_op)

    n_electron_ex = 1

    sign = -1
end


vc_ex = uccgsd(n_qubit, nocc = 1, orbital_rot = true, uccgsd = true, p_uccgsd = false)
#vc_ex = hev(n_qubit, depth )
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

taus = read_and_parse_float(ARGS[3])
println("taus=",taus)

Gfunc_ij_list, norm =  compute_gtau(
    jordan_wigner(ham_op),
    left_op,
    right_op,
    vc_ex,
    state_gs,
    state0_ex,
    taus,
    d_theta,
    verbose = verbose,
    algorithm = "vqs",
    recursive = true
)

Gfunc_ij_list *= sign

println("Gfunc_ij_list_plus=", Gfunc_ij_list)
println("norm=", norm)



function write_to_txt(file_name, x, y)
    open(file_name, "w") do fp
        for i = 1:length(x)
            println(fp, x[i], " ", real(y[i]))
        end
    end
end

function write_to_txt_(file_name, x, y)
    open(file_name, "w") do fp
        for i = 1:length(x)
            println(fp, 1000-x[i], " ", -real(y[i]))
        end
    end
end

write_to_txt("gf_dimer_plus.txt", taus, Gfunc_ij_list)
write_to_txt("dimer_plus_vqs_norm.txt", taus, norm)
#write_to_txt_("gf_dimer_recurisive_direct_minus3_freq.txt", taus, Gfunc_ij_list)
println("done!!")
