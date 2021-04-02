include("uccgsd_hubbard.jl")

using HDF5
import MPI

# Initialize MPI environment
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

nsite = 4
ham = generate_ham(nsite)
n_electron = 4
cost_history, circuit, exact_gs_ene, opt = solve(ham, n_electron, comm=comm)

if rank == 0
    println("cost_history", cost_history)
    println(opt["x"])

    h5open("opt.h5", "w") do file
        write(file, "theta_list", opt["x"])
        write(file, "cost_history", cost_history)
        write(file, "exact_gs_ene", exact_gs_ene)
        write(file, "n_electron", n_electron)
        write(file, "nsite", nsite)
    end
end
