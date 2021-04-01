include("uccgsd_hubbard.jl")

using HDF5
using FileIO

nsite = 2
ham = generate_ham(nsite)
n_electron = 2
cost_history, circuit, exact_gs_ene, opt = solve(ham, n_electron)
println("cost_history", cost_history)
println(opt["x"])

h5open("opt.h5", "w") do file
    write(file, "theta_list", opt["x"])
    write(file, "cost_history", cost_history)
    write(file, "exact_gs_ene", exact_gs_ene)
    write(file, "n_electron", n_electron)
    write(file, "nsite", nsite)
end

#save("opt.jld2", Dict("cost_history" => cost_history, "circuit" => circuit))

# TODO: save optimized parameters, i.e., theta_list
import PyPlot
PyPlot.plot(cost_history, color="red", label="VQE")
PyPlot.plot(1:length(cost_history), fill(exact_gs_ene, length(cost_history)),
    linestyle="dashed", color="black", label="Exact Solution")
PyPlot.xlabel("Iteration")
PyPlot.ylabel("Energy expectation value")
PyPlot.legend()
PyPlot.savefig("cost_history.pdf")