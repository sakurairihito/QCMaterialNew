module QCMaterial

import PyCall: pyimport, PyNULL, PyVector
using Requires

# Refer to https://discourse.julialang.org/t/pyimport-works-from-repl-defined-module-but-not-in-my-package/43539.
const ofermion = PyNULL()
const qulacs = PyNULL()
const pyutil = PyNULL()

function __init__()
    copy!(ofermion, pyimport("openfermion"))
    copy!(qulacs, pyimport("qulacs"))
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
    copy!(pyutil, pyimport("pyutil"))
    @require MPI="da04e1cc-30fd-572f-bb4f-1f8673147195" include("mpi.jl")
end

include("util.jl")
include("uccsd.jl")
include("hartree_fock.jl")
include("no_mpi.jl")

end