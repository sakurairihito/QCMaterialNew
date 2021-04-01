module QCMaterial

import PyCall: pyimport, PyNULL, PyVector

# Refer to https://discourse.julialang.org/t/pyimport-works-from-repl-defined-module-but-not-in-my-package/43539.
const ofermion = PyNULL()
const qulacs = PyNULL()
const pyutil = PyNULL()

function __init__()
    copy!(ofermion, pyimport("openfermion"))
    copy!(qulacs, pyimport("qulacs"))
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
    copy!(pyutil, pyimport("pyutil"))
end

include("util.jl")
include("uccsd.jl")
include("hartree_fock.jl")
include("mpi.jl")

end
