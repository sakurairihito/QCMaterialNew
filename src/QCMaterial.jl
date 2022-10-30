module QCMaterial

import PyCall: pyimport, PyNULL, PyVector
using Requires


# Refer to https://discourse.julialang.org/t/pyimport-works-from-repl-defined-module-but-not-in-my-package/43539.
const ofermion = PyNULL()
const ofpyscf = PyNULL()
const qulacs = PyNULL()
const pyutil = PyNULL()

function __init__()
    copy!(ofermion, pyimport("openfermion"))
    copy!(ofpyscf, pyimport("openfermionpyscf"))
    copy!(qulacs, pyimport("qulacs"))
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__) #@_DIR_ファイルのある場所=pyutilの場所
    copy!(pyutil, pyimport("pyutil"))
    @require MPI="da04e1cc-30fd-572f-bb4f-1f8673147195" include("mpi.jl")
end

const is_mpi_on = haskey(ENV, "OMPI_COMM_WORLD_RANK") || haskey(ENV, "PMI_RANK")
if is_mpi_on
    import MPI
end

include("exports.jl")
include("util.jl")
include("core.jl")
include("hamiltonian.jl")
include("uccsd.jl")
include("computation.jl")
include("hartree_fock.jl")
include("no_mpi.jl")
include("vqe.jl")
include("vqs.jl")
include("obs.jl")
include("hev.jl")
include("opt.jl")
end
