using QCMaterial
using QCMaterial.HartreeFock
using Test
using PyCall
using Requires
import PyCall: pyimport
import QCMaterial: uccgsd, convert_openfermion_op, up_index, down_index

function __init__()
    # Enable MPI tests only when MPI.jl is loaded
    @require MPI="da04e1cc-30fd-572f-bb4f-1f8673147195" include("mpi.jl")
end

include("util.jl")
include("gate.jl")
include("core.jl")
include("hamiltonian.jl")
include("hartree_fock.jl")
include("ucc.jl")
include("computation.jl")
include("vqs.jl")
