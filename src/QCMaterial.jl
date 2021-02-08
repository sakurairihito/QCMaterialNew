module QCMaterial

import PyCall: pyimport, PyNULL

# Refer to https://discourse.julialang.org/t/pyimport-works-from-repl-defined-module-but-not-in-my-package/43539.
const ofermion = PyNULL()
const qulacs = PyNULL()

function __init__()
    copy!(ofermion, pyimport("openfermion"))
    copy!(qulacs, pyimport("qulacs"))
end

include("util.jl")
include("uccsd.jl")

end
