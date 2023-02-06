using Test
using QCMaterial

@testset "gate.H_gate" begin
    n_qubit = 2
    state = QulacsQuantumState(n_qubit)
    index = 1
    h_gate = H(index)
end