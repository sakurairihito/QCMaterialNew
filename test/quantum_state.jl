using Test
using QCMaterial

@testset "state_load" begin
    n_qubit = 2
    state = QulacsQuantumState(n_qubit)
    #@show state
    vec = [0, 1, 2, 3]
    state_load!(state, vec)
    #@show state
    res = get_vector(state)
    @test res == vec
end