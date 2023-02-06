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


@testset "state_sampling" begin
    n_qubit = 2
    state = QulacsQuantumState(n_qubit)
    nshots = 5
    res = state_sampling(state, nshots)
    @test res == [0, 0, 0, 0, 0]
    @show res
end

@testset "state_sampling" begin
    n_qubit = 2
    state = QulacsQuantumState(n_qubit)
    #nshots = 5
    idex = 1
    
    res = update_quantum_state_gate!()
    #@test res == [0, 0, 0, 0, 0]
    @show res
end
