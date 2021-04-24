using Test
using QCMaterial

@testset "computation.apply_qubit_op" begin
    n_qubit = 2
    # Prepare |Psi> = |00>
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b00)
    println("state", get_vector(state))

    # Apply c^dagger_1 to |Psi>, which will yield |10>.
    op = jordan_wigner(FermionOperator("1^"))

    # Prepare |phi> = |10>
    state0_bra = QulacsQuantumState(n_qubit)
    set_computational_basis!(state0_bra, 0b10)
    circuit_bra = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=false)

    squared_norm = apply_qubit_op(op, state, circuit_bra, state0_bra)

    # Verify the result
    @test squared_norm ≈ 1.
end