using Test
using QCMaterial

@testset "computation.apply_qubit_op" begin
    n_qubit = 2
    # Prepare |Psi> = |00>
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b00)

    # Apply c^dagger_1 to |Psi>, which will yield |10>.
    op = jordan_wigner(FermionOperator("2^"))

    # Prepare |phi> = |01>
    state0_bra = QulacsQuantumState(n_qubit)
    set_computational_basis!(state0_bra, 0b01)
    
    #debug
    circuit_bra = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=false)
    #c = QulacsParametricQuantumCircuit(n_qubit)
    #add_parametric_RY_gate!(c, 1, 0.5*pi)
    #circuit_bra = QulacsVariationalQuantumCircuit(c)

    squared_norm = apply_qubit_op!(op, state, circuit_bra, state0_bra)
    println("squared_norm=",squared_norm)


    # Verify the result
    @test squared_norm ≈ 1.0
end

#@testset "computation.apply_qubit_op2" begin
    #n_qubit = 2
    ## Prepare |Psi> = (|01> + |10>)/sqrt(2)
    #state = QulacsQuantumState(n_qubit)
    #set_computational_basis!(state, 0b01)
#
    #state2 = QulacsQuantumState(n_qubit)
    #set_computational_basis!(state, 0b10)
#
    ##state + state2
#
#end