
export PauliID, pauli_I, pauli_X, pauli_Y, pauli_Z
export QuantumCircuit, QulacsQuantumCircuit
export ParametricQuantumCircuit, QulacsParametricQuantumCircuit
export add_parametric_multi_Pauli_rotation_gate!, set_parameter!
export add_CNOT_gate!, add_H_gate!, add_X_gate!, add_S_gate!,add_Sdag_gate!, add_Z_gate!
export get_parameter_count, get_parameter
export add_parametric_RX_gate!
export add_parametric_RY_gate!
export add_parametric_RZ_gate!
export add_RY_gate!
export add_RZ_gate!
export add_SWAP_gate!
export add_gate!
export get_thetas

export VariationalQuantumCircuit
export QulacsVariationalQuantumCircuit

<<<<<<< HEAD
export Gate, QulacsGate, RX, RY, RZ, get_matrix
=======
export Gate, QulacsGate, RX, RY, RZ, get_matrix, add_U1_gate!
>>>>>>> master

export QuantumState, QulacsQuantumState
export set_computational_basis!, create_hf_state
export FermionOperator, jordan_wigner, get_number_preserving_sparse_operator
export get_vector, get_n_qubit

export QubitOperator, OFQubitOperator
export get_term_count, get_n_qubit, terms_dict, is_hermitian
export get_expectation_value, get_transition_amplitude
export inner_product

export get_expectation_value, create_operator_from_openfermion
export hermitian_conjugated
export create_observable

export imag_time_evolve
export compute_gtau
export compute_A
export compute_C
export compute_thetadot
#export _create_quantum_state
export taus_list
export read_taus_list
export compute_next_thetas_direct
<<<<<<< HEAD
export compute_next_thetas_vqs
=======
export compute_next_thetas_vqs
export compute_fubini
export compute_F
export overlap, compute_B, compute_B2, compute_B2x, compute_B3 #vqs.jl
>>>>>>> master
