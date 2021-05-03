export QulacsParametricQuantumCircuit
export add_parametric_multi_Pauli_rotation_gate!, set_parameter!

export QuantumState, QulacsQuantumState
export set_computational_basis!, create_hf_state
export FermionOperator, jordan_wigner, get_number_preserving_sparse_operator
export get_vector, get_n_qubit

export QubitOperator, OFQubitOperator
export get_term_count, get_n_qubit, terms_dict, is_hermitian
export get_expectation_value, get_transition_amplitude

export get_expectation_value, create_operator_from_openfermion
export hermitian_conjugated
export create_observable

using PyCall


up_index(i) = 2*(i-1)
down_index(i) = 2*(i-1)+1

@enum PauliID pauli_I=0 pauli_X=1 pauli_Y=2 pauli_Z=3
pauli_id_lookup = Dict("I"=>pauli_I, "X"=>pauli_X, "Y"=>pauli_Y, "Z"=>pauli_Z)
pauli_id_str = ["I", "X", "Y", "Z"]

################################################################################
############################# QUANTUM  CIRCUIT #################################
################################################################################
abstract type ParametricQuantumCircuit end
struct QulacsParametricQuantumCircuit <: ParametricQuantumCircuit
    pyobj::PyObject
end

function QulacsParametricQuantumCircuit(n_qubit::Int)
    QulacsParametricQuantumCircuit(qulacs.ParametricQuantumCircuit(n_qubit))
end

function add_parametric_multi_Pauli_rotation_gate!(circuit::QulacsParametricQuantumCircuit,
    pauli_indices::Vector{Int}, pauli_ids::Vector{PauliID}, theta_init::Float64=0.0)
    circuit.pyobj.add_parametric_multi_Pauli_rotation_gate(
        pauli_indices, Int.(pauli_ids), theta_init)
end


function set_parameter!(circuit::QulacsParametricQuantumCircuit, index, theta)
    circuit.pyobj.set_parameter(index, theta)
end

################################################################################
############################# QUANTUM STATE  ###################################
################################################################################
abstract type QuantumState end

# Wrap qulacs.QuantumState
struct QulacsQuantumState <: QuantumState
    pyobj
end

function QulacsQuantumState(n_qubit::Int)
    QulacsQuantumState(qulacs.QuantumState(n_qubit))
end

function Base.:+(state1::QulacsQuantumState, state2::QulacsQuantumState)
    FermionOperator(state1.pyobj + state2.pyobj)
end

function Base.:-(state1::QulacsQuantumState, state2::QulacsQuantumState)
    FermionOperator(state1.pyobj - state2.pyobj)
end

function Base.:copy(state::QulacsQuantumState)
    QulacsQuantumState(state.pyobj.copy())
end

function set_computational_basis!(state::QulacsQuantumState, int_state)
    state.pyobj.set_computational_basis(int_state)
end

function create_hf_state(n_qubit, n_electron)
    int_state = parse(Int, repeat("0", n_qubit-n_electron) * repeat("1", n_electron), base=2)
    state = QulacsQuantumState(n_qubit) 
    set_computational_basis!(state, int_state)
    state
end

function get_n_qubit(state::QulacsQuantumState)
    state.pyobj.get_qubit_count()
end

function get_vector(state::QuantumState)
    state.pyobj.get_vector()
end


################################################################################
################################# OPERATOR #####################################
################################################################################
abstract type SecondQuantOperator end

function Base.:(==)(op1::SecondQuantOperator, op2::SecondQuantOperator)
    op1.pyobj == op2.pyobj
end

function Base.:*(op1::SecondQuantOperator, op2::SecondQuantOperator)
    typeof(op1)(op1.pyobj * op2.pyobj)
end

function Base.:+(op1::SecondQuantOperator, op2::SecondQuantOperator)
    typeof(op1)(op1.pyobj + op2.pyobj)
end

function Base.:-(op1::SecondQuantOperator, op2::SecondQuantOperator)
    typeof(op1)(op1.pyobj - op2.pyobj)
end

function Base.:/(op::SecondQuantOperator, x::Number)
    typeof(op)(op.pyobj/x)
end

################################################################################
############################# FERMION OPERATOR #################################
################################################################################
# Wrap openfermion.ops.operators.FermionOperator
struct FermionOperator <: SecondQuantOperator
    pyobj::PyObject
end


function FermionOperator(ops::Vector{Tuple{Int,Int}}, coeff::Number=1.0) 
    FermionOperator(ofermion.ops.operators.FermionOperator(Tuple(ops), coeff))
end

function FermionOperator(op_str::String="", coeff::Number=1.0)
    FermionOperator(ofermion.ops.operators.FermionOperator(op_str, coeff))
end

function get_number_preserving_sparse_operator(ham::FermionOperator, n_qubit::Int, n_electron::Int)::PyObject
    ofermion.linalg.get_number_preserving_sparse_operator(ham.pyobj, n_qubit, n_electron)
end

function jordan_wigner(op::FermionOperator)
    OFQubitOperator(ofermion.transforms.jordan_wigner(op.pyobj))
end

################################################################################
############################# QUBIT OPERATOR ###################################
################################################################################
abstract type QubitOperator <: SecondQuantOperator end

# Wrap openfermion.ops.operators.qubit_operator.QubitOperator
struct OFQubitOperator <: QubitOperator
    pyobj
end

function OFQubitOperator(str::String, coeff::Number=1.0)
    return OFQubitOperator(ofermion.ops.operators.qubit_operator.QubitOperator(str, coeff))
end

function hermitian_conjugated(op::OFQubitOperator)
    OFQubitOperator(ofermion.utils.hermitian_conjugated(op.pyobj))
end

function get_n_qubit(op::OFQubitOperator)
    count_qubit_in_qubit_operator(op.pyobj)
end

function get_term_count(op::OFQubitOperator)
    length(op.pyobj.terms)
end

function terms_dict(op::OFQubitOperator)::Dict{Any,Any}
    op.pyobj.terms
end

function is_hermitian(op::OFQubitOperator)
    ofermion.utils.operator_utils.is_hermitian(op.pyobj)
end

function convert_to_qulacs_op(op::OFQubitOperator, n_qubit::Int)
    convert_openfermion_op(n_qubit, op.pyobj)
end

function get_expectation_value(op::OFQubitOperator, state::QulacsQuantumState)
    if !is_hermitian(op)
        error("op must be hermite for get_expectation_value!")
    end
    convert_to_qulacs_op(op, get_n_qubit(state)).get_expectation_value(state.pyobj) 
end

function get_transition_amplitude(op::OFQubitOperator, state_bra::QulacsQuantumState, state_ket::QulacsQuantumState)
    convert_to_qulacs_op(op, get_n_qubit(state_bra)).get_transition_amplitude(state_bra.pyobj, state_ket.pyobj)
end


"""
Parse a tuple representing a Pauli string
  When x is ((0, "X"), (5, "Y")), returns [0, 5], [PauliID.X, PauliID.Y]
"""
function parse_pauli_str(x)::Tuple{Vector{Int64}, Vector{PauliID}}
    collect(e[1] for e in x), collect(pauli_id_lookup[e[2]] for e in x)
end


"""
Update a state using a circuit
"""
function update_quantum_state!(circuit::QulacsParametricQuantumCircuit, state::QulacsQuantumState)
    circuit.pyobj.update_quantum_state(state.pyobj)
end
