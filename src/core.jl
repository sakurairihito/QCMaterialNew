export QulacsParametricQuantumCircuit
export num_theta, num_pauli, pauli_coeff, theta_offset

export QuantumState, QulacsQuantumState
export set_computational_basis!, create_hf_state
export FermionOperator, jordan_wigner, get_number_preserving_sparse_operator

export QubitOperator, OFQubitOperator
export get_term_count, n_qubit, terms_dict

export get_expectation_value, create_operator_from_openfermion
export update_quantum_state!, update_circuit_param!, get_circuit_param
export hermitian_conjugated
export create_observable

using PyCall


up_index(i) = 2*(i-1)
down_index(i) = 2*(i-1)+1

@enum PauliID pauli_I=0 pauli_X=1 pauli_Y=2 pauli_Z=3
pauli_id_lookup = Dict("I"=>pauli_I, "X"=>pauli_X, "Y"=>pauli_Y, "Z"=>pauli_Z)

################################################################################
############################# QUANTUM  CIRCUIT #################################
################################################################################
abstract type ParametricQuantumCircuit end
struct QulacsParametricQuantumCircuit <: ParametricQuantumCircuit
    pyobj::PyObject
    thetas::Vector{Float64}
    # Vector of (num_term_count::Int64, ioff::Int64, pauli_coeffs::Float64)
    theta_offsets::Vector{Tuple{Int64, Int64, Vector{Float64}}}
end

function num_theta(circuit::QulacsParametricQuantumCircuit)
    size(circuit.theta_offsets)[1]
end

function num_pauli(circuit::QulacsParametricQuantumCircuit, idx_theta::Int)
    circuit.theta_offsets[idx_theta][1]
end

function pauli_coeff(circuit::QulacsParametricQuantumCircuit, idx_theta::Int, idx_pauli::Int)
    circuit.theta_offsets[idx_theta][3][idx_pauli]
end

function theta_offset(circuit::QulacsParametricQuantumCircuit, idx_theta::Int)
    circuit.theta_offsets[idx_theta][2]
end

function add_parametric_multi_Pauli_rotation_gate!(circuit::QulacsParametricQuantumCircuit,
    pauli_indices::Vector{Int}, pauli_ids::Vector{PauliID}, theta_init::Float64=0.0)
    circuit.pyobj.add_parametric_multi_Pauli_rotation_gate(
        pauli_indices, Int.(pauli_ids), theta_init)
end

function QulacsParametricQuantumCircuit(n_qubit::Int)
    QulacsParametricQuantumCircuit(qulacs.ParametricQuantumCircuit(n_qubit), [], [])
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

function set_computational_basis!(state::QulacsQuantumState, int_state::Int)
    state.pyobj.set_computational_basis(int_state)
end

function create_hf_state(n_qubit, n_electron)
    int_state = parse(Int, repeat("0", n_qubit-n_electron) * repeat("1", n_electron), base=2)
    state = QulacsQuantumState(n_qubit) 
    set_computational_basis!(state, int_state)
    state
end

################################################################################
############################# FERMION OPERATOR #################################
################################################################################
# Wrap openfermion.ops.operators.FermionOperator
struct FermionOperator
    pyobj::PyObject
end


function FermionOperator(ops::Vector{Tuple{Int,Int}}, coeff::Number=1.0) 
    FermionOperator(ofermion.ops.operators.FermionOperator(Tuple(ops), coeff))
end

function FermionOperator(op_str::String="", coeff::Number=1.0)
    FermionOperator(ofermion.ops.operators.FermionOperator(op_str, coeff))
end

function Base.:+(op1::FermionOperator, op2::FermionOperator)
    FermionOperator(op1.pyobj + op2.pyobj)
end

function Base.:-(op1::FermionOperator, op2::FermionOperator)
    FermionOperator(op1.pyobj - op2.pyobj)
end

function Base.:/(op1::FermionOperator, x::Number)
    FermionOperator(op.pyobj/x)
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
abstract type QubitOperator end

# Wrap openfermion.ops.operators.qubit_operator.QubitOperator
struct OFQubitOperator <: QubitOperator
    pyobj
end

function OFQubitOperator(str::String, coeff::Number=1.0)
    return OFQubitOperator(ofermion.ops.operators.qubit_operator.QubitOperator(str, coeff))
end

function Base.:+(op1::OFQubitOperator, op2::OFQubitOperator)
    OFQubitOperator(op1.pyobj + op2.pyobj)
end

function Base.:-(op1::OFQubitOperator, op2::OFQubitOperator)
    OFQubitOperator(op1.pyobj - op2.pyobj)
end

function Base.:/(op::OFQubitOperator, x::Number)
    OFQubitOperator(op.pyobj/x)
end

function hermitian_conjugated(op::OFQubitOperator)
    OFQubitOperator(ofermion.utils.hermitian_conjugated(op.pyobj))
end

function n_qubit(op::OFQubitOperator)
    count_qubit_in_qubit_operator(op.pyobj)
end

function get_term_count(op::OFQubitOperator)
    length(op.pyobj.terms)
end

function terms_dict(op::OFQubitOperator)::Dict{Any,Any}
    op.pyobj.terms
end


################################################################################
############################# Observable #######################################
################################################################################
abstract type Observable end

# Wrap qulacs.Observable
struct QulacsObservable <: Observable
    qulacsobj
end

function create_observable(op::OFQubitOperator, n_qubit::Int=nothing)
    _n_qubit = n_qubit === nothing ? n_qubit(op) : n_qubit
    QulacsObservable(convert_openfermion_op(_n_qubit, op.pyobj))
end

function get_expectation_value(obs::QulacsObservable, state::QulacsQuantumState)
    obs.qulacsobj.get_expectation_value(state.pyobj) 
end


################################################################################
#############################  MANIPULATION  ###################################
################################################################################
"""
Update a state using a circuit
"""
function update_quantum_state!(circuit::QulacsParametricQuantumCircuit, state::QulacsQuantumState)
    circuit.pyobj.update_quantum_state(state.pyobj)
end


"""
Update circuit parameters
"""
function update_circuit_param!(circuit::QulacsParametricQuantumCircuit, thetas::Vector{Float64})
    if num_theta(circuit) != length(thetas)
        error("Invalid length of thetas!")
    end
    for (idx, theta) in enumerate(thetas)
        for ioff in 1:num_pauli(circuit, idx)
            pauli_coef = pauli_coeff(circuit, idx, ioff)
            circuit.pyobj.set_parameter(
                theta_offset(circuit, idx)+ioff-1, theta*pauli_coef) 
        end
    end
    circuit.thetas .= thetas
end


"""
Apply a qubit operator to a given quantum state and fit it with a circuit

This is done by maximizing |<phi|A|Psi>|^2, where |phi> = circuit * |init_state>,
with respect to circuit parameters.

operator:
    A qubit operator, A

state:
    State for which the operator is applied to, |Psi>

circuit:
    Circuit

thetas:
    Initial values of the circuit parameters

init_state:
    Initial state for which the circuit is applied to, |init_state>
"""
#function fit_operator_and_state(operator::QubitOperator, state::QuantumState, circuit::ParametricQuantumCircuit,
    #thetas::Vector{Float64} init_state::QuantumState)
#end

"""
Parse a tuple representing a Pauli string
  When x is ((0, "X"), (5, "Y")), returns [0, 5], [PauliID.X, PauliID.Y]
"""
function parse_pauli_str(x)::Tuple{Vector{Int64}, Vector{PauliID}}
    collect(e[1] for e in x), collect(pauli_id_lookup[e[2]] for e in x)
end


function add_parametric_circuit_using_generator!(circuit::ParametricQuantumCircuit, generator::QubitOperator,
    theta::Float64) 
    pauli_coeffs = Float64[]
    for (pauli_str, pauli_coef) in terms_dict(generator)
        pauli_index_list, pauli_id_list = parse_pauli_str(pauli_str)
        if length(pauli_index_list) == 0
            continue
        end
        pauli_coef = imag(pauli_coef) #coef should be pure imaginary
        push!(pauli_coeffs, pauli_coef)
        add_parametric_multi_Pauli_rotation_gate!(
            circuit, pauli_index_list, pauli_id_list, pauli_coef*theta)
    end
    num_thetas = num_theta(circuit)
    ioff = num_thetas == 0 ? 0 : theta_offset(circuit, num_thetas) + num_pauli(circuit, num_thetas)
    push!(circuit.thetas, theta)
    push!(circuit.theta_offsets, (get_term_count(generator), ioff, pauli_coeffs))
end