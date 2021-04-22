export Circuit, QulacsCircuit
export num_theta, num_pauli, pauli_coeff, theta_offset
export QuantumState, QulacsQuantumState
export set_computational_basis!, create_hf_state
export FermionOperator, jordan_wigner, get_number_preserving_sparse_operator
export QubitOperator, OFQubitOperator
export get_expectation_value, create_operator_from_openfermion
export update_quantum_state!, update_circuit_param!
export hermitian_conjugated
export create_observable

using PyCall

################################################################################
############################### CIRCUIT ########################################
################################################################################
abstract type Circuit end
struct QulacsCircuit <: Circuit
    qulacs_circuit
    theta_offsets
end

function num_theta(circuit::QulacsCircuit)
    size(circuit.theta_offsets)[1]
end

function num_pauli(circuit::QulacsCircuit, idx_theta::Int)
    circuit.theta_offsets[idx_theta][1]
end

function pauli_coeff(circuit::QulacsCircuit, idx_theta::Int, idx_pauli::Int)
    circuit.theta_offsets[idx_theta][3][idx_pauli]
end

function theta_offset(circuit::QulacsCircuit, idx_theta::Int)
    circuit.theta_offsets[idx_theta][2]
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
    #OFQubitOperator(
        #qulacs.observable.create_observable_from_openfermion_text(
            #ofermion.transforms.jordan_wigner(ham.pyobj).__str__()
        #)
    #)
end

################################################################################
############################# QUBIT OPERATOR ###################################
################################################################################
abstract type QubitOperator end

# Wrap openfermion.ops.operators.qubit_operator.QubitOperator
struct OFQubitOperator <: QubitOperator
    pyobj
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

################################################################################
############################# Observable #######################################
################################################################################
abstract type Observable end

# Wrap qulacs.Observable
struct QulacsObservable <: Observable
    qulacsobj
end

function create_observable(op::OFQubitOperator, n_qubit::Int)
    QulacsObservable(convert_openfermion_op(n_qubit, op.pyobj))
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
function update_quantum_state!(circuit::QulacsCircuit, state::QulacsQuantumState)
    circuit.qulacs_circuit.update_quantum_state(state.pyobj)
end


"""
Update circuit parameters
"""
function update_circuit_param!(circuit::Circuit, thetas::Vector{Float64})
    if num_theta(circuit) != length(thetas)
        error("Invalid length of thetas!")
    end
    for (idx, theta) in enumerate(thetas)
        for ioff in 1:num_pauli(circuit, idx)
            pauli_coef = pauli_coeff(circuit, idx, ioff)
            circuit.qulacs_circuit.set_parameter(
                theta_offset(circuit, idx)+ioff-1, theta*pauli_coef) 
        end
    end
end


"""
Get circuit parameters
"""
#function get_circuit_param(circuit::Circuit)
    #thetas = get_num_theta(circuit)
    #for (idx, theta) in enumerate(theta_list)
        #for ioff in 1:circuit.theta_offsets[idx][1]
            #pauli_coef = circuit.theta_offsets[idx][3][ioff]
            #circuit.qulacs_circuit.set_parameter(circuit.theta_offsets[idx][2]+ioff-1, 
                              #theta*pauli_coef) 
        #end
    #end
#end


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
#function apply_operator_and_fit(operator::QubitOperator, state::QuantumState, circuit::Circuit,
    #thetas::Vector{Float64} init_state::QuantumState)
#
#end

function hermitian_conjugated(op::OFQubitOperator)
    OFQubitOperator(ofermion.utils.hermitian_conjugated(op.pyobj))
end
