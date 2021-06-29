################################################################################
############################# QUANTUM  CIRCUIT #################################
################################################################################
abstract type QuantumCircuit end
abstract type ParametricQuantumCircuit <: QuantumCircuit end

#  QuantumCircuit
struct QulacsQuantumCircuit <: QuantumCircuit
    pyobj::PyObject
end

function QulacsQuantumCircuit(n_qubit::Int)
    QulacsQuantumCircuit(qulacs.QuantumCircuit(n_qubit))
end

function Base.copy(circuit::QulacsQuantumCircuit)
    QulacsQuantumCircuit(circuit.pyobj.copy())
end

function add_X_gate!(circuit::QulacsQuantumCircuit, idx_qubit::Int)
    circuit.pyobj.add_X_gate(idx_qubit-1)
end

function add_H_gate!(circuit::QulacsQuantumCircuit, idx_qubit::Int)
    circuit.pyobj.add_H_gate(idx_qubit-1)
end

function add_S_gate!(circuit::QulacsQuantumCircuit, idx_qubit::Int)
    circuit.pyobj.add_S_gate(idx_qubit-1)
end

function add_Sdag_gate!(circuit::QulacsQuantumCircuit, idx_qubit::Int)
    circuit.pyobj.add_Sdag_gate(idx_qubit-1)
end


function add_CNOT_gate!(circuit::QulacsQuantumCircuit, control::Int, target::Int)
    circuit.pyobj.add_CNOT_gate(control-1, target-1)
end

function add_RY_gate!(circuit::QulacsQuantumCircuit, idx_qubit::Int, angle::Float64)
    circuit.pyobj.add_RY_gate(idx_qubit-1, angle)
end

function add_RZ_gate!(circuit::QulacsQuantumCircuit, idx_qubit::Int, angle::Float64)
    circuit.pyobj.add_RZ_gate(idx_qubit-1, angle)
end



function add_SWAP_gate!(circuit::QulacsQuantumCircuit, idx_qubit_1::Int, idx_qubit_2::Int)
    circuit.pyobj.add_SWAP_gate(idx_qubit_1-1, idx_qubit_2-1)
end


################################################################################
############################# Quantum Parametric CIRCUIT #######################
################################################################################
struct QulacsParametricQuantumCircuit <: ParametricQuantumCircuit
    pyobj::PyObject
end

function QulacsParametricQuantumCircuit(n_qubit::Int)
    QulacsParametricQuantumCircuit(qulacs.ParametricQuantumCircuit(n_qubit))
end

function Base.copy(circuit::QulacsParametricQuantumCircuit)
    QulacsParametricQuantumCircuit(circuit.pyobj.copy())
end

function add_parametric_multi_Pauli_rotation_gate!(circuit::QulacsParametricQuantumCircuit,
    pauli_indices::Vector{Int}, pauli_ids::Vector{PauliID}, theta_init::Float64=0.0)
    if !all(pauli_indices .>= 1)
        error("pauli indices are out of range!")
    end
    circuit.pyobj.add_parametric_multi_Pauli_rotation_gate(
        pauli_indices .- 1, Int.(pauli_ids), theta_init)
end

function set_parameter!(circuit::QulacsParametricQuantumCircuit, index, theta)
    if index <= 0 || index > get_parameter_count(circuit)
        error("index is out of range!")
    end
    circuit.pyobj.set_parameter(index-1, theta)
end

function get_parameter_count(circuit::QulacsParametricQuantumCircuit)::Int64
    circuit.pyobj.get_parameter_count()
end

function get_parameter(circuit::QulacsParametricQuantumCircuit, idx::Int)
    @assert idx >= 1
    circuit.pyobj.get_parameter(idx-1)
end

function add_parametric_RX_gate!(circuit::QulacsParametricQuantumCircuit, i::Int, angle::Float64)
    circuit.pyobj.add_parametric_RX_gate(i-1, angle)
end

function add_parametric_RY_gate!(circuit::QulacsParametricQuantumCircuit, i::Int, angle::Float64)
    circuit.pyobj.add_parametric_RY_gate(i-1, angle)
end

function add_parametric_RZ_gate!(circuit::QulacsParametricQuantumCircuit, i::Int, angle::Float64)
    circuit.pyobj.add_parametric_RZ_gate(i-1, angle)
end

function add_S_gate!(circuit::QulacsParametricQuantumCircuit, idx_qubit::Int)
    circuit.pyobj.add_S_gate(idx_qubit-1)
end

function add_Sdag_gate!(circuit::QulacsParametricQuantumCircuit, idx_qubit::Int)
    circuit.pyobj.add_Sdag_gate(idx_qubit-1)
end

function add_Z_gate!(circuit::QulacsParametricQuantumCircuit, idx_qubit::Int)
    circuit.pyobj.add_S_gate(idx_qubit-1)
end

function add_X_gate!(circuit::QulacsParametricQuantumCircuit, idx_qubit::Int)
    circuit.pyobj.add_X_gate(idx_qubit-1)
end

function add_CNOT_gate!(circuit::QulacsParametricQuantumCircuit, control::Int, target::Int)
    circuit.pyobj.add_CNOT_gate(control-1, target-1)
end

function add_RY_gate!(circuit::QulacsParametricQuantumCircuit, idx_qubit::Int, angle::Float64)
    circuit.pyobj.add_RY_gate(idx_qubit-1, angle)
end

function add_RZ_gate!(circuit::QulacsParametricQuantumCircuit, idx_qubit::Int, angle::Float64)
    circuit.pyobj.add_RZ_gate(idx_qubit-1, angle)
end

#function add_SWAP_gate!(circuit::QulacsPatametricQuantumCircuit, idx_qubit_1::Int, idx_qubit_2::Int)
#    circuit.pyobj.add_SWAP_gate(idx_qubit_1-1, idx_qubit_2-1)
#end



################################################################################
##################  Variational QUANTUM  CIRCUIT ###############################
################################################################################
abstract type VariationalQuantumCircuit end

"""
Return the number of independent variational parameters
"""
function num_theta(circuit::VariationalQuantumCircuit)::Int64
    0
end

"""
Return a copy of variational parameters
"""
function get_thetas(circuit::VariationalQuantumCircuit)::Vector{Float64}
    Float64[]
end

"""
Update the values of the independent variational parameters
"""
function update_circuit_param!(circuit::VariationalQuantumCircuit, thetas::Vector{Float64})
    # update parameters
end

"""
Update a state using a circuit
"""
function update_quantum_state!(ucccirc::VariationalQuantumCircuit, state::QuantumState)
    # do something
end

"""
Wrap a QulacsParametricQuantumCircuit object, which will not be copied.
"""
struct QulacsVariationalQuantumCircuit <: VariationalQuantumCircuit
    qcircuit::QulacsParametricQuantumCircuit
end

"""
Copy a QulacsVariationalQuantumCircuit object.
This makes a copy of the underlying QulacsParametricQuantumCircuit object. 
"""
function Base.copy(circuit::QulacsVariationalQuantumCircuit)
    QulacsVariationalQuantumCircuit(copy(circuit.qcircuit))
end

function num_theta(circuit::QulacsVariationalQuantumCircuit)
    get_parameter_count(circuit.qcircuit)
end

function get_thetas(circuit::QulacsVariationalQuantumCircuit)::Vector{Float64}
    return Float64[get_parameter(circuit.qcircuit, i)
        for i in 1:get_parameter_count(circuit.qcircuit)]
end

function update_circuit_param!(circuit::QulacsVariationalQuantumCircuit, thetas::Vector{Float64})
    for i in eachindex(thetas)
        set_parameter!(circuit.qcircuit, i, thetas[i])
    end
end

function update_quantum_state!(circuit::QulacsVariationalQuantumCircuit, state::QulacsQuantumState)
    update_quantum_state!(circuit.qcircuit, state)
end