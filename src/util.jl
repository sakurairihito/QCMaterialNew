export topylist, doublefunc, numerical_grad
export up_index, down_index, update_circuit_param!, update_quantum_state!
import PyCall: PyVector


up_index(i) = 2*(i-1)
down_index(i) = 2*(i-1)+1

"""
Convert openfermion operator for generic cases (non-Hermitian operators)
Args:
    n_qubit (:class:`int`)
    openfermion_op (:class:`openfermion.ops.QubitOperator`)
Returns:
    :class:`qulacs.GeneralQuantumOperator`
"""

function parse_of_general_operators(num_qubits, openfermion_operators)
    ret = qulacs.GeneralQuantumOperator(num_qubits)
    for (pauli_product, coef) in openfermion_operators.terms
        pauli_string = ""
        for pauli_operator in pauli_product
            pauli_string *= "$(pauli_operator[2]) $(pauli_operator[1]) "
        end
        ret.add_operator(coef, pauli_string[1:end-1])
    end
    ret
end


function count_qubit_in_qubit_operator(op)
    n_qubits = 0
    for pauli_product in op.terms
        for pauli_operator in pauli_product[1]
            if n_qubits < pauli_operator[1]
                n_qubits = pauli_operator[1]
            end
        end
    end
    n_qubits+1
end



"""
wrapper for openfermion.jordan_wigner which directly converts 
openfermion.FermionOperator to qulacs.GeneralQuantumOperator
Args:
    fermion_operator (openfermion.FermionOperator)
    n_qubits (int):
        # of qubits (if not given, n_qubits is assumed to be 
        the number of orbitals which appears in the given fermion operator)
Return:
    qulacs.GeneralQuantumOperator
"""

function qulacs_jordan_wigner(fermion_operator, n_qubits=nothing)
    qubit_operator = ofermion.transforms.jordan_wigner(fermion_operator)
    _n_qubits = n_qubits === nothing ? count_qubit_in_qubit_operator(qubit_operator) : n_qubits
    qulacs_operator = parse_of_general_operators(_n_qubits, qubit_operator)
    return qulacs_operator
end


"""
Convert_openfermion_op

Args:
    n_qubit (:class:`int`)
    openfermion_op (:class:`openfermion.ops.QubitOperator`)
Returns:
    :class:`qulacs.Observable`
"""
function convert_openfermion_op(n_qubit, openfermion_op)
    ret = qulacs.Observable(n_qubit)
    for (pauli_product, coef) in openfermion_op.terms
        pauli_string = ""
        for pauli_operator in pauli_product
            pauli_string *= pauli_operator[2] * " $(pauli_operator[1]) "
        end
        ret.add_operator(real(coef), pauli_string[1:end-1])
    end
    ret
end


function add_parametric_circuit_using_generator!(circuit,
                                           generator, theta) 
    for i_term in 0:generator.get_term_count()-1
        pauli = generator.get_term(i_term)
        pauli_id_list = pauli.get_pauli_id_list()
        pauli_index_list = pauli.get_index_list()
        pauli_coef = imag(pauli.get_coef()) #coef should be pure imaginary
        circuit.add_parametric_multi_Pauli_rotation_gate( 
                        pauli_index_list, pauli_id_list,
                        theta)
    end
end


function add_theta_value_offset!(theta_offsets, generator, ioff)
    pauli_coef_lists = Float64[]
    for i in 0:generator.get_term_count()-1
        pauli = generator.get_term(i)
        push!(pauli_coef_lists, imag(pauli.get_coef())) #coef should be pure imaginary
    end
    push!(theta_offsets, [generator.get_term_count(), ioff, pauli_coef_lists])
    ioff = ioff + generator.get_term_count()
    return theta_offsets, ioff
end


function doublefunc(x)
    return pyutil.doublefunc(x)
end

function topylist(array::Array{T}) where T
    pylist = PyVector(Vector{Any}())
    for x in array
        push!(pylist, x)
    end
    pylist
end



"""
Compute partial derivative of a given function at a point x
"""
function numerical_grad(f, x::Vector{Float64}; dx=1e-8, first_idx=1, last_idx=length(x))
    deriv = zero(x)
    x_new1 = copy(x)
    x_new2 = copy(x)
    for i in first_idx:last_idx
        x_new1[i] += dx
        x_new2[i] -= dx
        deriv[i] = (f(x_new1) - f(x_new2))/(2*dx)
        x_new1[i] = x_new2[i] = x[i]
    end
    deriv
end


"""
Check if the Python type of a given PyObject matches the expected one
"""
function check_py_type(py_object::PyObject, py_class_name::String)
    if py_object.__class__.__name__ != py_class_name
        error("Expected PyObject type $(py_object.__class__.__name__), expected $(py_class_name)")
    end
    py_object
end