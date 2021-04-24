export topylist, doublefunc, numerical_grad
export up_index, down_index, update_circuit_param!, update_quantum_state!
using PyCall

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

"""
Make a wrapped scipy minimizer
"""
function mk_scipy_minimize(method::String="BFGS", callback=nothing, options=nothing)
    scipy_opt = pyimport("scipy.optimize")
    function minimize(cost, x0)
        res = scipy_opt.minimize(cost, x0, method=method, callback=callback, options=options)
        res["x"]
    end
    return minimize
end