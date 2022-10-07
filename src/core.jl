using PyCall
export _convert_qubitop_str_from_py_to_jl
export FermionOperator
export rm_identity
# i = 1, 2, ...
up_index(i) = 2 * (i - 1) + 1
down_index(i) = 2 * i

@enum PauliID pauli_I = 0 pauli_X = 1 pauli_Y = 2 pauli_Z = 3
pauli_id_lookup = Dict("I" => pauli_I, "X" => pauli_X, "Y" => pauli_Y, "Z" => pauli_Z)
pauli_id_str = ["I", "X", "Y", "Z"]

################################################################################
############################# QUANTUM  CIRCUIT #################################
################################################################################
include("gate.jl")

################################################################################
############################# QUANTUM STATE  ###################################
################################################################################
include("quantum_state.jl")

################################################################################
############################# QUANTUM  CIRCUIT #################################
################################################################################
include("circuit.jl")


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
    typeof(op)(op.pyobj / x)
end

################################################################################
############################# FERMION OPERATOR #################################
################################################################################
# Wrap openfermion.ops.operators.FermionOperator
struct FermionOperator <: SecondQuantOperator
    pyobj::PyObject
end


function _parse_fermion_operator_str(str::String)
    regex = r"\d+\^*"
    res = Tuple{Int64,Int64}[]
    for x in eachmatch(regex, str)
        m = x.match
        op_type = m[end] == '^' ? 1 : 0
        idx = (op_type == 1 ? parse(Int64, m[1:end-1]) : parse(Int64, m))
        push!(res, (idx, op_type))
    end
    res
end

function FermionOperator(ops::Vector{Tuple{Int,Int}}, coeff::Number=1.0)
    for (idx, _) in ops
        if idx <= 0
            error("idx for fermion operator must be positive.")
        end
    end
    ops_py = [(idx - 1, op_type) for (idx, op_type) in ops]
    FermionOperator(ofermion.ops.operators.FermionOperator(Tuple(ops_py), coeff))
end

function FermionOperator()
    FermionOperator("", 0.0)
end

function FermionOperator(op_str::String, coeff::Number=1.0)
    FermionOperator(_parse_fermion_operator_str(op_str), coeff)
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

function _convert_qubitop_str_from_py_to_jl(str::String)
    regex = r"[XYZI]\d+"
    res = ""
    for x in eachmatch(regex, str)
        m = x.match
        pauli_op_char = m[1]
        idx = parse(Int64, m[2:end])
        if idx <= 0
            error("idx for qubit must be positive!")
        end
        res *= " $pauli_op_char$(idx-1)"
    end
    res
end

#function OFQubitOperator(str::String, coeff::Number=1.0)
#    return OFQubitOperator(ofermion.ops.operators.qubit_operator.QubitOperator(
#        _convert_qubitop_str_from_py_to_jl(str), coeff))
#end

function OFQubitOperator(str::String, coeff::Number=1.0)
    return OFQubitOperator(ofermion.ops.operators.qubit_operator.QubitOperator(
        _convert_qubitop_str_from_py_to_jl(str), coeff))
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
    # Change index convention
    dict_jl = Dict()
    for (k, v) in op.pyobj.terms
        if k == ()
            continue
        end
        k_new = Tuple(((k_[1] + 1, k_[2]) for k_ in k))
        dict_jl[k_new] = v
    end
    dict_jl
end


function rm_identity(op::OFQubitOperator)
    # Change index convention
    # op is operators after JW transformation
    #dict_jl = Dict()
    op1 = OFQubitOperator("", 0.0)
    op1 = nothing
    #println("op1_init=", op1)
    for (k, v) in op.pyobj.terms
        #println("k_before=", k)
        if k == ()
            #println("k == () is true")
            #@show k
            continue
        end
        #println("k=", k)
        #println("k[1]=", k[1])
        #println("k[1][1]=", k[1][1])
        #println("k[1][2]=", k[1][2])
        #println("k[2]=", k[2])

        # julia側のインデックスに直す
        k_new = Tuple(((k_[1] + 1, k_[2]) for k_ in k))
        #println("k_new", k_new)
        #k = k_new
        #println("k=", k)
        #dict_jl[k_new] = v
        #@show "$(k[1][2]) $(k[1][1] + 1)"
        #op1 += OFQubitOperator("$(k[1][2]) $(k[1][1])", v)
        # k=((1, "Z"), (2, "Z"))

        tmpop = OFQubitOperator("", 1.0)
        for (i, o) in k_new
            #@show (i, o)
            myop = OFQubitOperator("$(o)$(i)", 1.0)
            #@show myop
            tmpop *= myop
            #@show tmpop
        end
        tmpop *= OFQubitOperator("", v)
        #@show tmpop
        if op1 === nothing
            op1 = tmpop
        else
            op1 += tmpop
        end
        #println("op1=", op1)
        #op1 += prod(OFQubitOperator("$(o)$(i)", v) for (i, o) in k)
        # OFQubitOperator("Z1", 1.0) * OFQubitOperator("Z2", 1.0)
        #op1 += OFQubitOperator("$(k_[i][2]) $(k_[i][1])" for i in k, v) 
    end
    #println("Final result", op1)
    op1
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

function inner_product(state_bra::QulacsQuantumState, state_ket::QulacsQuantumState)
    qulacs.state.inner_product(state_bra.pyobj, state_ket.pyobj)
end


"""
Parse a tuple representing a Pauli string
  When x is ((1, "X"), (5, "Y")), returns [1, 5], [PauliID.X, PauliID.Y]
"""
function parse_pauli_str(x)::Tuple{Vector{Int64},Vector{PauliID}}
    collect(e[1] for e in x), collect(pauli_id_lookup[e[2]] for e in x)
end


"""
Update a state using a circuit
"""
function update_quantum_state!(circuit::QulacsParametricQuantumCircuit, state::QulacsQuantumState)
    circuit.pyobj.update_quantum_state(state.pyobj)
end

function update_quantum_state!(circuit::QulacsQuantumCircuit, state::QulacsQuantumState)
    circuit.pyobj.update_quantum_state(state.pyobj)
end

