export topylist, doublefunc, numerical_grad, read_and_parse_float,write_to_txt_1,write_to_txt_2
export up_index, down_index, update_circuit_param!, update_quantum_state!, ParamInfo, expand 
export compact_paraminfo, make_long_param_from_compact, make_compact_params
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
function mk_scipy_minimize(method::String="BFGS";
    callback=nothing, options=nothing, use_mpi=true, verbose=false)
    scipy_opt = pyimport("scipy.optimize")
    function minimize(cost, x0)
        jac = nothing
        if use_mpi
            jac = generate_numerical_grad(cost, verbose=verbose)
            if verbose
                println("Using parallelized numerical grad")
            end
        end
        res = scipy_opt.minimize(cost, x0, method=method,
           jac=jac, callback=callback, options=options)
        res["x"]
    end
    return minimize
end

"""
Generates parallelized numerical grad_cost
"""
function generate_numerical_grad(f; verbose=true, comm=MPI_COMM_WORLD)
    function grad(x)
        t1 = time_ns()
        if comm === nothing
            first_idx, size = 1, length(x)
        else
            first_idx, size = distribute(length(x), MPI.Comm_size(comm), MPI.Comm_rank(comm))
        end
        last_idx = first_idx + size - 1
        res = numerical_grad(f, x, first_idx=first_idx, last_idx=last_idx)
        if comm !== nothing
            res = MPI.Allreduce(res, MPI.SUM, comm)
        end
        t2 = time_ns()
        if verbose && MPI_rank == 0
            println("g: ", (t2-t1)*1e-9)
        end
        res
    end
    return grad
end


function fit_svd(y, A, eps=1e-10)
    U, S, V = svd(A)
    num_nz = sum(S .> S[1]*eps)
    Uy = (U[:,1:num_nz]') * y
    SUy = Uy ./ S[1:num_nz]
    V[:,1:num_nz] * SUy
end

"""
y = Ax
tikhov regularization: minimize ||Ax-b||^2 + λ||x^2|| 
"""
function tikhonov(y, A, eps = 1e-10)
    AA(λ) = A'*A + λ*I
    x = inv(AA(eps)) * A' * y
end




"""
Make mapping from  
redundant parameter information
"""

struct ParamInfo
    nparam::Int  # Number of unique parameters
    nparamlong::Int # Number of all parameters
    mapping::Vector{Int}

    function ParamInfo(ukeys)
        d = Dict()
        mapping = Vector(undef, length(ukeys))
        for (i, k) in enumerate(ukeys)
            if k ∈ keys(d)
                mapping[i] = d[k]
                #@show i
                #@show mapping
            else 
                mapping[i] = length(d) + 1
                d[k] = length(d) + 1
                #@show mapping
                #@show i
            end
        end
        new(length(d), length(ukeys), mapping)
    end
end

"""
make redundant parameter(=θlong) from copact params(=θunique)
"""
function expand(p::ParamInfo, θunique)
    θlong = Vector{Float64}(undef, p.nparamlong)
    for i in eachindex(θlong)
        θlong[i] = θunique[p.mapping[i]]
    end
    return θlong
end


#read file.txt, and return list[Int]
function read_and_parse_float(file_name)
    x = Float64[]
    open(file_name, "r") do fp
        num_elm = parse(Int64,readline(fp))
        for i in 1:num_elm
            push!(x, parse(Float64, readline(fp)))
        end
    end
    return x
end

"""
読み込む変数が2つの場合
"""

function write_to_txt_2(file_name, x, y)
    open(file_name, "w") do fp
        for i = 1:length(x)
            println(fp, x[i], " ", real(y[i]))
        end
    end
end

"""
読み込む変数が一つの場合
"""

function write_to_txt_1(file_name, x)
    open(file_name, "w") do fp
        for i = 1:length(x)
            println(fp, x[i])
        end
    end
end


"""
make compact parameter information (Not used)
"""

function compact_paraminfo(keys)
    d = Dict()
    for (i, k) in enumerate(keys)
        if haskey(d, k) # floatが入るとややこしい。
            continue
        end
        #@show (i,k)
        d[k] = i
    end
    return d
end

function make_long_param_from_compact(keys, thetas)
    d = compact_paraminfo(keys)
    θ = zeros(Float64, length(thetas))
    for (ik, k) in enumerate(keys)
        #println(ik, k)
        i = d[k]
        θ[ik] = thetas[i] # d は辞書ではなく、配列にする。
    end
    return θ
end

"""
Given parameter information, make "long params" compact param sets
long params is redundant 
"""

function make_compact_params(thetas, keys)
    d = Dict()
    θunique = []
    for (i,k) in enumerate(keys)
        if haskey(d, k) # floatが入るとややこしい。
            continue
        end
        #@show (i,k)
        d[k] = i
        push!(θunique, thetas[d[k]]) 
    end
    return θunique
end