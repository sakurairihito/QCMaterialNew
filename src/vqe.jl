using PyCall
using LinearAlgebra

#using SciPy: SciPy

"""
Compute ground state
"""

function solve_gs(ham_qubit::QubitOperator, circuit::VariationalQuantumCircuit, state0::QuantumState;
    theta_init=nothing, comm=MPI_COMM_WORLD, maxiter=200, gtol=1e-5, verbose=false)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end
    if comm === nothing
        rank = 0
    else
        rank = MPI.Comm_rank(comm)
    end
    # これを使わずに SciPy.optimize でよい
    scipy_opt = pyimport("scipy.optimize")

    ## parameterinfo = [((1, 1), 1), ((3, 1), 2), ((1, 2, 1, 1), 5), ((1, 2, 1, 1), 6), ((1, 1), 7), ((3, 1), 8)]
    ## parameterref = [(1, 1), (3, 1), (1, 2, 1, 1)]
    ## paraminit =  [1.0, 1.2, 1.3 ,,,,,]
    #update_param = [] # 同じ軌道のペアのパラメータの値を揃える。
    #theta_init = [update されたパラメータ]
    
    # Define a cost function
    function cost(theta_list)
        # 与えられたtheta_listを変える必要がある
        # theta_list = [1.0, 2.0,,, 100, 101,,,]
        # ここで軌道のペアが同じパラメータの値は揃えるようにtheta_listがupdateされる。

        update_circuit_param!(circuit, theta_list)
        state = copy(state0)
        update_quantum_state!(circuit, state)
        return get_expectation_value(ham_qubit, state)
    end

    if theta_init === nothing
        theta_init = rand(size(circuit.theta_offsets)[1])
    end
    if comm !== nothing
        # Make sure all processes use the same initial values
        MPI.Bcast!(theta_init, 0, comm)
    end
    cost_history = Float64[] #コスト関数の箱
    init_theta_list = theta_init
    push!(cost_history, cost(init_theta_list))
    
    method = "BFGS"
    options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol)
    function callback(x)
        push!(cost_history, cost(x))
        if verbose && rank == 0
            println("iter ", length(cost_history), " ", cost_history[end])
        end
    end
    # opt = SciPy.optimize.minimize(...)
    opt = scipy_opt.minimize(cost, init_theta_list, method=method,
        callback=callback, jac=generate_numerical_grad(cost),
        options=options)

    return cost_history, get_thetas(circuit)
end


function solve_gs_kucj(ham_qubit::QubitOperator, circuit::VariationalQuantumCircuit, state0::QuantumState, keys;
    theta_init=nothing, comm=MPI_COMM_WORLD, maxiter=200, gtol=1e-5, verbose=false)
    # theta_init => theta_unique (compact parameters) 
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end

    if comm === nothing
        rank = 0
    else
        rank = MPI.Comm_rank(comm)
    end
    # これを使わずに SciPy.optimize でよい
    scipy_opt = pyimport("scipy.optimize")
    ## parameterinfo = [((1, 1), 1), ((3, 1), 2), ((1, 2, 1, 1), 5), ((1, 2, 1, 1), 6), ((1, 1), 7), ((3, 1), 8)]
    ## parameterref = [(1, 1), (3, 1), (1, 2, 1, 1)]
    ## theta_init =  [1.0, 1.2, 1.3 ,,,,,]
    #update_param = [] # 同じ軌道のペアのパラメータの値を揃える。
    #theta_init_update = [update されたパラメータ]
    # Define a cost function

    # 冗長なパラメータが代入される(θlong)
    function cost(theta_list)
        update_circuit_param!(circuit, theta_list)
        state = copy(state0)
        update_quantum_state!(circuit, state)
        return get_expectation_value(ham_qubit, state)
    end
    
    pinfo = ParamInfo(keys)
    # expandも別ファイル
    #θunique = rand(pinfo.nparam)
    #θlong = expand(θunique)
    
    #function cost_tmp(θunique)
    #    θ_long = expand(θunique)
    #    return cost(θ_long)
    #end
    cost_tmp(θunique) = cost(expand(pinfo, θunique))    
    #if theta_init === nothing
    #    theta_init = rand(size(circuit.theta_offsets)[1])
    #end

    if comm !== nothing
        # Make sure all processes use the same initial values
        MPI.Bcast!(expand(pinfo, theta_init), 0, comm)
    end

    cost_history = Float64[] #コスト関数の箱
    #init_theta_list = theta_init
    push!(cost_history, cost_tmp(theta_init))
    
    method = "L-BFGS-B"
    options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol)

    function callback(x)
        push!(cost_history, cost_tmp(x))
        if verbose && rank == 0
            println("iter ", length(cost_history), " ", cost_history[end])
        end
    end
    # opt = SciPy.optimize.minimize(...)
    # ERROR: JULIA: Invalid length of thetas!
    # θunique NO
    opt = scipy_opt.minimize(cost_tmp, theta_init, method=method,
        callback=callback, jac=generate_numerical_grad(cost_tmp),
        options=options)
    return cost_history, get_thetas(circuit) 
    #return cost_history, get_thetas(circuit)
end
