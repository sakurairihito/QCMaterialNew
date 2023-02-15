# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     main_language: julia
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
export apply_qubit_op!, get_transition_amplitude_with_obs, apply_ham!, apply_qubit_ham!
export divide_real_imag
# %%
"""
Divide a qubit operator into the hermite and antihermite parts.
"""

# %%
divide_real_imag(op::QubitOperator) = 
    (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im

# %%
"""
Apply a qubit operator op to |state_ket> and fit the result with
circuit * |state_bra>.
The circuit object will be updated on exit.
The squared norm of op * |state_ket>  will be returned.
state0_bra will not be modified.
"""

function apply_qubit_op!(
    op::QubitOperator,
    state_ket::QuantumState,
    circuit::VariationalQuantumCircuit, 
    state0_bra::QuantumState;
    minimizer=mk_scipy_minimize(),
    verbose=true,
    comm=MPI_COMM_WORLD
    )

    if comm === nothing
        rank = 0
    else
        rank = MPI.Comm_rank(comm)
    end

    her, antiher = divide_real_imag(op)
    scipy_opt = pyimport("scipy.optimize")
    function cost(thetas::Vector{Float64})
        update_circuit_param!(circuit, thetas)
        re_ = get_transition_amplitude_with_obs(circuit, state0_bra, her, state_ket)
        im_ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
        abs2(1.0 - (re_ + im_ * im))
    end
          
    cost_history = []
    function callback(x)
        push!(cost_history, cost(x))
        #@show rank
        if verbose && rank == 0
            println("iter ", length(cost_history), " ", cost_history[end])
        end
    end
    
    function minimize(cost, x0)
    jac = nothing
        #if use_mpi
            jac = generate_numerical_grad(cost, verbose=verbose)
            if verbose
                println("Using parallelized numerical grad")
            end
        #end
        res = scipy_opt.minimize(cost, x0, method="BFGS",
            jac=jac, callback=callback, options=nothing) #options?
        res["x"]
    end
        
    #=
    push!(cost_history, cost(x))
    if verbose && rank == 0
        println("iter ", length(cost_history), " ", cost_history[end])
    end
    =#

    thetas_init = get_thetas(circuit)
    if comm !== nothing
        MPI.Bcast!(thetas_init, 0, comm)
    end
    opt_thetas = minimize(cost, thetas_init)
    println("cost_opt=", cost(opt_thetas))
    norm_right = sqrt(get_expectation_value(hermitian_conjugated(op) * op, state_ket))
    if verbose
       println("norm_right",norm_right)
    end

    update_circuit_param!(circuit, opt_thetas)
    re__ = get_transition_amplitude_with_obs(circuit, state0_bra, her, state_ket)
    println("re_=", re__)
    im__ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
    println("im_=", im__)
    z = re__ + im__ * im
    if verbose
        println("Match in apply_qubit_op!: ", z/norm_right)
    end
    return z
end


# %%
function apply_qubit_ham!(
    op::QubitOperator,
    state_ket::QuantumState,
    circuit::VariationalQuantumCircuit, state0_bra::QuantumState;
    minimizer=mk_scipy_minimize(),
    verbose=true,
    comm=MPI_COMM_WORLD
    )

    if comm === nothing
        rank = 0
    else
        rank = MPI.Comm_rank(comm)
    end

    her, antiher = divide_real_imag(op)
    scipy_opt = pyimport("scipy.optimize")
    function cost(thetas::Vector{Float64})
        update_circuit_param!(circuit, thetas)
        re_ = get_transition_amplitude_with_obs(circuit, state0_bra, her, state_ket)
        im_ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
        - abs((re_ ))
    end
          
    cost_history = []
    function callback(x)
        push!(cost_history, cost(x))
        #@show rank
        if verbose && rank == 0
            println("iter ", length(cost_history), " ", cost_history[end])
        end
    end
    
    function minimize(cost, x0)
    jac = nothing
        #if use_mpi
            jac = generate_numerical_grad(cost, verbose=verbose)
            if verbose
                println("Using parallelized numerical grad")
            end
        #end
        res = scipy_opt.minimize(cost, x0, method="BFGS",
            jac=jac, callback=callback, options=nothing) #options?
        res["x"]
    end
        
    #=
    push!(cost_history, cost(x))
    if verbose && rank == 0
        println("iter ", length(cost_history), " ", cost_history[end])
    end
    =#

    thetas_init = get_thetas(circuit)
    if comm !== nothing
        MPI.Bcast!(thetas_init, 0, comm)
    end
    opt_thetas = minimize(cost, thetas_init)
    println("cost_opt=", cost(opt_thetas))
    norm_right = sqrt(get_expectation_value(hermitian_conjugated(op) * op, state_ket))
    if verbose
       println("norm_right",norm_right)
    end

    update_circuit_param!(circuit, opt_thetas)
    re__ = get_transition_amplitude_with_obs(circuit, state0_bra, her, state_ket)
    println("re_=", re__)
    im__ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
    println("im_=", im__)
    z = re__ + im__ * im
    if verbose
        println("Match in apply_qubit_op!: ", z/norm_right)
    end
    return z
end

# %%
#=
function apply_qubit_ham!(
    op::QubitOperator,
    state_ket::QuantumState,
    circuit::VariationalQuantumCircuit, state0_bra::QuantumState;
    minimizer=mk_scipy_minimize(),
    verbose=true,
    comm=MPI_COMM_WORLD
    )
    #her, antiher = divide_real_imag(op)

    function cost(thetas::Vector{Float64})
        update_circuit_param!(circuit, thetas)
        re_ = get_transition_amplitude_with_obs(circuit, state0_bra, op, state_ket)
        #im_ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
        #println("transition_amplitude_value=", -abs(re_))
        - abs((re_ ))
    end
   
    thetas_init = get_thetas(circuit)
    if comm !== nothing
        MPI.Bcast!(thetas_init, 0, comm)
    end
    opt_thetas = minimizer(cost, thetas_init)
    println("cost_opt=", cost(opt_thetas))
    norm_right = sqrt(get_expectation_value(hermitian_conjugated(op) * op, state_ket))
    if verbose
       println("norm_right",norm_right)
    end

    update_circuit_param!(circuit, opt_thetas)
    re__ = get_transition_amplitude_with_obs(circuit, state0_bra, op, state_ket)
    #im__ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
    #z = re__ + im__ * im
    z = re__
    if verbose
        println("Match in apply_qubit_op!: ", z/norm_right)
    end
    return z
end
=#

# %%
"""
Compute <state_bra| circuit^+ obs |state_ket>, where obs is a hermite observable.
"""
function get_transition_amplitude_with_obs(
    circuit::VariationalQuantumCircuit, 
    state0_bra::QuantumState,
    op::QubitOperator,
    state_ket::QuantumState)
    state_bra = copy(state0_bra)
    update_quantum_state!(circuit, state_bra)
    return get_transition_amplitude(op, state_bra, state_ket)
end


# %%
"""
Apply a Hamiltonian to |state_ket> and fit the result with
circuit * |state_bra>.
The circuit object will be updated on exit.
The squared norm of op * |state_ket>  will be returned.
state0_bra will not be modified.
"""
function apply_ham!(
    op::QubitOperator,
    state_ket::QuantumState,
    circuit::VariationalQuantumCircuit, state0_bra::QuantumState;
    minimizer=mk_scipy_minimize(),
    verbose=true,
    comm=MPI_COMM_WORLD
    )
    #her, antiher = divide_real_imag(op)

    #options = Dict("disp" => verbose, "maxiter" => 300, "gtol" => 1e-8)

    function cost(thetas::Vector{Float64})
        update_circuit_param!(circuit, thetas)
        # amplitude = <bra(\theta)| H |ket>
        amplitude = get_transition_amplitude_with_obs(circuit, state0_bra, op, state_ket)
        abs2(1.0 - (amplitude))
    end
   
    thetas_init = get_thetas(circuit)
    if comm !== nothing
        MPI.Bcast!(thetas_init, 0, comm)
    end
    opt_thetas = minimizer(cost, thetas_init)
    # sqrt(<ket| H H |ket>)
    norm_right = sqrt(get_expectation_value(hermitian_conjugated(op) * op, state_ket))
    if verbose
       println("norm_of_H*|ket>=", norm_right)
    end

    update_circuit_param!(circuit, opt_thetas)

    square_norm = get_transition_amplitude_with_obs(circuit, state0_bra, op, state_ket)
    #im__ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
    #z = re__ + im__ * im
    if verbose
        println("Match in apply_qubit_op!: ", square_norm/norm_right)
    end
end

# %%

# %%
function apply_qubit_op_kucj!(
    op::QubitOperator,
    state_ket::QuantumState,
    circuit::VariationalQuantumCircuit, state0_bra::QuantumState;
    minimizer=mk_scipy_minimize(),
    verbose=true,
    comm=MPI_COMM_WORLD
    )

    if comm === nothing
        rank = 0
    else
        rank = MPI.Comm_rank(comm)
    end

    her, antiher = divide_real_imag(op)
    scipy_opt = pyimport("scipy.optimize")

    function cost(thetas::Vector{Float64})
        update_circuit_param!(circuit, thetas)
        re_ = get_transition_amplitude_with_obs(circuit, state0_bra, her, state_ket)
        im_ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
        abs2(1.0 - (re_ + im_ * im))
    end

    pinfo = ParamInfo(keys)
    # コンパクトなパラメータを受け取って、冗長なパラメータに直して、それをコスト関数に代入する。
    
    cost_tmp(θunique) = cost(expand(pinfo, θunique))

    if comm !== nothing
        # Make sure all processes use the same initial values
        MPI.Bcast!(expand(pinfo, theta_init), 0, comm)
    end


    cost_history = []
    function callback(x)
        push!(cost_history, cost_tmp(x))
        #@show rank
        if verbose && rank == 0
            println("iter ", length(cost_history), " ", cost_history[end])
        end
    end
    
    function minimize(cost, x0)
    jac = nothing
        #if use_mpi
            jac = generate_numerical_grad(cost, verbose=verbose)
            if verbose
                println("Using parallelized numerical grad")
            end
        #end
        res = scipy_opt.minimize(cost, x0, method="BFGS",
            jac=jac, callback=callback, options=nothing) #options?
        res["x"]
    end
        
    #=
    push!(cost_history, cost(x))
    if verbose && rank == 0
        println("iter ", length(cost_history), " ", cost_history[end])
    end
    =#

    thetas_init = get_thetas(circuit)
    if comm !== nothing
        MPI.Bcast!(thetas_init, 0, comm)
    end
    opt_thetas = minimize(cost_tmp, thetas_init)
    println("cost_opt=", cost_tmp(opt_thetas))
    norm_right = sqrt(get_expectation_value(hermitian_conjugated(op) * op, state_ket))
    if verbose
       println("norm_right",norm_right)
    end

    update_circuit_param!(circuit, opt_thetas)
    re__ = get_transition_amplitude_with_obs(circuit, state0_bra, her, state_ket)
    println("re_=", re__)
    im__ = get_transition_amplitude_with_obs(circuit, state0_bra, antiher, state_ket)
    println("im_=", im__)
    z = re__ + im__ * im
    if verbose
        println("Match in apply_qubit_op!: ", z/norm_right)
    end
    return z
end

