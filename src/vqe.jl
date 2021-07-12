using PyCall
using LinearAlgebra

"""
Compute ground state
"""
function solve_gs(ham_qubit::QubitOperator, circuit::VariationalQuantumCircuit, state0::QuantumState;
    theta_init=nothing, comm=nothing, maxiter=200, gtol=1e-5, verbose=false)
    scipy_opt = pyimport("scipy.optimize")

    if comm === nothing
        rank = 0
    else
        rank = MPI.Comm_rank(comm)
    end

    # Define a cost function
    function cost(theta_list)
        update_circuit_param!(circuit, theta_list)
        state = copy(state0)
        update_quantum_state!(circuit, state)
        return get_expectation_value(ham_qubit, state)
    end

    # Define the gradient of the cost function
    function grad_cost(theta_list)
        t1 = time_ns()
        if comm === nothing
            first_idx, size = 1, length(theta_list)
        else
            first_idx, size = distribute(length(theta_list), MPI.Comm_size(comm), MPI.Comm_rank(comm))
        end
        last_idx = first_idx + size - 1
        res = numerical_grad(cost, theta_list, first_idx=first_idx, last_idx=last_idx)
        if comm !== nothing
            res = MPI.Allreduce(res, MPI.SUM, comm)
        end
        t2 = time_ns()
        if verbose && rank == 0
            println("g: ", (t2-t1)*1e-9)
        end
        res
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
    opt = scipy_opt.minimize(cost, init_theta_list, method=method, callback=callback, jac=grad_cost,
        options=options)

    return cost_history, get_thetas(circuit)
end