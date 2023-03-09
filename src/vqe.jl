using PyCall
using LinearAlgebra
export get_expected_value_sampling, solve_gs_sampling
#using SciPy: SciPy
export solve_gs

"""
Compute ground state
"""

function solve_gs(
    ham_qubit::QubitOperator,
    circuit::VariationalQuantumCircuit,
    state0::QuantumState;
    theta_init = nothing,
    comm = MPI_COMM_WORLD,
    maxiter = 300,
    gtol = 1e-7,
    verbose = false,
)
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
        theta_init = rand(get_thetas(circuit))
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
    opt = scipy_opt.minimize(
        cost,
        init_theta_list,
        method = method,
        callback = callback,
        jac = generate_numerical_grad(cost),
        options = options,
    )
    
    return cost_history, get_thetas(circuit)
end



function solve_gs_kucj(
    ham_qubit::QubitOperator,
    circuit::VariationalQuantumCircuit,
    state0::QuantumState,
    keys;
    theta_init = nothing,
    comm = MPI_COMM_WORLD,
    maxiter = 300,
    gtol = 1e-8,
    verbose = false,
)
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
    # 冗長なパラメータが代入される(θlong)
    function cost(theta_list)
        update_circuit_param!(circuit, theta_list)
        state = copy(state0)
        update_quantum_state!(circuit, state)
        return get_expectation_value(ham_qubit, state)
    end

    pinfo = ParamInfo(keys)
    cost_tmp(θunique) = cost(expand(pinfo, θunique))

    if comm !== nothing
        # Make sure all processes use the same initial values
        MPI.Bcast!(expand(pinfo, theta_init), 0, comm)
    end

    cost_history = Float64[] #コスト関数の箱
    #init_theta_list = theta_init
    push!(cost_history, cost_tmp(theta_init))

    method = "BFGS"
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
    opt = scipy_opt.minimize(
        cost_tmp,
        theta_init,
        method = method,
        callback = callback,
        jac = generate_numerical_grad(cost_tmp),
        options = options,
    )
    return cost_history, get_thetas(circuit)
    #return cost_history, get_thetas(circuit)
end


function get_expected_value_sampling(ham, state; nshots=2^15)
    n_qubit = get_n_qubit(state)
    total_energy = 0
    for (k, v) in terms_dict(ham) 
        #circuit_rot = QulacsParametricQuantumCircuit(n_qubit) 
        #circuit_rot = UCCQuantumCircuit(n_qubit)  
        
        #update_quantum_state!(h_gate, state)
        for (index, op_name) in k 
            if op_name == "Y" 
                Sdag_gate = Sdag(index)
                update_quantum_state!(Sdag_gate, state) 
                
                H_gate = H(index)
                update_quantum_state!(H_gate, state) 
            elseif op_name == "X" 
                H_gate = H(index)
                update_quantum_state!(H_gate, state) 
            end 
        end 
 
        # state に circuit_rotをかける 
        

        # 期待値の測定を行う 
        samples = state_sampling(state, nshots) 
        estimated_energy = 0
        mask = UInt(sum([1*2^(x[1] - 1) for x in k]))  
        for s in samples
            #@show s
            bitcount = count_ones(s & mask) #1のビット数を数える。
            estimated_energy += (-1)^bitcount/nshots #Z|0>->1|0>, Z|1>->-1|1>
        end
        
        # pauli_coeffをかける 
        estimated_energy = estimated_energy * (v) 
        total_energy += estimated_energy

        # 上でかけた操作をもとに戻す。
        # circuit_rot_inv = QulacsParametricQuantumCircuit(n_qubit)
        # dx=1e-8
        #circuit = UCCQuantumCircuit(n_qubit)
        for (index, op_name) in k
            if op_name == "Y"
                H_gate = H(index)
                update_quantum_state!(H_gate, state) 
                Sdag_gate = S(index)
                update_quantum_state!(Sdag_gate, state) 
            elseif op_name == "X"
                H_gate = H(index)
                update_quantum_state!(H_gate, state) 
            end
        end
    end
    # Hamiltonianの定数部分を足す
    #
    #@show total_energy
    if haskey(ham.pyobj.terms, ())
        total_energy += (ham.pyobj.terms[()])   
    end

    return real(total_energy)
end


function solve_gs_sampling(
    ham_qubit::QubitOperator,
    circuit::VariationalQuantumCircuit,
    state0::QuantumState;
    theta_init = nothing,
    comm = MPI_COMM_WORLD,
    maxiter = 300,
    gtol = 1e-7,
    verbose = false,
    nshots = 2^15,
    dx=1e-1
)
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


    # Define a cost function
    function cost(theta_list)
        update_circuit_param!(circuit, theta_list)
        state = copy(state0)
        update_quantum_state!(circuit, state)
        
        # 微分を計算するためにコスト関数を計算するとき→ allreduceしない
        res = get_expected_value_sampling(ham_qubit, state, nshots=nshots)
        
        #コスト関数の値のみが欲しいとき→全プロセスで同じ値が欲しいので、allreduceしてプロセス数で割る
        # 最適化の途中で、コスト関数の値だけを計算するときがあるんですよね、1次元探索とか。
        # その時はallreduceして全プロセスで値を一致させないと行けない。 
        if comm === nothing
            
            res = MPI.Allreduce(res, MPI.SUM, comm)
            res = res/MPI.Comm_size(comm)
        end

        return res
    end

    if theta_init === nothing
        theta_init = rand((num_theta(circuit)))
    end
    if comm !== nothing
    #    # Make sure all processes use the same initial values
        MPI.Bcast!(theta_init, 0, comm)
    end
    cost_history = Float64[] #コスト関数の箱
    init_theta_list = theta_init
    push!(cost_history, cost(init_theta_list))

    method = "BFGS"
    options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol)
    function callback(x)
        #@show "callback begin"
        push!(cost_history, cost(x))
        if verbose && rank == 0
            println("iter ", length(cost_history), " ", cost_history[end])
        end
        #@show "callback end"
    end
    # opt = SciPy.optimize.minimize(...)
    
    opt = scipy_opt.minimize(
        cost,
        init_theta_list,
        method = method,
        callback = callback,
        jac = generate_numerical_grad(cost,  comm=comm, verbose=true, dx=dx),
        options = options,
    )
    #println("after opt") 
    #println(cost_history)
    #println(cost_history[end])
    return cost_history, get_thetas(circuit)
end