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
export add_control_qubit_for_circuit, make_cotrolled_pauli_gate
export get_transition_amplitude_sampling_obs_real, get_transition_amplitude_sampling_obs_imag
export apply_qubit_ham_sampling!, apply_qubit_op_sampling! 
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
        #@show im_
        #@show  abs2(1.0 - (re_ + im_ * im))
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
    # temporaly
    #return z, opt_thetas
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

function add_control_qubit_for_circuit(circuit::Union{ParametricQuantumCircuit,VariationalQuantumCircuit}, control_index::Vector{Int}, total_num_qubits::Int)
    num_gate = get_gate_count(circuit)
    control_circuit = QulacsQuantumCircuit(total_num_qubits)
    for i in 1:num_gate
        gate_tmp = get_gate(circuit, i)
        gate_tmp = to_matrix_gate(gate_tmp)
        #println(i)
        #@show i
        add_control_qubit!(gate_tmp, control_index, 1)
        add_gate!(control_circuit,  gate_tmp)
    end
    return control_circuit
end


function make_cotrolled_pauli_gate(total_nqubits::Int, pauli_target_list::Vector{Int}, pauli_index_list::Vector{Int})
    ```
    Add controlled pauli-gate
    ancilla qubits corresponds to final qubits.
    ```
    pauli_gate = Pauli(pauli_target_list, pauli_index_list)
    pauli_gate = to_matrix_gate(pauli_gate)
    control_circuit = QulacsQuantumCircuit(total_nqubits)
    add_control_qubit!(pauli_gate, [total_nqubits], 1) #[nqubits+1]
    add_gate!(control_circuit, pauli_gate)
    return control_circuit
end


# imaginary-parts
function get_transition_amplitude_sampling_obs_real(
    state_augmented::QuantumState, 
    circuit_bra::VariationalQuantumCircuit, 
    ham::QubitOperator, 
    circuit_ket::VariationalQuantumCircuit; 
    nshots::Int=2^15)

    """
    <0| U_1^{dagger} P U_2 |0>
    """
    op_name_to_id = Dict("X" => 1, "Y" => 2, "Z" => 3)
    #total_num_qubits = nqubits + 1 
    total_num_qubits = get_n_qubit(state_augmented)
    control_index = [total_num_qubits]
    total_amp = 0 # total transition amplitude
    nqubits = total_num_qubits -1 
    
    # for loop of each term of Hamiltonian
    for (k, v) in terms_dict(ham) 
        #state = QulacsQuantumState(total_num_qubits)
        state = copy(state_augmented) 
        ###############
        #=
        function circuit_transition_amplitude()
        =#
        H_anci = H(total_num_qubits)
        update_quantum_state!(H_anci, state)
        X_anci = X(total_num_qubits)
        update_quantum_state!(X_anci, state)
    
        control_circuit1 = add_control_qubit_for_circuit(circuit_bra, control_index, total_num_qubits)
        update_quantum_state!(control_circuit1, state)
        update_quantum_state!(X_anci, state)
        control_circuit2 = add_control_qubit_for_circuit(circuit_ket, control_index, total_num_qubits)
        update_quantum_state!(control_circuit2, state)
        
        pauli_target_list = [x[1] for x in k] # [1,2,3]
        pauli_name = [x[2] for x in k] # ["Y", "Z", "Y"]
        pauli_index_list = map(x -> get(op_name_to_id, x, 0), pauli_name) ## ["Y", "Z", "Y"]-> [2, 3, 2]
        pauli_circuit = make_cotrolled_pauli_gate(total_num_qubits, pauli_target_list, pauli_index_list)
        update_quantum_state!(pauli_circuit, state)

        update_quantum_state!(H_anci, state)
        ###############
        #nshots = 2^24
        samples = state_sampling(state, nshots)
        estimated_amp = 0
        # This nqubits is without ancilla qubits
        
        mask = 2^nqubits # == Int("1" + "0" * (nqubits), 2)
        for s in samples
            bitcount = count_ones(s & mask) #1のビット数を数える。
            estimated_amp += (-1)^bitcount/nshots  #Z|0>->1|0>, Z|1>->-1|1>
        end
        total_amp += estimated_amp * v #pauli_coeffをかける
    end
    
    # ハミルトニアンのIの部分の項をここで足し込んでいる。
    if haskey(ham.pyobj.terms, ())
        #println("Special value found")
        #state = QulacsQuantumState(total_num_qubits)
        state = copy(state_augmented)
        H_anci = H(total_num_qubits)
        update_quantum_state!(H_anci, state)
        X_anci = X(total_num_qubits)
        update_quantum_state!(X_anci, state)
        #circuit = makecircuit1(nqubits)
        control_index = [total_num_qubits]
        control_circuit1 = add_control_qubit_for_circuit(circuit_bra, control_index, total_num_qubits)
        update_quantum_state!(control_circuit1, state)
        update_quantum_state!(X_anci, state)
        #circuit2 = makecircuit2(nqubits)
        control_circuit2 = add_control_qubit_for_circuit(circuit_ket, control_index, total_num_qubits)
        update_quantum_state!(control_circuit2, state)
        update_quantum_state!(H_anci, state)
        samples = state_sampling(state, nshots)
        estimated_amp = 0
        # This nqubits is without ancilla qubits
        mask = 2^nqubits # == Int("1" + "0" * (nqubits), 2)
        for s in samples
            bitcount = count_ones(s & mask) #1のビット数を数える。
            estimated_amp += (-1)^bitcount/nshots  #Z|0>->1|0>, Z|1>->-1|1>
        end
        total_amp += estimated_amp * (ham.pyobj.terms[()]) #real(pauli_coeff)をかける
    else
        #println("No special value found")
    end
    return total_amp
end


#=
function f(x, y, z=3, w=9)
    return 2x + y * z + w
end

f(1,2) <- こういう使い方ができる z = 3 の場合
f(1,2,3) <- これと１行上のは同じ
f(1,2, 9999) <- 動く
f(1,2,z=3) <- Python だとこういう書き方はできるが， Julia だとできない
=#

#=
function f(x, y; z=3, w=9)
    return 2x + y * z + w
end

f(1, 2, z=9) <- こういう書き方をしたい
f(1, 2) 
関数の仮引数と呼び出しがわの変数を対応させたい

z_no_atai = 999
f(1, 2, z=z_no_atai) <- このように呼び出す. 
f(1, 2, z_no_atai) # これはだめ

z = 999
f(1, 2; z) この書き方は許される． f(1,2,z=z) と同じ z=z のようにzをふたつ書くのはめんんどうだよね？　そういう時に便利です
=#

# imaginary-parts
function get_transition_amplitude_sampling_obs_imag(
    state_augmented::QuantumState, 
    circuit_bra::VariationalQuantumCircuit, 
    ham::QubitOperator, 
    circuit_ket::VariationalQuantumCircuit; 
    nshots::Int=2^15)

    """
    <0| U_1^{dagger} P U_2 |0>
    """
    op_name_to_id = Dict("X" => 1, "Y" => 2, "Z" => 3)
    #total_num_qubits = nqubits + 1
    total_num_qubits = get_n_qubit(state_augmented)
    control_index = [total_num_qubits]
    total_amp = 0 # total transition amplitude
    nqubits = total_num_qubits -1 
    
    # for loop of each term of Hamiltonian
    for (k, v) in terms_dict(ham) 
        #state = QulacsQuantumState(total_num_qubits)
        state = copy(state_augmented)
        H_anci = H(total_num_qubits)
        update_quantum_state!(H_anci, state)
        s_anci = S(total_num_qubits)
        update_quantum_state!(s_anci, state)
        X_anci = X(total_num_qubits)
        update_quantum_state!(X_anci, state)
    
        control_circuit1 = add_control_qubit_for_circuit(circuit_bra, control_index, total_num_qubits)
        update_quantum_state!(control_circuit1, state)
        update_quantum_state!(X_anci, state)
        control_circuit2 = add_control_qubit_for_circuit(circuit_ket, control_index, total_num_qubits)
        update_quantum_state!(control_circuit2, state)
        
        pauli_target_list = [x[1] for x in k] # [1,2,3]
        pauli_name = [x[2] for x in k] # ["Y", "Z", "Y"]
        pauli_index_list = map(x -> get(op_name_to_id, x, 0), pauli_name) ## ["Y", "Z", "Y"]-> [2, 3, 2]
        pauli_circuit = make_cotrolled_pauli_gate(total_num_qubits, pauli_target_list, pauli_index_list)
        update_quantum_state!(pauli_circuit, state)
        update_quantum_state!(H_anci, state)

        #nshots = 2^14
        samples = state_sampling(state, nshots)
        estimated_amp = 0
        # This nqubits is without ancilla qubits
        mask = 2^nqubits # == Int("1" + "0" * (nqubits), 2)
        for s in samples
            bitcount = count_ones(s & mask) #1のビット数を数える。
            estimated_amp += (-1)^bitcount/nshots  #Z|0>->1|0>, Z|1>->-1|1>
        end
        total_amp += estimated_amp * v #pauli_coeffをかける
    end

    if haskey(ham.pyobj.terms, ())
        #println("Special value found")
        #state = QulacsQuantumState(total_num_qubits)
        state = copy(state_augmented)
        H_anci = H(total_num_qubits)
        update_quantum_state!(H_anci, state)
        s_anci = S(total_num_qubits)
        update_quantum_state!(s_anci, state)
        X_anci = X(total_num_qubits)
        update_quantum_state!(X_anci, state)
        #circuit = makecircuit1(nqubits)
        control_index = [total_num_qubits]
        control_circuit1 = add_control_qubit_for_circuit(circuit_bra, control_index, total_num_qubits)
        update_quantum_state!(control_circuit1, state)

        update_quantum_state!(X_anci, state)

        #circuit2 = makecircuit2(nqubits)
        control_circuit2 = add_control_qubit_for_circuit(circuit_ket, control_index, total_num_qubits)
        update_quantum_state!(control_circuit2, state)

        update_quantum_state!(H_anci, state)
        samples = state_sampling(state, nshots)
        estimated_amp = 0
        # This nqubits is without ancilla qubits
        mask = 2^nqubits # == Int("1" + "0" * (nqubits), 2)
        for s in samples
            bitcount = count_ones(s & mask) #1のビット数を数える。
            estimated_amp += (-1)^bitcount/nshots  #Z|0>->1|0>, Z|1>->-1|1>
        end
        total_amp += estimated_amp * (ham.pyobj.terms[()]) #real(pauli_coeff)をかける
           
    else
        #println("No special value found")
    end

    return -total_amp
end


function apply_qubit_ham_sampling!(
    op::QubitOperator, 
    state_augmented::QuantumState, 
    circuit_bra::VariationalQuantumCircuit, 
    circuit_ket::VariationalQuantumCircuit; 
    minimizer=mk_scipy_minimize(), 
    verbose=true,
    comm=MPI_COMM_WORLD,
    nshots = 2^15,
    dx = 1e-1
    )
    if comm === nothing
        rank = 0
    else
        rank = MPI.Comm_rank(comm)
    end

    # her, antiher = divide_real_imag(op)
    scipy_opt = pyimport("scipy.optimize")
    function cost(thetas::Vector{Float64})

        update_circuit_param!(circuit_bra, thetas)
        #op_re_real = get_transition_amplitude_with_obs_sampling_re(circuit_ket, state0_bra, her, state_ket)
        #image op_re
        op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_bra, op, circuit_ket, nshots=nshots)
        op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_bra, op, circuit_ket; nshots)
        #op_re_ = op_re_re + im * op_re_im
        #op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_bra, antiher, circuit_ket)
        #op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_bra, antiher, circuit_ket)
        #op_im_ = op_im_re + im * op_im_im
        #op_transioton_amp = op_re_ + im * op_im_
        - abs2(op_re_re + op_re_im * im)
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
            jac = generate_numerical_grad(cost, verbose=verbose, dx=dx)
            if verbose
                println("Using parallelized numerical grad")
            end
        #end
        res = scipy_opt.minimize(cost, x0, method="BFGS",
            jac=jac, callback=callback, options=nothing) #options?
        res["x"]
    end
        
    
    #push!(cost_history, cost(x))

    thetas_init = get_thetas(circuit_bra)
    if comm !== nothing
        MPI.Bcast!(thetas_init, 0, comm)
    end
    opt_thetas = minimize(cost, thetas_init)
    #println("cost_opt=", cost(opt_thetas))

    #norm_right = sqrt(get_expectation_value(hermitian_conjugated(op) * op, state_ket))
    #if verbose
    #   println("norm_right", norm_right)
    #end

    update_circuit_param!(circuit_bra, opt_thetas)
    op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_bra, op, circuit_ket; nshots)
    op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_bra, op, circuit_ket; nshots)
    re__ = op_re_re + im * op_re_im
    #println("re__=", re__)
    #op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_bra, antiher, circuit_ket)
    #op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_bra, antiher, circuit_ket)
    #im__ = op_im_re + im * op_im_im
    #println("im__=", im__)
    #z = re__ + im__ * im
    #if verbose
    #    println("Match in apply_qubit_op!: ", z/norm_right)
    #end
    return re__
end


function apply_qubit_op_sampling!(
    op::QubitOperator,
    state_augmented::QuantumState,
    circuit_bra::VariationalQuantumCircuit, 
    circuit_ket::VariationalQuantumCircuit,
    minimizer=mk_scipy_minimize();
    verbose=true,
    comm=MPI_COMM_WORLD,
    nshots = 2^15,
    dx = 1e-1,
    maxiter = 300,
    gtol = 1e-7,
    disp = true
    )
    #@show "improve"
    if comm === nothing
        rank = 0
    else
        rank = MPI.Comm_rank(comm)
    end

    #@show "2"

    her, antiher = divide_real_imag(op) # op = jordan(c^dag) = (X + i Y)
     
    scipy_opt = pyimport("scipy.optimize")
    function cost(thetas::Vector{Float64})
        #=
        circuit_braを定義する？
        =#
        update_circuit_param!(circuit_bra, thetas)

        #op_re_real = get_transition_amplitude_with_obs_sampling_re(circuit_ket, state0_bra, her, state_ket)
        #image op_re

        op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_bra, her, circuit_ket; nshots)
        #op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_bra, her, circuit_ket; nshots)
        #op_re_ = op_re_re + im * op_re_im
        
        #op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_bra, antiher, circuit_ket; nshots)
        op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_bra, antiher, circuit_ket; nshots)
        #op_im_ = op_im_re + im * op_im_im
        #op_transioton_amp = op_re_ + im * op_im_
        #abs2(1.0 - (op_re_ + op_im_ * im))
        # fitting_normは常に実数
        # 虚部は、op_im*im
        # 合計で、op_re_re + im * 虚部
        #abs2(1.0 - (op_re_re + op_im_im * im * im))
        #@show op_re_re
        #@show op_im_im
        #@show op_re_re + op_im_im * im * im
        
        abs2(1.0 -  (op_re_re + op_im_im * im * im))
        #-abs2( (op_re_re + op_im_im * im * im))
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
        #@show "3"
        #if use_mpi
            jac = generate_numerical_grad(cost, verbose=verbose, dx=dx)
            #@show "4"
            if verbose
                println("Using parallelized numerical grad")
            end
            #@show "5"
        #end
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol, "disp" => disp)
        res = scipy_opt.minimize(cost, x0, method="BFGS", 
            jac=jac, callback=callback, options=options) #optio@[^ns?
        res["x"]
    end
        
    #=
    push!(cost_history, cost(x))
    if verbose && rank == 0
        println("iter ", length(cost_history), " ", cost_history[end])
    end
    =#

    thetas_init = get_thetas(circuit_bra)
    if comm !== nothing
        MPI.Bcast!(thetas_init, 0, comm)
    end
    opt_thetas = minimize(cost, thetas_init)
    #println("cost_opt=", cost(opt_thetas))

    #norm_right = sqrt(get_expectation_value(hermitian_conjugated(op) * op, state_ket))
    #if verbose
    #   println("norm_right", norm_right)
    #end

    update_circuit_param!(circuit_bra, opt_thetas)
    #@show "end"
    op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_bra, her, circuit_ket; nshots)
    #op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_bra, her, circuit_ket; nshots)
    #re__ = op_re_re + im * op_re_im
    #println("re__=", re__)
    #op_im_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_bra, antiher, circuit_ket; nshots)
    op_im_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_bra, antiher, circuit_ket; nshots)
    #im__ = op_im_re + im * op_im_im
    #println("im__=", im__)
    #z = re__ + im__ * im
    z = op_re_re + op_im_im * im * im
    #if verbose
    #    println("Match in apply_qubit_op!: ", z/norm_right)
    #end
    return z
end

