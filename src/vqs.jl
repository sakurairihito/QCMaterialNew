using LinearAlgebra
using HDF5
using Random
#Aの計算
"""
Compute <phi (theta_bra) | phi(theta_ket)>
"""
function overlap(vc::VariationalQuantumCircuit, state0::QulacsQuantumState,
    thetas_left::Vector{Float64}, thetas_right::Vector{Float64})

    circ_tmp = copy(vc)

    # Compute state_left
    update_circuit_param!(circ_tmp, thetas_left)
    state_left = copy(state0)

    update_quantum_state!(circ_tmp, state_left)

    #println("state_left_overlap=", state_left)

    # Compute state_right
    update_circuit_param!(circ_tmp, thetas_right)
    state_right = copy(state0)
    update_quantum_state!(circ_tmp, state_right)
    #println("state_right_overlap=", state_right)

    res = inner_product(state_left, state_right)
    res
end

function compute_B(vc::VariationalQuantumCircuit, state0::QulacsQuantumState, delta_theta=1e-8;
    comm=MPI_COMM_WORLD)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end

    num_thetas = num_theta(vc)
    thetas = get_thetas(vc)
    upper_triangle_size = (num_thetas^2 + num_thetas) ÷ 2 

    B = zeros(Complex{Float64}, num_thetas, num_thetas)
    local_start, local_size = distribute(upper_triangle_size, mpisize(comm), mpirank(comm))
    idx = 1
    for i in 1:num_thetas
        thetas_i = copy(thetas)
        thetas_i2 = copy(thetas)
        thetas_i[i] += delta_theta
        thetas_i2[i] -= delta_theta
        for j in i:num_thetas
            if local_start <= idx < local_start + local_size
                thetas_j = copy(thetas)
                thetas_j[j] += delta_theta
                thetas_j2 = copy(thetas)
                thetas_j2[j] -= delta_theta
                B[i,j] -= real(
                 ((overlap(vc, state0, thetas_i, thetas)
                - overlap(vc, state0, thetas_i2, thetas))/2*delta_theta)
                *((overlap(vc, state0, thetas, thetas_j)
                - overlap(vc, state0, thetas, thetas_j2))/2*delta_theta)
            )
            B[j, i] = B[i, j]
            end
            idx += 1
        end
    end
    if comm === nothing
      return B
    else
      return Allreduce(B, MPI.SUM, comm)
    end
end

function diff_state(vc::VariationalQuantumCircuit, state0::QulacsQuantumState,thetas, thetas_delta)
    circ_tmp = copy(vc)
    update_circuit_param!(circ_tmp, thetas)
    state_tmp = copy(state0)
    update_quantum_state!(circ_tmp, state_tmp)

    update_circuit_param!(circ_tmp, thetas_delta)
    state_tmp_delta = copy(state0)
    update_quantum_state!(circ_tmp, state_tmp_delta)
    #diff
    res = (get_vector(state_tmp_delta) - get_vector(state_tmp))
    res
end

function compute_B2x(vc::VariationalQuantumCircuit, state0::QulacsQuantumState, delta_theta=1e-8;
    comm=MPI_COMM_WORLD)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end

    num_thetas = num_theta(vc)
    thetas = get_thetas(vc)
    upper_triangle_size = (num_thetas^2 + num_thetas) ÷ 2 

    B = zeros(Complex{Float64}, num_thetas, num_thetas)
    local_start, local_size = distribute(upper_triangle_size, mpisize(comm), mpirank(comm))
    idx = 1
    for i in 1:num_thetas
        thetas_i = copy(thetas)
        thetas_i2 = copy(thetas)
        thetas_i[i] += delta_theta
        thetas_i2[i] -= delta_theta
        for j in i:num_thetas
            if local_start <= idx < local_start + local_size
                thetas_j = copy(thetas)
                thetas_j[j] += delta_theta
                thetas_j2 = copy(thetas)
                thetas_j2[j] -= delta_theta
                #computeδ|φ>/δθ
                B_right_dif = diff_state(vc, state0, thetas, thetas_i)/delta_theta
                #B_right = inner_product(state, B_right_dif)
                println("B_right_dif_$i$j=",B_right_dif)
                B_left_dif = B_right_dif'
                println("B_left_dif_$i$j=", B_left_dif)
                B[i,j] += real(i*j)
                #println("B[$i $j]=", B[i,j])
            B[j, i] = B[i, j]
            end
            idx += 1
        end
    end
    if comm === nothing
      return B
    else
      return Allreduce(B, MPI.SUM, comm)
    end
end


function compute_B2(vc::VariationalQuantumCircuit, state0::QulacsQuantumState, delta_theta=1e-8;
    comm=MPI_COMM_WORLD)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end

    num_thetas = num_theta(vc)
    thetas = get_thetas(vc)
    upper_triangle_size = (num_thetas^2 + num_thetas) ÷ 2 

    B = zeros(Complex{Float64}, num_thetas, num_thetas)
    local_start, local_size = distribute(upper_triangle_size, mpisize(comm), mpirank(comm))
    idx = 1
    for i in 1:num_thetas
        thetas_i = copy(thetas)
        thetas_i2 = copy(thetas)
        thetas_i[i] += delta_theta
        thetas_i2[i] -= delta_theta
        for j in i:num_thetas
            if local_start <= idx < local_start + local_size
                thetas_j = copy(thetas)
                thetas_j[j] += delta_theta
                thetas_j2 = copy(thetas)
                thetas_j2[j] -= delta_theta

                B_right = ((overlap(vc, state0, thetas, thetas_j)
                - overlap(vc, state0, thetas, thetas))/delta_theta)
                println("B_right_$i$j=",B_right)
                B_left = B_right'
                println("B_left_$i$j=", B_left)
                B[i,j] += real(B_left*B_right)
                println("B[$i $j]=", B[i,j])
            B[j, i] = B[i, j]
            end
            idx += 1
        end
    end
    if comm === nothing
      return B
    else
      return Allreduce(B, MPI.SUM, comm)
    end
end



function compute_F(vc::VariationalQuantumCircuit, state0::QulacsQuantumState, delta_theta=1e-8;
    comm=MPI_COMM_WORLD)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end

    num_thetas = num_theta(vc)
    thetas = get_thetas(vc)
    upper_triangle_size = (num_thetas^2 + num_thetas) ÷ 2 

    F = zeros(Complex{Float64}, num_thetas, num_thetas)
    local_start, local_size = distribute(upper_triangle_size, mpisize(comm), mpirank(comm))
    idx = 1
    for i in 1:num_thetas
        thetas_i = copy(thetas)
        thetas_i2 = copy(thetas)
        thetas_i[i] += delta_theta
        thetas_i2[i] -= delta_theta
        for j in i:num_thetas
            if local_start <= idx < local_start + local_size
                thetas_j = copy(thetas)
                thetas_j[j] += delta_theta
                thetas_j2 = copy(thetas)
                thetas_j2[j] -= delta_theta
                F[i, j] = real(
                          overlap(vc, state0, thetas_i, thetas_j)
                        - overlap(vc, state0, thetas_i, thetas, )
                        - overlap(vc, state0, thetas,   thetas_j)
                        + overlap(vc, state0, thetas,   thetas, )
                    )/delta_theta^2
                F[i,j] -= real(
                 ((overlap(vc, state0, thetas_i, thetas)
                - overlap(vc, state0, thetas_i2, thetas))/2*delta_theta)
                *((overlap(vc, state0, thetas, thetas_j)
                - overlap(vc, state0, thetas, thetas_j2))/2*delta_theta)
            )
            #    F[i,j] -= real(
            #     ((overlap(vc, state0, thetas_i, thetas)
            #    - overlap(vc, state0, thetas, thetas))/delta_theta)
            #    *((overlap(vc, state0, thetas, thetas_j)
            #    - overlap(vc, state0, thetas, thetas))/delta_theta)
            #    )
            #    F[i,j] -= real(
            #        overlap(vc, state0, thetas_i, thetas) * overlap(vc, state0, thetas, thetas_j)
            #       -overlap(vc, state0, thetas_i, thetas) * overlap(vc, state0, thetas, thetas)
            #       -overlap(vc, state0, thetas, thetas) * overlap(vc, state0, thetas, thetas_j)
            #       +overlap(vc, state0, thetas, thetas) * overlap(vc, state0, thetas, thetas) 
            #       )/delta_theta^2
            F[j, i] = F[i, j]
            end
            idx += 1
            #if MPI_rank == 0
                #println("timing in compute_A: $(1e-9*(t2-t1))")
            #end
        end
    end
    if comm === nothing
      return F
    else
      return Allreduce(F, MPI.SUM, comm)
    end
end

function compute_B3(vc::VariationalQuantumCircuit, state0::QulacsQuantumState, delta_theta=1e-8;
    comm=MPI_COMM_WORLD)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end

    num_thetas = num_theta(vc)
    thetas = get_thetas(vc)
    upper_triangle_size = (num_thetas^2 + num_thetas) ÷ 2

    B = zeros(Complex{Float64}, num_thetas, num_thetas)
    local_start, local_size = distribute(upper_triangle_size, mpisize(comm), mpirank(comm))
    idx = 1
    for i in 1:num_thetas
        thetas_i = copy(thetas)
        thetas_i[i] += delta_theta
        for j in i:num_thetas
            if local_start <= idx < local_start + local_size
                thetas_j = copy(thetas)
                thetas_j[j] += delta_theta
                B[i, j] =( 
                  (overlap(vc, state0, thetas_i, thetas)
                  - overlap(vc, state0, thetas, thetas)) *
                  (overlap(vc, state0, thetas, thetas_j)
                  - overlap(vc, state0, thetas, thetas))
              /delta_theta^2)
            B[j, i] = B[i, j] 
            end
            idx += 1
        end
    end
    if comm === nothing
      return B
    else
      return Allreduce(B, MPI.SUM, comm)
    end
end



function compute_A(vc::VariationalQuantumCircuit, state0::QulacsQuantumState, delta_theta=1e-8;
    comm=MPI_COMM_WORLD)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end

    num_thetas = num_theta(vc)
    thetas = get_thetas(vc)
    upper_triangle_size = (num_thetas^2 + num_thetas) ÷ 2

    A = zeros(Complex{Float64}, num_thetas, num_thetas)
    local_start, local_size = distribute(upper_triangle_size, mpisize(comm), mpirank(comm))
    idx = 1
    for i in 1:num_thetas
        thetas_i = copy(thetas)
        thetas_i[i] += delta_theta
        for j in i:num_thetas
            if local_start <= idx < local_start + local_size
                thetas_j = copy(thetas)
                thetas_j[j] += delta_theta
                A[i, j] = real(
                          overlap(vc, state0, thetas_i, thetas_j)
                        - overlap(vc, state0, thetas_i, thetas, )
                        - overlap(vc, state0, thetas,   thetas_j)
                        + overlap(vc, state0, thetas,   thetas, )
                    )/delta_theta^2
                A[j, i] = A[i, j]
            end
            idx += 1
            #if MPI_rank == 0
                #println("timing in compute_A: $(1e-9*(t2-t1))")
            #end
        end
    end
    if comm === nothing
      return A
    else
      return Allreduce(A, MPI.SUM, comm)
    end
end


#Cの計算
"""
Compute <phi (theta_bra) |H| phi(theta_ket)>
"""
function transition(op::OFQubitOperator, vc::VariationalQuantumCircuit, state0::QulacsQuantumState,
    thetas_left::Vector{Float64}, thetas_right::Vector{Float64})

    circ_tmp = copy(vc)

    # Compute state_left
    update_circuit_param!(circ_tmp, thetas_left)
    state_left = copy(state0)
    update_quantum_state!(circ_tmp, state_left)

    # Compute state_right
    update_circuit_param!(circ_tmp, thetas_right)
    state_right = copy(state0)
    update_quantum_state!(circ_tmp, state_right)
    
    # Compute <state_right|H|state_right>
    res = get_transition_amplitude(op, state_left, state_right)
    res
end


function compute_C(op::OFQubitOperator, vc::VariationalQuantumCircuit,
    state0::QulacsQuantumState, delta_theta=1e-8;comm=MPI_COMM_WORLD)
    if is_mpi_on && comm === nothing
        error("comm must be given whn mpi is one!")
    end
    num_thetas = num_theta(vc)
    thetas = get_thetas(vc)

    C = zeros(Complex{Float64}, num_thetas)
    local_start, local_size = distribute(num_thetas, mpisize(comm), mpirank(comm))
    for i in local_start:local_start+local_size-1
        thetas_i = copy(thetas)
        thetas_i[i] += delta_theta

        thetas_i2 = copy(thetas)
        thetas_i2[i] -= delta_theta

        C[i] = -real(
            transition(op, vc, state0, thetas, thetas_i)
            -transition(op, vc, state0, thetas, thetas_i2)
        )/(2*delta_theta)
    end
    if comm === nothing
      return C
    else
      return Allreduce(C, MPI.SUM, comm)
    end
end



function compute_next_thetas_vqs(op::OFQubitOperator, vc::VariationalQuantumCircuit,
    state0::QulacsQuantumState, dtau; delta_theta=1e-8, comm=MPI_COMM_WORLD, verbose=false)
    thetas_dot = compute_thetadot(op, vc, state0, delta_theta, comm=comm, verbose=verbose)
    thetas_dot = convert(Vector{Float64}, thetas_dot)
    return get_thetas(vc) .+ dtau * thetas_dot
end

"Improved VQS"

function compute_next_thetas_direct(op::OFQubitOperator, vc::VariationalQuantumCircuit,
    state0::QulacsQuantumState, dtau;
    comm=MPI_COMM_WORLD, maxiter=100, gtol=1e-5,verbose=true
    )

    state_tau = copy(state0)
    update_quantum_state!(vc, state_tau)
    Etau = get_expectation_value(op, state_tau)

    function cost(thetas::Vector{Float64})
        vc_ = copy(vc)
        state_ = copy(state0)
        update_circuit_param!(vc_, thetas)
        update_quantum_state!(vc_, state_)
        cost_term1 = dtau * get_transition_amplitude(op, state_, state_tau)
        cost_term2 = (dtau * Etau + 1) * inner_product(state_, state_tau)
        real(cost_term1 - cost_term2) 
    end

    cost_history = Float64[] #コスト関数の箱
    thetas_init = get_thetas(vc)
    init_theta_list = thetas_init

    function callback(x)
        push!(cost_history, cost(x))
        if verbose && rank == 0
            println("iter ", length(cost_history), " ", cost_history[end])
        end
    end

    if comm !== nothing
        MPI.Bcast!(thetas_init, 0, comm)
    end
    
    scipy_opt = pyimport("scipy.optimize")
    method = "BFGS"
    options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol)
    opt = scipy_opt.minimize(cost, init_theta_list, method=method,
        callback=callback, jac=generate_numerical_grad(cost),
        options=options)
    return opt["x"]
end    

function _expval(vc, state0, thetas, op)
    state = copy(state0)
    vc_ = copy(vc)
    update_circuit_param!(vc_, thetas)
    update_quantum_state!(vc_, state)
    return get_expectation_value(op, state)
end

function compute_next_thetas_safe(op::OFQubitOperator, vc::VariationalQuantumCircuit,
    state0::QulacsQuantumState, dtau, tau::Float64;
    algorithm="direct", comm=MPI_COMM_WORLD, maxiter=100, gtol=1e-5, delta_theta=1e-8,
    verbose=true, max_recursion=10)
 
    Etau = _expval(vc, state0, get_thetas(vc), op)
    if algorithm == "direct"
        thetas_next = compute_next_thetas_direct(op, vc, state0, dtau,
            comm=comm, maxiter=maxiter, gtol=gtol,verbose=verbose)
    else
        thetas_next = compute_next_thetas_vqs(op, vc, state0, dtau,
            comm=comm, verbose=verbose, delta_theta=delta_theta)
    end
    Etau_next = _expval(vc, state0, thetas_next, op)

    if verbose
       println("dtau: $(dtau), Etau: $(Etau) -> $(Etau_next)")
    end

    if max_recursion == 0 || Etau_next <= Etau
        if verbose && mpirank(comm) == 0
            println("next (recusive) tau point= ", tau+dtau)
            println("Etau=")
        end
        return thetas_next
    end

    # Bad case: we need to decrease dtau
    if verbose
       println("Falling back to recursive model with max_recursion = $(max_recursion)")
    end
    thetas_next1 = compute_next_thetas_safe(op, vc, state0, 0.5*dtau, tau,
        comm=comm, maxiter=maxiter, gtol=gtol,verbose=verbose, max_recursion=max_recursion-1)
    vc_ = copy(vc)
    update_circuit_param!(vc_, thetas_next1)
    return compute_next_thetas_safe(op, vc_, state0, 0.5*dtau, tau+0.5*dtau,
        comm=comm, maxiter=maxiter, gtol=gtol,verbose=verbose, max_recursion=max_recursion-1)
end

function compute_next_thetas_unsafe(op::OFQubitOperator, vc::VariationalQuantumCircuit,
    state0::QulacsQuantumState, dtau;
    algorithm="direct", comm=MPI_COMM_WORLD, maxiter=100, gtol=1e-5, delta_theta=1e-8,
    verbose=true
    )
    if algorithm == "direct"
        thetas_next = compute_next_thetas_direct(op, vc, state0, dtau, 
            comm=comm, maxiter=maxiter, gtol=gtol, verbose=verbose)
    else
        thetas_next = compute_next_thetas_vqs(op, vc, state0, dtau,
            comm=comm, verbose=verbose, delta_theta=delta_theta)
    end
    return thetas_next 
end

#theta(tau)の微分の計算
"""
Compute thetadot = A^(-1) C
"""

function compute_thetadot(op::OFQubitOperator, vc::VariationalQuantumCircuit,
    state0::QulacsQuantumState,delta_theta=1e-8; comm=MPI_COMM_WORLD, verbose=false)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end
    num_thetas = num_theta(vc)
    t1 = time_ns()
    A = compute_A(vc, state0, delta_theta; comm=comm)
    C = compute_C(op, vc, state0, delta_theta)
    t2 = time_ns()
    
    # 今並列計算を実行している。各プロセスで違う乱数がよばれている可能性や変なバグなどがあるかもしれないのでroot=0の値を書くプロセスにブロードキャストして合わせる必要がある。
    if comm !== nothing
        MPI.Bcast!(A, 0, comm)
        MPI.Bcast!(C, 0, comm)
    end

    t3 = time_ns()
    if verbose && mpirank(comm) == 0
        println("timing in compute_thetadot: $(1e-9*(t2-t1)) sec for A $(1e-9*(t3-t2)) sec for C")
        println("condition number of A =", LinearAlgebra.cond(A))
        U, s, Vt = LinearAlgebra.svd(A)
        println("maximum of singular value of A=", findmax(s)[1])
        println("minimum of singular value of A=", findmin(s)[1])
        for i in eachindex(s)
            println("singular values of A = ", s[i])
        end
    end
    
    #thetadot, r = LinearAlgebra.LAPACK.gelsy!(A, C, 1e-5)
    thetadot = fit_svd(C, A, 1e-5)
    #thetadot = tikhonov(C, A, 1e-3)
    thetadot
end


"""
Perform imaginary-time evolution.

ham_op:
    Hamiltonian
vc:
    Variational circuit. The current value of variational parameters are
    used as the initial value of the imaginary-time evolution.
state0:
    The initial state to which the Variational circuit is applied to
taus:
    list of imaginary times in ascending order
    The first element must be 0.0. 
return:
    list of variational parameters at the given imaginary times.
"""
function imag_time_evolve(ham_op::OFQubitOperator, vc::VariationalQuantumCircuit, state0::QulacsQuantumState,
    taus::Vector{Float64}, delta_theta=1e-8;
    comm=MPI_COMM_WORLD, verbose=false, algorithm::String="direct", tol_dE_dtau=1e-5, recursive=true
    )::Tuple{Vector{Vector{Float64}}, Vector{Float64}}
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end
    if taus[1] != 0.0
        error("The first element of taus must be 0!")
    end
    thetas_tau = [copy(get_thetas(vc))]
    log_norm_tau = zeros(Float64, length(taus))  #エルミートの期待値だから必ず実数

    stop_imag_time_evol = false
    for i in 1:length(taus)-1
        if verbose && mpirank(comm) == 0
          println("imag_time_evolve: starting step $(i)...")
        end

        #compute expectation value
        Etau = _expval(vc, state0, thetas_tau[end], ham_op)
        if verbose && mpirank(comm) == 0
            println("tau_ite, Etau_ite=, ", taus[i], " ", Etau)
            #println("tau_ite=", taus[i])
        end

        dtau = taus[i+1] - taus[i]
        if dtau < 0.0
            error("taus must be in strictly asecnding order!")
        end
        if recursive 
            # Compute theta 

            if stop_imag_time_evol
                if verbose && mpirank(comm) == 0
                    println("Skipping imag_time_evolve")
                end
                thetas_next_ = copy(thetas_tau[end])
            else
                vc_ = copy(vc)
                update_circuit_param!(vc_, thetas_tau[i])
                thetas_next_ = compute_next_thetas_safe(ham_op, vc_, state0, dtau, taus[i],
                    algorithm=algorithm, comm=comm, verbose=verbose, delta_theta=delta_theta)
                Etau_next = _expval(vc_, state0, thetas_next_, ham_op)
                if abs(Etau - Etau_next) / dtau < tol_dE_dtau
                    if verbose && mpirank(comm) == 0
                        println("Energy converged!")
                    end
                    stop_imag_time_evol = true
                end
            end
            push!(thetas_tau, thetas_next_)
            # Compute norm
            log_norm_tau[i+1] = log_norm_tau[i] - Etau * (taus[i+1] - taus[i])

        else
            vc_ = copy(vc)
            update_circuit_param!(vc_, thetas_tau[i])
            thetas_next_ = compute_next_thetas_unsafe(ham_op, vc_, state0, dtau,
                algorithm=algorithm, comm=comm, delta_theta=delta_theta, verbose=verbose) 
            Etau_next = _expval(vc_, state0, thetas_next_, ham_op)
            push!(thetas_tau, thetas_next_)

            log_norm_tau[i+1] = log_norm_tau[i] - Etau * (taus[i+1] - taus[i])
        end
    end
    thetas_tau, log_norm_tau
end


function _create_quantum_state(c, state0)
    state0_ = copy(state0)
    update_quantum_state!(c, state0_)
    state0_
end

function _create_quantum_state(c, theta::Vector{Float64}, state0::QuantumState)
    c_ = copy(c)
    state0_ = copy(state0)
    update_circuit_param!(c_, theta)
    _create_quantum_state(c_, state0_)
end



"""
Calculate green function based on imaginary-time evolution.

ham_op:
    Hamiltonian
c_op:
    annihilation operator 
cdagg_op:
    creation operator 
vc:
    Variational circuit. The current value of variational parameters are
    used as the initial value of the imaginary-time evolution.
state_gs:
    The state is the ground state of the hamiltonian
state0_ex:
    The excited state to which the creation operator  is applied to
taus:
    list of imaginary times in ascending order
    The first element must be 0.0. The last element must be beta.
return:
    The list of Green function at each tau 
"""

function compute_gtau(
    ham_op::OFQubitOperator,
    left_op::OFQubitOperator,
    right_op::OFQubitOperator,
    vc_ex::VariationalQuantumCircuit,
    state_gs::QulacsQuantumState,　
    state0_ex::QulacsQuantumState,
    taus::Vector{Float64}, delta_theta=1e-8;
    comm=MPI_COMM_WORLD, verbose=false, algorithm::String="direct",recursive=true
    )
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end

    if taus[1] != 0.0
        error("The first element of taus must be 0!")
    end

    if !all(taus[2:end] .> taus[1:end-1])
       error("taus must in strictly asecnding order!")
    end

    if verbose && mpirank(comm) == 0
        println("Applying to an operator to the ket...")
    end
    circuit_right_ex = copy(vc_ex) 
    right_squared_norm = apply_qubit_op!(right_op, state_gs,
       circuit_right_ex, state0_ex,
       minimizer=mk_scipy_minimize(
           options = Dict("disp" => verbose),
           verbose=verbose)
        )
    state_right_ex = copy(state0_ex)
    update_quantum_state!(circuit_right_ex, state_right_ex)
    if verbose
        println("Successfully applied an operator to the ket!")
    end

    # exp(-tau H)c^{dag}_j|g.s>
    if algorithm == "vqs"
        thetas_tau_right, log_norm_tau_right = imag_time_evolve(
            ham_op, circuit_right_ex, state0_ex, taus, delta_theta,
            comm=comm, verbose=verbose, algorithm="vqs", recursive=recursive)
    elseif algorithm == "direct"
        thetas_tau_right, log_norm_tau_right = imag_time_evolve(
            ham_op, circuit_right_ex, state0_ex, taus, delta_theta,
            comm=comm, verbose=verbose, algorithm="direct", recursive=recursive)
    end
    #thetas_tau_right, log_norm_tau_right = imag_time_evolve(ham_op, circuit_right_ex, state0_ex, taus, delta_theta, comm=comm)

    Gfunc_ij_list = Complex{Float64}[]
    E_gs = get_expectation_value(ham_op, state_gs)

    norm = Complex{Float64}[]

    for t in eachindex(taus)
        state_right = _create_quantum_state(vc_ex, thetas_tau_right[t], state0_ex)
        state_left = copy(state_gs)
        # Divide the qubit operator of c_i into its real and imaginary parts.
        op_re, op_im = divide_real_imag(left_op)
        g_re = get_transition_amplitude(op_re, state_left, state_right)
        g_im = get_transition_amplitude(op_im, state_left, state_right)
        #基底エネルギーを基準にする。
        push!(norm, exp(log_norm_tau_right[t] + E_gs * taus[t]))
        push!(Gfunc_ij_list, -(g_re + im * g_im) * right_squared_norm * exp(log_norm_tau_right[t] + E_gs *  taus[t]))
    end
    Gfunc_ij_list, norm
end



