using LinearAlgebra
using Random
using Statistics
#using ConcreteStructs

export predict, mse, get_cost_gradient, update!, update_circuit_param!
export solve_gs_kucj_Adam
export solve_gs_sampling_Adam, apply_qubit_ham_sampling_Adam!
export apply_qubit_op_sampling_Adam!

function predict(x, θ₀::Real, θ₁::Real)
    x * θ₀ + θ₁
end

predict(x, θ::AbstractVector; kwargs...) = predict(x, θ[1], θ[2]; kwargs...)

function mse(X, y, θ₀, θ₁)
    ŷ = [predict(x, θ₀, θ₁) for x in X]
    return mean((ŷ - y) .^ 2)
end
mse(X, y, θ::AbstractVector) = mse(X, y, θ[1], θ[2])


function get_cost_gradient(mse, X, y, θ₀::Real, θ₁::Real; dx=1e-5)
    ∇ = zeros(2)
    θ = [θ₀, θ₁]
    tmpθ₁ = copy(θ)
    tmpθ₂ = copy(θ)
    for i in eachindex(θ)
        tmpθ₁[i] += dx
        tmpθ₂[i] -= dx
        ∇[i] = (mse(X, y, tmpθ₁) - mse(X, y, tmpθ₂)) / (2 * dx)
        tmpθ₁[i] = tmpθ₂[i] = θ[i]
    end
    return ∇
end

get_cost_gradient(mse, X, y, θ::AbstractVector) = get_cost_gradient(mse, X, y, θ[1], θ[2])

struct Trainer
    n_steps::Any
    mse::Any # cost_function?
    X::Any #?
    y::Any #? 
    #new() = 
end

struct SGD #一回の最適化の情報を書く
    tmp_theta::Any
    eta::Any # learning_rate
end

function update!(opt::SGD, mse, X, y)
    g = get_cost_gradient(mse, X, y, opt.tmp_theta)
    j = rand(1:length(opt.tmp_theta), 1) # parameter idex
    p = g[j] #i番目のパラメータに関する勾配
    opt.tmp_theta[j] .-= opt.eta * p
end


struct Momentum #一回の最適化の情報を書く
    tmp_theta::Any # θ₀, θ₁
    v::Any
    eta::Any # learning_rate
    momentum::Any
end


function update!(opt::Momentum, mse, X, y)
    g = get_cost_gradient(mse, X, y, opt.tmp_theta)
    opt.v .+= opt.momentum .* opt.v .- g .* opt.eta
    opt.tmp_theta .+= opt.v
end

struct AdaGrad #一回の最適化の情報を書く
    tmp_theta::Any # θ₀, θ₁
    s::Any #s1,s2
    eta::Any # learning_rate
    eps::Any
end

## moduleを使う場合
function update!(opt::AdaGrad, mse, X, y) #opt::adagmy.AdaGraとしないといけない！
    g = get_cost_gradient(mse, X, y, opt.tmp_theta)
    opt.s .+= g .^ 2
    opt.tmp_theta .-= (opt.eta ./ sqrt.(opt.s .+ opt.eps)) .* g
end


struct RMSprop #一回の最適化の情報を書く
    tmp_theta::Any # θ₀, θ₁
    s::Any #s1,s2
    eta::Any # learning_rate
    eps::Any
    decay_rate::Any
end

function update!(opt::RMSprop, mse, X, y) #opt::adagmy.AdaGraとしないといけない！
    opt.s .*= opt.decay_rate
    g = get_cost_gradient(mse, X, y, opt.tmp_theta)
    opt.s .+= (1 - opt.decay_rate) .* g .^ 2
    opt.tmp_theta .-= (opt.eta) .* g ./ (sqrt.(opt.s) .+ opt.eps)
end

mutable struct Adam #一回の最適化の情報を書く
    tmp_theta::Vector{Float64} # θ₀, θ₁
    m::Vector{Float64}
    v::Vector{Float64}
    eta::Float64 # learning_rate
    eps::Float64
    beta::Vector{Float64}
end


function update!(opt::Adam, mse, X, y, n_steps)
    for i = 1:n_steps
        g = get_cost_gradient(mse, X, y, opt.tmp_theta)
        opt.m .+= (1 - opt.beta[1]) .* (g .- opt.m)
        opt.v .+= (1 - opt.beta[2]) .* (g .^ 2 .- opt.v)
        opt.tmp_theta .-=
            ((opt.eta .* sqrt.(1.0 - opt.beta[2]^i)) ./ (sqrt.(1.0 .- opt.beta[1]^i))) .*
            opt.m ./ (sqrt.(opt.v) .+ opt.eps)
    end
end

function callback_(x)
    push!(cost_history, cost(x))
    if verbose && rank == 0
        println("iter ", length(cost_history), " ", cost_history[end])
    end
end

function update_circuit_param!(opt::Adam, cost, n_steps; verbose=true, change_rate=500, dx=1e-5)
    for i = 1:n_steps
        #eta = opt.eta * 0.8
        if i == change_rate
            opt.eta = opt.eta * 0.8
        end
        #@show opt.eta
        #@show verbose
        #if verbose 
        if verbose && MPI_rank  == 0
            println("iter ", i, " ", cost(opt.tmp_theta))
        end
        g = generate_numerical_grad(cost; verbose, dx)
        g = g(opt.tmp_theta)
        opt.m .+= (1 - opt.beta[1]) .* (g .- opt.m) #g,mの次元は一致させる
        opt.v .+= (1 - opt.beta[2]) .* (g .^ 2 .- opt.v) #g,vの次元は一致させる。
        opt.tmp_theta .-=
            ((opt.eta .* sqrt.(1.0 - opt.beta[2]^i)) ./ (sqrt.(1.0 .- opt.beta[1]^i))) .*
            opt.m ./ (sqrt.(opt.v) .+ opt.eps)
    end
end


function solve_gs_Adam(
    ham_qubit::QubitOperator,
    circuit::VariationalQuantumCircuit,
    m,
    v,
    state0::QuantumState;
    theta_init=nothing,
    comm=MPI_COMM_WORLD,
    eta=0.2,
    eps=1e-7,
    beta=[0.7, 0.777],
    n_steps=3000,
    verbose=false,
    dx = 1e-5
)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end
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

    adam = QCMaterial.Adam(init_theta_list, m, v, eta, eps, beta)
    update_circuit_param!(adam, cost, n_steps; verbose, dx)
    cost_opt = cost(init_theta_list)
    return cost_opt, get_thetas(circuit)
end

function solve_gs_kucj_Adam(
    ham_qubit::QubitOperator,
    circuit::VariationalQuantumCircuit,
    m,
    v,
    state0::QuantumState,
    keys;
    theta_init=nothing,
    comm=MPI_COMM_WORLD,
    eta=0.2,
    eps=1e-7,
    beta=[0.7, 0.777],
    n_steps=100,
    verbose=false
)
    if is_mpi_on && isnothing(comm) 
        error("comm must be given when mpi is one!")
    end
    
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

    pinfo = ParamInfo(keys)
    cost_tmp(θunique) = cost(expand(pinfo, θunique))

    #if theta_init === nothing
    #    theta_init = rand(size(circuit.theta_offsets)[1])
    #end

    if comm !== nothing
        # Make sure all processes use the same initial values
        MPI.Bcast!(theta_init, 0, comm)
    end

    cost_history = Float64[] #コスト関数の箱
    #init_theta_list = theta_init
    push!(cost_history, cost_tmp(theta_init))
    #method = "BFGS"
    #options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol)
    @show cost_tmp(theta_init)
    adam = QCMaterial.Adam(theta_init, m, v, eta, eps, beta)
    update_circuit_param!(adam, cost_tmp, n_steps; verbose)
    cost_opt = cost_tmp(theta_init)
    return cost_opt, get_thetas(circuit)
end



function solve_gs_sampling_Adam(
    ham_qubit::QubitOperator,
    circuit::VariationalQuantumCircuit,
    m,
    v,
    state0::QuantumState;
    theta_init=nothing,
    comm=MPI_COMM_WORLD,
    eta=0.2,
    eps=1e-7,
    beta=[0.7, 0.777],
    n_steps=3000,
    verbose=false,
    dx = 1e-1,
    nshots = 2^15
)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end
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
        #return get_expectation_value(ham_qubit, state)
        return get_expected_value_sampling(ham_qubit, state, nshots=nshots)
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
    #method = "BFGS"
    #options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol)
    #function callback(x)
    #    push!(cost_history, cost(x))
    #    if verbose && rank == 0
    #        println("iter ", length(cost_history), " ", cost_history[end])
    #    end
    #end
    adam = QCMaterial.Adam(init_theta_list, m, v, eta, eps, beta)
    update_circuit_param!(adam, cost, n_steps; verbose, dx)
    cost_opt = cost(init_theta_list)
    return cost_opt, get_thetas(circuit)
end


function apply_qubit_ham_sampling_Adam!(
    op::QubitOperator,
    state_augmented::QuantumState, 
    circuit_bra::VariationalQuantumCircuit, 
    circuit_ket::VariationalQuantumCircuit,
    m,
    v;
    theta_init=nothing,
    comm=MPI_COMM_WORLD,
    eta=0.2,
    eps=1e-7,
    beta=[0.9, 0.99],
    n_steps=3000,
    verbose=false,
    dx = 1e-1,
    nshots = 2^14
)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end

    if comm === nothing
        rank = 0
    else
        rank = MPI.Comm_rank(comm)
    end

    function cost(thetas::Vector{Float64})
        update_circuit_param!(circuit_bra, thetas)
        op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_bra, op, circuit_ket, nshots=nshots)
        op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_bra, op, circuit_ket; nshots)
        - abs2(op_re_re + op_re_im * im)
    end

    if theta_init === nothing
        theta_init = rand(num_theta(circuit_bra)) #
    end

    if comm !== nothing
        # Make sure all processes use the same initial values
        MPI.Bcast!(theta_init, 0, comm)
    end
    cost_history = Float64[] #コスト関数の箱
    init_theta_list = theta_init
    push!(cost_history, cost(init_theta_list))

    adam = QCMaterial.Adam(init_theta_list, m, v, eta, eps, beta)
    update_circuit_param!(adam, cost, n_steps; verbose, dx)
    cost_opt = cost(init_theta_list)
    opt_thetas = get_thetas(circuit_bra) 
    #return cost_opt, get_thetas(circuit)

    update_circuit_param!(circuit_bra, opt_thetas)
    op_re_re = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_bra, op, circuit_ket; nshots)
    op_re_im = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_bra, op, circuit_ket; nshots)
    re__ = op_re_re + im * op_re_im
    return re__
end



function apply_qubit_op_sampling_Adam!(
    op::QubitOperator,
    state_augmented::QuantumState, 
    circuit_bra::VariationalQuantumCircuit, 
    circuit_ket::VariationalQuantumCircuit,
    m,
    v;
    theta_init=nothing,
    comm=MPI_COMM_WORLD,
    eta=0.2,
    eps=1e-7,
    beta=[0.9, 0.99],
    n_steps=3000,
    verbose=false,
    dx = 1e-1,
    nshots = 2^14
)
    if is_mpi_on && comm === nothing
        error("comm must be given when mpi is one!")
    end

    if comm === nothing
        rank = 0
    else
        rank = MPI.Comm_rank(comm)
    end

    her, antiher = divide_real_imag(op)
    function cost(thetas::Vector{Float64})
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
        #@show op_re_re + op_im_im * im * im 
        abs2(1.0 - (op_re_re + op_im_im * im * im ))
        #-abs2( (op_re_ + op_im_ * im))
    end

    if theta_init === nothing
        theta_init = rand(num_theta(circuit_bra)) #
    end

    if comm !== nothing
        # Make sure all processes use the same initial values
        MPI.Bcast!(theta_init, 0, comm)
    end
    cost_history = Float64[] #コスト関数の箱
    init_theta_list = theta_init
    push!(cost_history, cost(init_theta_list))

    adam = QCMaterial.Adam(init_theta_list, m, v, eta, eps, beta)
    update_circuit_param!(adam, cost, n_steps; verbose, dx)
    cost_opt = cost(init_theta_list)
    opt_thetas = get_thetas(circuit_bra) 
    #return cost_opt, get_thetas(circuit)

    update_circuit_param!(circuit_bra, opt_thetas)
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
    return z
end

get_number_preserving_sparse_operator