using Test
using LinearAlgebra
using QCMaterial


import Random
import PyCall: pyimport

@testset "vqs.A" begin
    n_qubit = 2
    n_electron = 1

    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 0.0)

    state0 = QulacsQuantumState(n_qubit, 0b01)

    new_thetas = copy(c.thetas)

    new_thetas .+= 1e-8
    c_new = copy(c)
    update_circuit_param!(c_new, new_thetas)
    A = compute_A(c, state0, 1e-2)
    @test A ≈ [0.25] atol=1e-5
end

@testset "vqs.C" begin
    n_qubit = 2
    n_electron = 1

    ham = OFQubitOperator("X1 X2", 1.0)

    c = UCCQuantumCircuit(n_qubit)
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 0.0)

    state0 = QulacsQuantumState(n_qubit, 0b01)

    C = compute_C(ham, c, state0, 1e-2)
    @test C ≈ [0.5] atol=1e-5
end

@testset "vqs.thetadot" begin
    n_qubit = 2
    n_electron = 1
    ham = OFQubitOperator("X1 X2", 1.0)
    c = UCCQuantumCircuit(n_qubit)
    
    # a_1^dagger a_2 - a^2 a_1 -> 0.5i (X1 Y2 - X2 Y1)
    generator = gen_t1(1, 2)
    add_parametric_circuit_using_generator!(c, generator, 0.0)

    state0 = QulacsQuantumState(n_qubit,0b01)
    thetadot = compute_thetadot(ham, c, state0, 1e-2) 
    @test thetadot ≈ [2] atol=1e-5
end

@testset "vqs.imag_time_evolve_single_qubit" begin
    """
    Hamiltonian H = Z1
    No time evolution occurs.
    exp(- tau * H) |1> = A |1>,
    where A is a normalization constant.
    """
    n_qubit = 1
    d_theta = 0.01

    ham = OFQubitOperator("Z1", 1.0)
    state0 = QulacsQuantumState(n_qubit, 0b1)

    c = QulacsParametricQuantumCircuit(n_qubit)
    add_parametric_RY_gate!(c, 1, 0.0)
    vc = QulacsVariationalQuantumCircuit(c)

    taus = [0.0, 1.0]
    thetas_tau = imag_time_evolve(ham, vc, state0, taus, d_theta)[1]
    
    for i in eachindex(taus)
        @test all(thetas_tau[i] .≈ [0.0])
    end
end

@testset "vqs.imag_time_evolve_single_qubit2" begin
    """
    Hamiltonian H = Z1
    No time evolution occurs.
    exp(- tau * H) (|0> + |1>)/sqrt(2) = R_Y(theta(tau)) |0>,
    where A is a normalization constant and
    theta(tau) = acos(exp(-tau)/sqrt(exp(-2tau) + exp(2*tau))).
    theta = pi/2 (tau=0) => pi (tau=+infty)
    """
    n_qubit = 1
    d_theta = 0.01

    ham = OFQubitOperator("Z1", 1.0)
    state0 = QulacsQuantumState(n_qubit, 0b0)

    c = QulacsParametricQuantumCircuit(n_qubit)
    add_parametric_RY_gate!(c, 1, 0.5*pi)
    vc = QulacsVariationalQuantumCircuit(c)

    taus = collect(range(0.0, 1, length=100))
    thetas_tau = imag_time_evolve(ham, vc, state0, taus, d_theta)[1]

    theta_extact(τ) = 2*acos(exp(-τ)/sqrt(exp(-2*τ) + exp(2*τ)))

    thetas_ref = theta_extact.(taus)
    thetas_res = [thetas_tau[i][1] for i in eachindex(taus)]
    @test isapprox(thetas_ref, thetas_res, rtol=0.01)
end


@testset "vqs.imag_time_evolve_two_qubits" begin
    """
    1site hubbard under magnetic field
    Hamiltonian H = Un_(1up)n_(2dn) - hn_(1up) + hn_(2dn) 
    No time evolution occurs.
    exp(- tau * H) (|1up> + |1dn>)/sqrt(2) = exp(theta/2*(c_1up^(dag)c_1dn-c_1dn^(dag)c_1up)) |1up>,

    """
    n_qubit = 2
    d_theta = 0.01
    
    U = 1.0
    h = 1.0

    ham = FermionOperator()
    up = up_index(1)
    down = down_index(1)
    ham += FermionOperator("$(up)^ $(down)^ $(up) $(down)", -U)
    ham += FermionOperator("$(up)^  $(up) ", -h)
    ham += FermionOperator("$(down)^ $(down)", h)
    ham = jordan_wigner(ham)

    state0 = QulacsQuantumState(n_qubit, 0b01)

    c = QulacsParametricQuantumCircuit(n_qubit)
    # add_parametric_RYY_gate! -> add_parametric_multi_Pauli_rotation_gate!
    target = [1,2] 
    #X->pauli_X, Y->pauli_Y, Z->pauli_Z
    pauli_ids = [pauli_Y, pauli_Y] 
    add_parametric_multi_Pauli_rotation_gate!(c, target, pauli_ids, 0.5*pi)
    #QulacsVariationalQuantumCircuitに対してadd_S_gate!を定義する。
    #QulacsVariationalQuantumCircuitとQulacsQuantumCircuitは両方ともQuantumCircuitを親に持つが、兄弟間同士で継承(多重継承)できないことに注意する。
    add_S_gate!(c, 2)
    vc = QulacsVariationalQuantumCircuit(c)

    taus = collect(range(0.0, 1, length=100))
    thetas_tau = imag_time_evolve(ham, vc, state0, taus, d_theta)[1]

    theta_extact(τ) = 2*acos(exp(h*τ)/sqrt(exp(2*h*τ) + exp(-2*h*τ)))

    #generate vector whose elements are theta_exact(τ) 
    thetas_ref = theta_extact.(taus)
    #i is tau and num of parameter is 1
    thetas_res = [thetas_tau[i][1] for i in eachindex(taus)]
    @test isapprox(thetas_ref, thetas_res, rtol=0.01)

end


#compute_gtaを使った１サイトハバードのグリーン関数の一部の計算（一つの基底状態のみの計算）
##ノルムを考慮した
@testset "vqs.compute_gtau_norm_1site_hubbard" begin
    """
    """
    n_qubit　= 2
    nsite = 1
    t = 0.0
    U = 2.0
    µ = 1.0
    up = up_index(1)
    down = down_index(1)
    d_theta = 0.01
    
    ham_op = generate_ham_1d_hubbard(t, U, nsite, μ)
    ham_op = jordan_wigner(ham_op)
     
    
    c_op = FermionOperator("$(down) ", 1.0)
    c_op = jordan_wigner(c_op)
    cdagg_op = FermionOperator("$(down)^ ", 1.0)
    cdagg_op = jordan_wigner(cdagg_op)

    c_ = QulacsParametricQuantumCircuit(n_qubit)
    #add_X_gate!(c_, 1)
    #add_X_gate!(c_, 2)
    target = [1,2] 
    pauli_ids = [pauli_Y, pauli_Y] 
    add_parametric_multi_Pauli_rotation_gate!(c_, target, pauli_ids, 0.01*pi)
    vc = QulacsVariationalQuantumCircuit(c_)
  

    state_gs = QulacsQuantumState(n_qubit,0b01)　#state_gs is the ground state after VQE
    state0_ex = QulacsQuantumState(n_qubit,0b11)
    
    taus = collect(range(0.0, 1, length=5))
    beta = taus[end]

    Gfunc_ij_list = compute_gtau_before(ham_op, c_op, cdagg_op, vc,  state_gs, state0_ex, taus, d_theta)
    Gfunc_ij_list_exact(τ) = -exp(-U * τ + µ * τ)
    Gfunc_ij_list_ref = Gfunc_ij_list_exact.(taus) 
    println("Gfunc_ij_list_ref=",Gfunc_ij_list_ref)
    println("Gfunc_ij_list=",Gfunc_ij_list)
    @test isapprox(Gfunc_ij_list_ref, Gfunc_ij_list, rtol=0.01)
end


#compute_gtaを使った2サイトハバード(U=0,かつ1サイトのポテンシャルあり)のグリーン関数の一部の計算（一つの基底状態のみの計算）
@testset "vqs.compute_gtau_norm_2site_hubbard_U=0_debug" begin
    """
    """
    nsite = 2
    n_qubit　= 2*nsite
    t = 1.0
    ε = 1.0
    up1 = up_index(1)
    dn1 = down_index(1)
    up2 = up_index(2)
    dn2 = down_index(2)
    d_theta = 0.01
    
    #Hamiltonian
    ham_op = FermionOperator()
    ham_op += FermionOperator("$(up1)^ $(up2)", -t)
    ham_op += FermionOperator("$(up2)^ $(up1)", -t)
    ham_op += FermionOperator("$(dn1)^ $(dn2)", -t)
    ham_op += FermionOperator("$(dn2)^ $(dn1)", -t)

    ham_op += FermionOperator("$(up1)^ $(up1)", -ε)
    ham_op += FermionOperator("$(dn1)^ $(dn1)", -ε)
    
    n_electron = 2
    @assert mod(n_electron, 2) == 0
    sparse_mat = get_number_preserving_sparse_operator(ham_op, n_qubit, n_electron);
    enes_ed = eigvals(sparse_mat.toarray());

    ham_op = jordan_wigner(ham_op)

    #c = QulacsParametricQuantumCircuit(n_qubit)
    #add_X_gate!(c, 1)
    #add_X_gate!(c, 2)

    #add_CNOT_gate!(c, 3, 2)
    #add_RZ_gate!(c, 3, -pi*1)
    #add_RY_gate!(c, 3, -pi*0.5)
    #θ1 is used
    #add_parametric_RY_gate!(c, 3, -pi*0.5)
    #add_CNOT_gate!(c, 2, 3)
    #add_RY_gate!(c, 3, pi*0.5)
    #θ2 is used, but we want to set θ2=θ1
    #add_parametric_RY_gate!(c, 3, pi*0.5)
    #add_RZ_gate!(c, 3, pi*1)
    #add_CNOT_gate!(c, 3, 2)

    #add_CNOT_gate!(c, 4, 1)
    #add_RZ_gate!(c, 4, -pi*1)
    #add_RY_gate!(c, 4, -pi*0.5)
    #θ3 is used
    #add_parametric_RY_gate!(c, 4, -pi*0.5)
    #add_CNOT_gate!(c, 1, 4)
    #dd_RY_gate!(c, 4, pi*0.5)
    #θ4 is used,  but we want to set θ3=θ4
    #add_parametric_RY_gate!(c, 4, pi*0.5)
    #add_RZ_gate!(c, 4, pi*1)
    #add_CNOT_gate!(c, 4, 1)
    
    #ERROR!:no method matching add_SWAP_gate!(::QulacsParametricQuantumCircuit, ::Int64, ::Int64)
    #add_SWAP_gate!(c, 1, 2) 
    #add_CNOT_gate!(c, 2, 1)
    #add_CNOT_gate!(c, 1, 2)
    #add_CNOT_gate!(c, 2, 1)   

    #vc = QulacsVariationalQuantumCircuit(c)
    vc = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=false)
    
    #Perform VQE
    function cost(theta_list)
        state0_gs = create_hf_state(n_qubit, n_electron)
        #state0_gs = QulacsQuantumState(n_qubit,0b0000)
        update_circuit_param!(vc, theta_list)
        update_quantum_state!(vc, state0_gs)
        get_expectation_value(ham_op, state0_gs)
    end

    theta_init = rand(num_theta(vc))
    cost_history = Float64[]
    init_theta_list = theta_init
    push!(cost_history, cost(init_theta_list))

    method = "BFGS"
    options = Dict("disp" => true, "maxiter" =>200, "gtol" =>1e-5)
    callback(x) = push!(cost_history, cost(x))
    Random.seed!(1)
    scipy_opt = pyimport("scipy.optimize")
    opt = scipy_opt.minimize(cost, init_theta_list, method=method, callback=callback)
    
    Eigval_min = minimum(enes_ed)
    ε_2 = (-ε - (ε^2 + 4*t^2)^0.5) / 2
    println("E_vqe =",cost_history[end])
    println("Eigaval_min_ed=",Eigval_min)
    println("Eigval_min_analytical=",2 * ε_2)
    #@test abs(Eigval_min - 2 * ε_2) < 1e-6
    #@test abs(Eigval_min - cost_history[end]) < 1e-6

    #c^{dag},c
    c_op = FermionOperator("$(up1) ", 1.0)
    c_op = jordan_wigner(c_op)
    cdagg_op = FermionOperator("$(up1)^ ", 1.0)
    cdagg_op = jordan_wigner(cdagg_op)

    ##Ansatz -> apply_qubit_op & imag_time_evolve
    c_ex = QulacsParametricQuantumCircuit(n_qubit)
    
    add_X_gate!(c_ex, 1)
    add_X_gate!(c_ex, 2)
    add_X_gate!(c_ex, 3)
    target = [2,4] 
    pauli_ids = [pauli_Y, pauli_Y] 
    add_parametric_multi_Pauli_rotation_gate!(c_ex, target, pauli_ids, 0.2*pi)
    add_S_gate!(c_ex, 4)

    vc_ex = QulacsVariationalQuantumCircuit(c_ex)

    #state_gs = QulacsQuantumState(n_qubit,0b0000)
    state_gs = create_hf_state(n_qubit, n_electron)
    update_quantum_state!(vc, state_gs)
    println("state_gs=", get_vector(state_gs)) 
    E_gs_debug = get_expectation_value(ham_op, state_gs)
    #println("E_gs_debug =", E_gs_debug )
    norm_gs = inner_product(state_gs, state_gs)
    println("norm_gs=", norm_gs) #norm_gs=0.9999999999999997 + 0.0im
    state0_ex = QulacsQuantumState(n_qubit,0b0000)
    
    taus = collect(range(0.0, 1, length=10))
    beta = taus[end]

    
    A = [-ε -t
    -t 0]
    println(A)
    println(typeof(A))
    e,u = eigen(A) 
    println(e)
    println(u)
    ε_minus = e[1]
    println("2*ε_minus", ε_minus*2 )
    ε_plus = e[2] 

    U_11 = u[1,1]
    U_21 = u[2,1]
    U_12 = u[1,2]
    U_22 = u[2,2]

    

    #exact G_func
    k = (2 * t)/(ε + (ε^2 + 4 * t^2)^0.5)  
    s = (2 * t)/(ε - (ε^2 + 4 * t^2)^0.5)
    D = (1 + k^2)^0.5
    E = (1 + s^2)^0.5
    ε_1 = (-ε - (ε^2 + 4*t^2)^0.5) / 2
    println("ε_1=", ε_1)
    ε_2 = (-ε + (ε^2 + 4*t^2)^0.5) / 2
    println("ε_2=", ε_2)
    coef_1 = (k - s) / (E^2 * D)
    coef_3 = (s - k) * (s*k + 1) / (E^3 * D^2)
    E_G = 2*ε_1
    #println("E_G =", E_G )
    E_12 = 2 * ε_1 + ε_2
    E_34 = ε_1 + 2 * ε_2
    
    #-exp(τ * E_G)*(E)^(-2)*(H+G)
    #Gfunc_ij_exact(τ) = -exp(τ * E_G)/(E^2) * ((exp(-τ * E_12) * coef_1 * (k-s)/(E^2 * D) * (s^2 + 1)) + (exp(-τ * E_34) * coef_3 * (s-k)/(E * D^2) * (k*s + 1)))
    Gfunc_ij_exact(τ) = -exp(τ * (-ε_2)) * coef_1^2
    Gfunc_ij_list_ref = Gfunc_ij_exact.(taus) 

    println("Gfunc_ij_list_ref=",Gfunc_ij_list_ref)

    Gfunc_ij_list = compute_gtau_norm(ham_op, c_op, cdagg_op, vc_ex,  state_gs, state0_ex, taus, d_theta)
    println("Gfunc_ij_list=",Gfunc_ij_list)
    @test isapprox(Gfunc_ij_list_ref, Gfunc_ij_list, rtol=0.01)
end