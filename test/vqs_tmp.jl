using Test
using LinearAlgebra
using QCMaterial


import Random
import PyCall: pyimport


@testset "vqs.compute_gtau_2site_hubbard_U=0_tau_minus" begin
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

    #c^{dag},c
    right_op = FermionOperator("$(up1) ", 1.0)
    right_op = jordan_wigner(right_op)
    left_op = FermionOperator("$(up1)^ ", 1.0)
    left_op = jordan_wigner(left_op)

    ##Ansatz -> apply_qubit_op & imag_time_evolve
    #c_ex = QulacsParametricQuantumCircuit(n_qubit)
    #add_X_gate!(c_ex, 2)
    #target = [2,4] 
    #pauli_ids = [pauli_Y, pauli_Y] 
    #add_parametric_multi_Pauli_rotation_gate!(c_ex, target, pauli_ids, 0.3*pi)

    #debug
    #target_debug = [1,3] 
    #pauli_ids_debug = [pauli_Z, pauli_X] 
    #add_parametric_multi_Pauli_rotation_gate!(c_ex, target_debug, pauli_ids_debug, 0.3*pi)

    #add_Sdag_gate!(c_ex, 4)
    #vc_ex = QulacsVariationalQuantumCircuit(c_ex)

    vc_ex = uccgsd(n_qubit, orbital_rot=false, conserv_Sz_singles=false) 
    #get_thetas(circuit)
    #set_initial_parameter -> circuit

    theta_init_vcex = rand(num_theta(vc_ex))
    init_theta_vcex_list = theta_init_vcex
    update_circuit_param!(vc_ex, init_theta_vcex_list)

    #state_gs = QulacsQuantumState(n_qubit,0b0000)
    state_gs = create_hf_state(n_qubit, n_electron)
    update_quantum_state!(vc, state_gs)
    E_gs_debug = get_expectation_value(ham_op, state_gs)
    norm_gs = inner_product(state_gs, state_gs)
    state0_ex = QulacsQuantumState(n_qubit,0b0000)
    
    taus = collect(range(0.0, 1, length=200))
    beta = taus[end]

    k = (2 * t)/(ε + (ε^2 + 4 * t^2)^0.5)  
    s = (2 * t)/(ε - (ε^2 + 4 * t^2)^0.5)
    D = (1 + k^2)^0.5
    E = (1 + s^2)^0.5
    ε_1 = (-ε - (ε^2 + 4*t^2)^0.5) / 2
    ε_2 = (-ε + (ε^2 + 4*t^2)^0.5) / 2
    E_G = 2*ε_1
    E_1 = 2*ε_1 + ε
    
    Gfunc_ij_exact(τ) = exp(τ * (2*ε_1 - ε_1)) * s^2 / E^2
    #Gfunc_ij_exact(τ) = exp(2 * ε_1 *τ ) *(s^2* exp(ε*τ)+1) * s^2 / E^4
    Gfunc_ij_list_ref = Gfunc_ij_exact.(taus) 
    #println("Gfunc_ij_list_ref=", Gfunc_ij_list_ref)

    #Gfunc_ij_list = -compute_gtau(ham_op, left_op, right_op, vc_ex,  state_gs, state0_ex, taus, d_theta)
    #println("Gfunc_ij_list=", Gfunc_ij_list)
    #@test isapprox(Gfunc_ij_list_ref, Gfunc_ij_list, rtol=0.01)
end
