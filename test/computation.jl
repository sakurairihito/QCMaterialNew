using Test
using QCMaterial
using LinearAlgebra
using Random
import MPI
@testset "computation.apply_qubit_op" begin
    n_qubit = 2
    # Prepare |Psi> = |00>
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b00)
    # Apply c^dagger_2 to |Psi>, which will yield |10>.
    op = jordan_wigner(FermionOperator("2^"))
    # Prepare |phi> = |01>
    state0_bra = QulacsQuantumState(n_qubit)
    set_computational_basis!(state0_bra, 0b01)

    # Fit <01|U(theta)^dagger c_2^dagger |00>
    
    circuit_bra = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=false)

    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    #update_circuit_param!(circuit_bra, rand(size(circuit_bra.theta_offset)[1])) 
    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_qubit_op!(op, state, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
        verbose=verbose))
    println("squared_norm=",squared_norm)
    # Verify the result
    # @test≈ 1.0
    @test isapprox( abs(squared_norm), 1.0, rtol=1e-3)
end

@testset "computation.apply_qubit_ham" begin
    n_qubit = 4
    # Prepare |Psi> = |00>
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b0011)
    op = jordan_wigner(FermionOperator("2^ 2"))
    # Prepare |phi> = |01>
    state0_bra = QulacsQuantumState(n_qubit)
    set_computational_basis!(state0_bra, 0b0011)

    # Fit <01|U(theta)^dagger c_2^dagger |00>
    
    circuit_bra = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=false)

    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    #update_circuit_param!(circuit_bra, rand(size(circuit_bra.theta_offset)[1])) 
    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_qubit_ham!(op, state, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
        verbose=verbose))
    println("squared_norm=",squared_norm)


    # Verify the result
    #@test≈ 1.0
    @test isapprox( abs(squared_norm), 1.0, rtol=1e-3)
end


@testset "computation.apply_qubit_ham2" begin
    n_qubit = 4
    # Prepare |Psi> = |00>
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b1100)
    op = jordan_wigner(FermionOperator("2^ 2"))
    # Prepare |phi> = |01>
    state0_bra = QulacsQuantumState(n_qubit)
    set_computational_basis!(state0_bra, 0b0011)

    # Fit <01|U(theta)^dagger c_2^dagger |00>
    
    circuit_bra = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=false)

    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    #update_circuit_param!(circuit_bra, rand(size(circuit_bra.theta_offset)[1])) 
    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_qubit_ham!(op, state, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
        verbose=verbose))
    println("squared_norm=",squared_norm)


    # Verify the result
    #@test≈ 1.0
    @test isapprox( abs(squared_norm), 0.0, rtol=1e-3)
end


@testset "computation.apply_qubit_ham2.5" begin
    n_qubit = 4
    # Prepare |Psi> = |00>
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b1111)
    op = jordan_wigner(FermionOperator("2^ 2 1^ 1"))
    # Prepare |phi> = |01>
    state0_bra = QulacsQuantumState(n_qubit)
    set_computational_basis!(state0_bra, 0b1111)
    # Fit <01|U(theta)^dagger c_2^dagger |00>
    circuit_bra = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=false)
    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    #update_circuit_param!(circuit_bra, rand(size(circuit_bra.theta_offset)[1])) 
    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_qubit_ham!(op, state, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
        verbose=verbose))
    println("squared_norm=",squared_norm)


    # Verify the result
    #@test≈ 1.0
    @test isapprox( abs(squared_norm), 1.0, rtol=1e-3)
end


@testset "computation.apply_qubit_ham3" begin
    n_qubit = 4
    # Prepare |Psi> = |00>
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b0011)
    #op = jordan_wigner(FermionOperator("2^ 2"))
    op = jordan_wigner(FermionOperator("1 3^"))
    op += jordan_wigner(FermionOperator("3 1^"))
    # Prepare |phi> = |01>
    state0_bra = QulacsQuantumState(n_qubit)
    set_computational_basis!(state0_bra, 0b0011)

    # Fit <01|U(theta)^dagger c_2^dagger |00>
    
    circuit_bra = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=false)

    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    #update_circuit_param!(circuit_bra, rand(size(circuit_bra.theta_offset)[1])) 
    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_qubit_ham!(op, state, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
        verbose=verbose))
    println("squared_norm=",squared_norm)
    # Verify the result
    #@test≈ 1.0
    @test isapprox( abs(squared_norm), 1.0, rtol=1e-3)
end


@testset "computation.apply_numop" begin
    n_qubit = 2
    # Prepare |Psi> = |11>
    state = QulacsQuantumState(n_qubit)
    set_computational_basis!(state, 0b11)

    # Apply c^dagger_1　c_1 to |Psi>, which will yield |11>.
    op = jordan_wigner(FermionOperator("1^ 1 2^ 2"))
    # Prepare |phi> = |11>
    state0_bra = QulacsQuantumState(n_qubit)
    set_computational_basis!(state0_bra, 0b11)

    # Fit <11|U(theta)^dagger c_1^dagger c_1 |11>
    
    circuit_bra = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=false)

    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    #update_circuit_param!(circuit_bra, rand(size(circuit_bra.theta_offset)[1])) 
    squared_norm = apply_qubit_ham!(op, state, circuit_bra, state0_bra)
    println("squared_norm=",squared_norm)


    # Verify the result
    #@test≈ 1.0
    @test isapprox(abs(squared_norm), 1.0, rtol=1e-3)
end


#=
fitting for ground state of Hamiltonian
first VQE,
then, fitting for groud state obtained by VQE.
=#

@testset "computation.fitting_ground_state" begin
    nsite = 2 
    n_qubit = 2*nsite 
    U = 1.0
    t = -0.01
    μ = 0.0
    ham = generate_ham_1d_hubbard(t, U, nsite, μ)
    op = jordan_wigner(ham)
    n_electron = 2
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron);
    EigVal_min = minimum(eigvals(sparse_mat.toarray()));
    @show EigVal_min
    #vec = eigvecs(sparse_mat.toarray())[:, 1] 
    #@show vec
    # perform VQE for generating reference state (state)
    state0 = create_hf_state(n_qubit, n_electron)
    circuit = uccgsd(n_qubit, orbital_rot=true)
    theta_init = rand(num_theta(circuit))

    # VQE
    cost_history, thetas_opt = 
       QCMaterial.solve_gs(jordan_wigner(ham), circuit, state0, theta_init=theta_init, verbose=true,
           comm=QCMaterial.MPI_COMM_WORLD
       )
    
    update_circuit_param!(circuit, thetas_opt)
    update_quantum_state!(circuit, state0)
    @show  get_vector(state0)

    state0_bra = create_hf_state(n_qubit, n_electron)

    # Fit <11|U(theta)^dagger c_1^dagger c_1 |11>
    circuit_bra = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=false)

    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    #update_circuit_param!(circuit_bra, rand(size(circuit_bra.theta_offset)[1])) 
    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_qubit_ham!(op, state0, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )

    println("squared_norm=", squared_norm)
    # Verify the result
    #@test≈ 1.0
    @test isapprox( abs(squared_norm), abs(EigVal_min), rtol=1e-2)
    state_ = copy(state0_bra)
    update_quantum_state!(circuit_bra, state_)
    #@show  get_vector(state0_bra)
    #@test isapprox(abs.(get_vector(state0)), abs.(get_vector(state_)), rtol=1e-2)
end


#=
VQEでH^2の基底エネルギー|state>を計算する
それを変分量子状態でフィッティングする。
=#

@testset "computation.VQE_H^2" begin
    #H^2
    nsite = 2 
    n_qubit = 2*nsite 
    U = 1.0
    t = -0.01
    μ = 0.0
    ham = generate_ham_1d_hubbard(t, U, nsite, μ)
    @show ham
    #ham = ham^2 #square 
    @show ham
    op = jordan_wigner(ham)
    n_electron = 2
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron);
    EigVal_min = minimum(eigvals(sparse_mat.toarray()));
    @show EigVal_min
    #vec = eigvecs(sparse_mat.toarray())[:, 1] 
    #@show vec
    # perform VQE for generating reference state (state)
    state0 = create_hf_state(n_qubit, n_electron)
    circuit = uccgsd(n_qubit, orbital_rot=true)
    theta_init = rand(num_theta(circuit))

    # VQE
    cost_history, thetas_opt = 
       QCMaterial.solve_gs(jordan_wigner(ham), circuit, state0, theta_init=theta_init, verbose=true,
           comm=QCMaterial.MPI_COMM_WORLD
       )
    
    update_circuit_param!(circuit, thetas_opt)
    update_quantum_state!(circuit, state0)
    #@show  get_vector(state0)
    @test isapprox(cost_history[end], EigVal_min, rtol=1e-3)
end


#=
H |state>
here, |state> is |0011>
we optimize cost = <psi(theta)|H|state>, then |psi(theta)> approx c1=((<psi(theta)|H|state>)) * H|state>
<state|H|state> \approx c1* <state_0bra|psi(theta)> 
=#

@testset "computation.fitting_H_state" begin
    nsite = 2 
    n_qubit = 2*nsite 
    U = 2.0
    t = -1.4
    μ = 1.3
    ε1 = [1.0,-1.0,1.0] 
    V= 1.2
    #ham = generate_ham_1d_hubbard(t, U, nsite, μ)
    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite)
    @show ham
    op = jordan_wigner(ham)
    @show op
    n_electron = 2
    state0 = create_hf_state(n_qubit, n_electron)
    expec_val = get_expectation_value(op, state0)
    @show expec_val

    state0_bra = create_hf_state(n_qubit, n_electron)

    # Fit <11|U(theta)^dagger c_1^dagger c_1 |11>
    circuit_bra = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=true)

    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    #update_circuit_param!(circuit_bra, rand(size(circuit_bra.theta_offset)[1])) 
    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_ham!(op, state0, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )

    println("squared_norm=", squared_norm)
    # Verify the result
    #@test≈ 1.0
    #@test isapprox( abs(squared_norm), abs(EigVal_min), rtol=1e-2)
    state_ = copy(state0_bra)
    update_quantum_state!(circuit_bra, state_)
    #expec_val2  = inner_product(state_, state0_bra)
    c1 = get_transition_amplitude_with_obs(circuit_bra, state0_bra, op, state0)
    #@show c1
    #@test isapprox(squared_norm, c1, rtol=1e-2)
    expec_val1 = inner_product(state_, state0_bra) * c1
    @show expec_val
    @show expec_val1
    @show abs(expec_val - expec_val1)
    #@test isapprox(expec_val,  expec_val1 , rtol=1e-3)
end

#=
H |state>
here, |state> is |0011>
we optimize cost = <psi(theta)|H|state>, then c1=((<psi(theta)|H|state>)) * |psi(theta)> approx H|state>
<state|H|state> \approx c1* <state_0bra|psi(theta)> 
=#


@testset "computation.fitting_c^dag_groundstate" begin
    nsite = 2 
    n_qubit = 2*nsite 
    U = 1.0
    μ = 1.0
    ε1 = [1.0,-1.0,1.0] 
    V= 1.0
    #ham = generate_ham_1d_hubbard(t, U, nsite, μ)
    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite)
    #@show ham
    n_electron = 2
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron);
    EigVal_min = minimum(eigvals(sparse_mat.toarray()));
    ham = jordan_wigner(ham)
    #@show op
    # perform VQE for generating reference state (state)
    state0 = create_hf_state(n_qubit, n_electron)
    circuit = uccgsd(n_qubit, orbital_rot=true)
    theta_init = rand(num_theta(circuit))
    
    # VQE
    cost_history, thetas_opt = 
       QCMaterial.solve_gs(ham, circuit, state0, theta_init=theta_init, verbose=true,
           comm=QCMaterial.MPI_COMM_WORLD
       )
        
    update_circuit_param!(circuit, thetas_opt) #set opt params
    update_quantum_state!(circuit, state0) # approx ground state
    expec_val_g = get_expectation_value(ham, state0)
    @show expec_val_g
    @test isapprox(expec_val_g,  EigVal_min , rtol=1e-3) # OK

    # Apply c^dagger_1 to |Psi>, which will yield |10>.
    op = jordan_wigner(FermionOperator("1^"))
    # Prepare |phi> = |0011>
    n_electron_incremented = 3
    state0_bra = create_hf_state(n_qubit, n_electron_incremented)

    # Fit <1100|U(theta)^dagger c_2^dagger |GS>
    circuit_bra = uccgsd(n_qubit, orbital_rot=true)

    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    #update_circuit_param!(circuit_bra, rand(size(circuit_bra.theta_offset)[1])) 
    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_qubit_op!(op, state0, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
        verbose=verbose))
    println("squared_norm=",squared_norm)

    #ここまででc^dag |GS> の近似的状態をcircuit_bra |state0_bra>で表現した。
    update_quantum_state!(circuit_bra, state0_bra)
   
    #c_1に関する基底状態とフィッティングした状態の遷移確率を計算する。
    divide_real_imag(op::QubitOperator) = (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im

    op2 = jordan_wigner(FermionOperator("1"))
    op_re, op_im = divide_real_imag(op2)
    op_re = get_transition_amplitude(op_re, state0, state0_bra)
    op_im = get_transition_amplitude(op_im, state0, state0_bra)
    #res = get_transition_amplitude(op2, state0, state0_bra)
    #println("res=", res)
    res = op_re + im * op_im
    res = res * squared_norm
    println("res*coeff=",res)

    # 厳密な計算をする。つまり、フィッティングによって得られた値と比較をする。
    # <GS| c_1 c^{dag}_1 |GS>
    op_ = FermionOperator()
    op_ += FermionOperator(" 1  1^", 1.0) #c_1 c^dag_1
    exact_op_ = get_expectation_value(jordan_wigner(op_), state0)
    println("dif=", abs(res - exact_op_))
    @test isapprox(res, exact_op_, rtol=1e-3)

end

@testset "moment_first_order" begin
    Random.seed!(100)
    nsite = 2
    n_qubit = 2 * nsite 
    U = 0.0
    #μ = U/2
    μ = 0.5
    ε1 = [1.0, 1.0] 
    V= 1.0
    #ham = generate_ham_1d_hubbard(t, U, nsite, μ)
    ham_op = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite)
    #@show ham
    n_electron = 2
    sparse_mat = get_number_preserving_sparse_operator(ham_op, n_qubit, n_electron);
    EigVal_min = minimum(eigvals(sparse_mat.toarray()));
    #println("sparse_mat.toarray())=", size(sparse_mat.toarray()))
    #computer ground state
   # state_gs_exact = QulacsQuantumState(n_qubit)
    #vec_gs_exact = eigvecs((sparse_mat.toarray()))[:, 1]
    #e_gs_exact = eigvals((sparse_mat.toarray()))
    #println("e_gs_exact=", e_gs_exact)
    # @show state
    # vec = [0, 1, 2, 3]
    #println(vec_gs_exact)
    #state_load!(state_gs_exact, vec_gs_exact) #exact_ground_state
    #println("state_load QK")
    ham = jordan_wigner(ham_op)
    #@show op
    # perform VQE for generating reference state (state)
    state0 = create_hf_state(n_qubit, n_electron)
    circuit = uccgsd(n_qubit, orbital_rot=true)
    theta_init = rand(num_theta(circuit))

    # VQE
    cost_history, thetas_opt = 
    QCMaterial.solve_gs(ham, circuit, state0, theta_init=theta_init, verbose=true,
        comm=QCMaterial.MPI_COMM_WORLD
    )
        
    update_circuit_param!(circuit, thetas_opt) #set opt params
    update_quantum_state!(circuit, state0) # approx ground state
    #vec_gs = get_vector(state0)
    #println("vec_gs=", vec_gs)

    expec_val_g = get_expectation_value(ham, state0)
    @show expec_val_g
    println("dif_ground_energy=", abs(expec_val_g - EigVal_min))
    @assert isapprox(expec_val_g,  EigVal_min , rtol=1e-3) # OK

    # Apply c^dagger_1 to |Psi>, which will yield |10>.
    op = jordan_wigner(FermionOperator("1^"))
    # Prepare |phi> = |0011>
    n_electron_incremented = 3
    state0_bra = create_hf_state(n_qubit, n_electron_incremented)

    # Fit <1100|U(theta)^dagger c_2^dagger |GS> 
    circuit_bra = uccgsd(n_qubit, orbital_rot=true)
    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    #update_circuit_param!(circuit_bra, rand(size(circuit_bra.theta_offset)[1])) 
    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_qubit_op!(op, state0, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
        verbose=verbose))
    println("squared_norm=",squared_norm)

    #ここまででc^dag |GS> の近似的状態をcircuit_bra |state0_bra>で表現した。
    update_quantum_state!(circuit_bra, state0_bra)

    #c_1に関する基底状態とフィッティングした状態の遷移確率を計算する。
    divide_real_imag(op::QubitOperator) = (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im

    op2 = jordan_wigner(FermionOperator("1"))
    op_re, op_im = divide_real_imag(op2)
    op_re = get_transition_amplitude(op_re, state0, state0_bra)
    op_im = get_transition_amplitude(op_im, state0, state0_bra)
    #res = get_transition_amplitude(op2, state0, state0_bra)
    #println("res=", res)
    res = op_re + im * op_im
    res = res * squared_norm
    println("res*coeff=",res)

    # 厳密な計算をする。つまり、フィッティングによって得られた値と比較をする。
    # <GS| c_1 c^{dag}_1 |GS>
    op_mom0 = FermionOperator()
    op_mom0 += FermionOperator(" 1  1^", 1.0) #c_1 c^dag_1
    op_mom0_re, op_mom0_im = divide_real_imag(jordan_wigner(op_mom0))
    #op_mom0_exact_re = get_transition_amplitude(op_mom0_re, state0, state0)
    #op_mom0_exact_im = get_transition_amplitude(op_mom0_im, state0, state0)
    op_mom0_exact_re = get_transition_amplitude(op_mom0_re, state0, state0)
    op_mom0_exact_im = get_transition_amplitude(op_mom0_im, state0, state0)
    exact_mom0 = op_mom0_exact_re + im * op_mom0_exact_im  
    #exact_mom0 = get_expectation_value(jordan_wigner(op_), state0)
    println("exact_mom0=", exact_mom0)

    println("dif_mom0=", abs(res - exact_mom0)) # 3.4416913763379853e-15 
    @test isapprox(res, exact_mom0, rtol=1e-3)

    ## state_vector
    vec_cdag_gs = get_vector(state0_bra)
    println("vec_mom0=", squared_norm .* vec_cdag_gs)

    ##  1次のモーメント ##
    # <GS| c_1 c^{dag}_1 |GS>
    # H-E0の定義をする。
    ham_op_mom_exact = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite) 
    identity_exact = FermionOperator(" ", EigVal_min)
    ham_op_mom_exact = ham_op_mom_exact - identity_exact
    ham_op_mom1_exact = ham_op_mom_exact^1

    ## exact moments at m=1
    op_c = FermionOperator(" 1", 1.0) #c_1
    op_c = op_c * ham_op_mom1_exact #c H
    op_c = op_c * FermionOperator(" 1^", 1.0) #c H c^dagger
    op_c_re, op_c_im = divide_real_imag(jordan_wigner(op_c))
    op_c_exact_re = get_transition_amplitude(op_c_re, state0, state0)
    op_c_exact_im = get_transition_amplitude(op_c_im, state0, state0)
    exact_mom1 = op_c_exact_re + im * op_c_exact_im  
    println("exact_mom1=", exact_mom1)

    n_electron_incremented = 3
    state1_bra = create_hf_state(n_qubit, n_electron_incremented)
    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))

    # (H-E0)^1
    ham_op_mom = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite) 
    identity = FermionOperator(" ", expec_val_g)
    ham_op_mom = (ham_op_mom - identity)
    #println("ham_op_mom^1=", ham_op_mom3)
    ham_op_mom1 = ham_op_mom^1
    #println("ham_op_mom^$m=", ham_op_mom3)
    #println("ham_op_mom^$m=", ham_op_mom)
    verbose = true
    maxiter = 500
    gtol = 1e-8
    squared_norm1 = apply_qubit_ham!(jordan_wigner(ham_op_mom1), state0_bra, circuit_bra, state1_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
        verbose=verbose))
    println("squared_norm1=", squared_norm1)

    update_quantum_state!(circuit_bra, state1_bra)
    #c_1に関する基底状態とフィッティングした状態の遷移確率を計算する。
    #divide_real_imag(op::QubitOperator) = (op+hermitian_conjugated(op))/2, (op-hermitian_conjugated(op))/2im
    op = jordan_wigner(FermionOperator("1"))
    op_re, op_im = divide_real_imag(op)
    op_re1 = get_transition_amplitude(op_re, state0, state1_bra)
    op_im1 = get_transition_amplitude(op_im, state0, state1_bra)
    mom1 = op_re1 + im * op_im1 

    #coeeffを考慮する。
    mom1 = mom1 * squared_norm1 * squared_norm
    println("squared_norm=", squared_norm)
    println("squared_norm1=", squared_norm1)
    println("mom1_1の計算=", mom1)
    println("mom1_exact=", exact_mom1)
    println("dif_mom1=", abs(mom1 - exact_mom1))
    @test isapprox(mom1, exact_mom1, rtol=1e-3) 
end

@testset "add_control_qubit_for_circuit" begin
    # 動作
    nqubits = 1
    qcircuit = QulacsParametricQuantumCircuit(nqubits)
    add_parametric_RX_gate!(qcircuit, 1, pi/3)
    add_parametric_RZ_gate!(qcircuit, 1, pi/9)
    circuit = QulacsVariationalQuantumCircuit(qcircuit)
    control_index = [2]
    total_num_qubits = 2
    control_circuit_test = add_control_qubit_for_circuit(circuit, control_index, total_num_qubits)
    state_test = QulacsQuantumState(total_num_qubits)
    set_computational_basis!(state_test, 0b11)
    update_quantum_state!(control_circuit_test, state_test)
    # control_circuit = QulacsQuantumCircuit(total_num_qubits)

    control_circuit2 = QulacsQuantumCircuit(total_num_qubits)
    rx_gate = RX(1, pi/3)
    rx_gate = to_matrix_gate(rx_gate)
    rz_gate = RZ(1, pi/9)
    rz_gate = to_matrix_gate(rz_gate)
    control_index = [2]
    #gate_tmp = to_matrix_gate(gate_tmp)
    add_control_qubit!(rx_gate, control_index, 1)
    add_control_qubit!(rz_gate, control_index, 1)
    add_gate!(control_circuit2,  rx_gate)
    add_gate!(control_circuit2,  rz_gate)
    state = QulacsQuantumState(total_num_qubits)
    set_computational_basis!(state, 0b11)
    update_quantum_state!(control_circuit2, state)

    @test get_vector(state) == get_vector(state_test)
end

@testset "make_cotrolled_pauli_gate" begin
        # 動作
    total_nqubits = 3
    pauli_target_list = [1, 2]
    pauli_index_list = [1, 1]
    control_circuit = make_cotrolled_pauli_gate(total_nqubits, pauli_target_list, pauli_index_list)
    state = QulacsQuantumState(total_nqubits)
    set_computational_basis!(state, 0b100)
    update_quantum_state!(control_circuit, state)
    state_exact = QulacsQuantumState(total_nqubits)
    set_computational_basis!(state_exact, 0b111)
    @assert get_vector(state_exact) == get_vector(state)
end

@testset "get_transition_amplitude_sampling_obs_real" begin
    U = 4.0
    V = 1.0
    μ = 1.0
    ε = [1.0, 0.0]
    nsite = 2
    nqubits = nsite * 2
    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite) 
    ham_q = jordan_wigner(ham) 
    #@show ham_q
    circuit_bra = uccgsd(nqubits)
    Random.seed!(120)
    theta_init = rand(num_theta(circuit_bra))
    update_circuit_param!(circuit_bra, theta_init) 
    circuit_ket = uccgsd(nqubits)
    Random.seed!(90)
    theta_init2 = rand(num_theta(circuit_ket))
    update_circuit_param!(circuit_ket, theta_init2) 
    state_augmented = QulacsQuantumState(nqubits+1)
    set_computational_basis!(state_augmented, 0b00011)
    real_shot = get_transition_amplitude_sampling_obs_real(state_augmented, circuit_bra, ham_q, circuit_ket, nshots=2^20)
    @show real_shot
    
    #exact
    circuit_bra = uccgsd(nqubits)
    Random.seed!(120)
    theta_init = rand(num_theta(circuit_bra))
    update_circuit_param!(circuit_bra, theta_init) 
    state0_bra = QulacsQuantumState(nqubits)
    set_computational_basis!(state0_bra, 0b0011)
    circuit_ket = uccgsd(nqubits)
    Random.seed!(90)
    theta_init2 = rand(num_theta(circuit_ket))
    update_circuit_param!(circuit_ket, theta_init2) 
    state_ket = QulacsQuantumState(nqubits)
    set_computational_basis!(state_ket, 0b0011)
    update_quantum_state!(circuit_ket, state_ket)
    exact = get_transition_amplitude_with_obs(circuit_bra, state0_bra, ham_q, state_ket)
    @show real_shot
    @show real(exact)
    @show abs(real_shot - real(exact))
    @test isapprox(real_shot, real(exact), rtol=1e-2)     
end

@testset "get_transition_amplitude_sampling_obs_imag" begin
    U = 4.0
    V = 1.0
    μ = 1.0
    ε = [1.0, 0.0]
    nsite = 2
    nqubits = nsite * 2
    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite) 
    ham_q = jordan_wigner(ham)  
    circuit_bra = uccgsd(nqubits)
    Random.seed!(120)
    theta_init = rand(num_theta(circuit_bra))
    update_circuit_param!(circuit_bra, theta_init) 
    circuit_ket = uccgsd(nqubits)
    Random.seed!(90)
    theta_init2 = rand(num_theta(circuit_ket))
    update_circuit_param!(circuit_ket, theta_init2) 
    state_augmented = QulacsQuantumState(nqubits+1)
    set_computational_basis!(state_augmented, 0b00011)
    nshots = 2^20
    imag_shot = get_transition_amplitude_sampling_obs_imag(state_augmented, circuit_bra, ham_q, circuit_ket, nshots=nshots)
    @show imag_shot

    circuit_bra = uccgsd(nqubits)
    Random.seed!(120)
    theta_init = rand(num_theta(circuit_bra))
    update_circuit_param!(circuit_bra, theta_init) 
    state0_bra = QulacsQuantumState(nqubits)
    set_computational_basis!(state0_bra, 0b0011)
    circuit_ket = uccgsd(nqubits)
    Random.seed!(90)
    theta_init2 = rand(num_theta(circuit_ket))
    update_circuit_param!(circuit_ket, theta_init2) 
    state_ket = QulacsQuantumState(nqubits)
    set_computational_basis!(state_ket, 0b0011)
    update_quantum_state!(circuit_ket, state_ket)

    exact = get_transition_amplitude_with_obs(circuit_bra, state0_bra, ham_q, state_ket)

    @show imag_shot
    @show imag(exact) 
    @show abs(imag_shot - imag(exact))
    @test isapprox(imag_shot, imag(exact), atol=1e-2)      
end


@testset "apply_qubit_ham_sampling" begin
    MPI_COMM_WORLD = MPI.COMM_WORLD
    U = 0.0
    V = 0.0
    μ = 1.0
    ε = [1.0, 0.0]
    nsite = 2
    nqubits = nsite * 2
    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite) 
    ham_q = jordan_wigner(ham)  

    circuit_bra = uccgsd(nqubits)
    Random.seed!(120)
    theta_init = rand(num_theta(circuit_bra))
    update_circuit_param!(circuit_bra, theta_init) 
    circuit_ket = uccgsd(nqubits)
    Random.seed!(90)
    theta_init2 = rand(num_theta(circuit_ket))
    update_circuit_param!(circuit_ket, theta_init2) 
    state_augmented = QulacsQuantumState(nqubits+1)
    set_computational_basis!(state_augmented, 0b00011)

    #cdag = jordan_wigner(FermionOperator("1^"))
    fitting_ham = apply_qubit_ham_sampling!(ham_q, state_augmented, circuit_bra, circuit_ket, nshots=2^20, dx=1e-1)
    @show fitting_ham

    #exact 
    circuit_bra = uccgsd(nqubits)
    Random.seed!(120)
    theta_init = rand(num_theta(circuit_bra))
    update_circuit_param!(circuit_bra, theta_init) 
    state0_bra = QulacsQuantumState(nqubits)
    set_computational_basis!(state0_bra, 0b0011)

    circuit_ket = uccgsd(nqubits)
    Random.seed!(90)
    theta_init2 = rand(num_theta(circuit_ket))
    update_circuit_param!(circuit_ket, theta_init2) 
    state_ket = QulacsQuantumState(nqubits)
    set_computational_basis!(state_ket, 0b0011)
    update_quantum_state!(circuit_ket, state_ket)

    #exact = get_transition_amplitude_with_obs(circuit_bra, state0_bra, ham_q, state_ket)
    verbose = true
    maxiter = 500
    gtol = 1e-8
    squared_norm1 = apply_qubit_ham!(ham_q, state0_bra, circuit_bra, state_ket, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
        verbose=verbose)
        )
    println("squared_norm1=", squared_norm1)

    #@show (fitting_norm_sampling) 
    #@show real(fitting_norm_sampling) 
    @show squared_norm1
    err_abs = abs(real(fitting_ham) - squared_norm1)

    @show err_abs
    err_rel = err_abs/abs(squared_norm1)
    @show err_rel
    @test isapprox(real(fitting_ham), squared_norm1, rtol=1e-1)   
end

@testset "apply_qubit_op_sampling" begin
    MPI_COMM_WORLD = MPI.COMM_WORLD
    U = 0.0
    V = 0.0
    μ = 1.0
    ε = [1.0, 0.0]
    nsite = 2
    nqubits = nsite * 2
    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite) 
    ham_q = jordan_wigner(ham)  

    circuit_bra = uccgsd(nqubits, nx=3)
    Random.seed!(120)
    theta_init = rand(num_theta(circuit_bra))
    update_circuit_param!(circuit_bra, theta_init) 
    circuit_ket = uccgsd(nqubits, nx=2)
    Random.seed!(90)
    theta_init2 = rand(num_theta(circuit_ket))
    update_circuit_param!(circuit_ket, theta_init2) 
    state_augmented = QulacsQuantumState(nqubits+1)
    set_computational_basis!(state_augmented, 0b00000)

    op_q = FermionOperator(" 1^", 1.0) 
    op_q = jordan_wigner(op_q)

    #cdag = jordan_wigner(FermionOperator("1^"))
    fitting_fop = apply_qubit_ham_sampling!(op_q, state_augmented, circuit_bra, circuit_ket, nshots=2^20, dx=1e-1)
    @show fitting_fop

    #exact
    circuit_bra = uccgsd(nqubits)
    Random.seed!(120)
    theta_init = rand(num_theta(circuit_bra))
    update_circuit_param!(circuit_bra, theta_init) 
    state0_bra = QulacsQuantumState(nqubits)
    set_computational_basis!(state0_bra, 0b0111)

    circuit_ket = uccgsd(nqubits)
    Random.seed!(90)
    theta_init2 = rand(num_theta(circuit_ket))
    update_circuit_param!(circuit_ket, theta_init2) 
    state_ket = QulacsQuantumState(nqubits)
    set_computational_basis!(state_ket, 0b0011)
    update_quantum_state!(circuit_ket, state_ket)

    op_q = FermionOperator(" 1^", 1.0) 
    op_q = jordan_wigner(op_q)

    #exact = get_transition_amplitude_with_obs(circuit_bra, state0_bra, ham_q, state_ket)
    verbose = true
    maxiter = 500
    gtol = 1e-8
    squared_norm1 = apply_qubit_op!(op_q, state_ket, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol),
        verbose=verbose)
        ) # obs/ ket / circuit_bra/ state0_bra
    println("squared_norm1=", squared_norm1)

    exact_op_fitting = squared_norm1
    @show exact_op_fitting
    sampling_op_fitting = fitting_fop
    @show sampling_op_fitting
    err_abs = abs(sampling_op_fitting - exact_op_fitting)
    @show err_abs
    err_rel = abs(sampling_op_fitting - exact_op_fitting)/abs(exact_op_fitting)
    @show err_rel
    @test isapprox(sampling_op_fitting, exact_op_fitting, atol=1e-1)   
end

