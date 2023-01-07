using Test
using QCMaterial
using LinearAlgebra

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
    squared_norm = apply_ham!(op, state, circuit_bra, state0_bra)
    println("squared_norm=",squared_norm)


    # Verify the result
    #@test≈ 1.0
    @test isapprox( abs(squared_norm), 1.0, rtol=1e-3)
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
    squared_norm = apply_ham!(op, state0, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )

    println("squared_norm=", squared_norm)
    # Verify the result
    #@test≈ 1.0
    @test isapprox( abs(squared_norm), abs(EigVal_min), rtol=1e-2)
    state_ = copy(state0_bra)
    update_quantum_state!(circuit_bra, state_)
    #@show  get_vector(state0_bra)
    @test isapprox(abs.(get_vector(state0)), abs.(get_vector(state_)), rtol=1e-2)
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

@testset "computation.fitting_H_c^dag_groundstate" begin
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

    n_electron_incremented = 3
    state1_bra = create_hf_state(n_qubit, n_electron_incremented)
    #update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
   
    #c_1に関する、基底状態とフィッティングした状態の間の遷移確率を計算する。
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
    op_ += FermionOperator(" 1^", 1.0) #c_1 c^dag_1
    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε1, nsite)
    op_ *= ham 
    op_ *= FermionOperator(" 1", 1.0) 
    exact_op_ = get_expectation_value(jordan_wigner(op_), state0)
    println("dif=", abs(res - exact_op_))
    @test isapprox(res, exact_op_, rtol=1e-3)
end


@testset "computation.fitting_ham^2_state" begin
    nsite = 2 
    n_qubit = 2*nsite 
    U = 2.0
    t = 0.1
    μ = 1.3
    ham = generate_ham_1d_hubbard(t, U, nsite, μ)
    ham2 = ham^2
    op = jordan_wigner(ham)
    op2 = jordan_wigner(ham2)
    n_electron = 1

    state0 = create_hf_state(n_qubit, n_electron)
    expec_val = get_expectation_value(op2, state0) #<psi|H^2|psi>
    
    state0_bra = create_hf_state(n_qubit, n_electron)
    circuit_bra = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=true)
    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))

    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_ham!(op, state0, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )
    state_h = copy(state0_bra)
    update_quantum_state!(circuit_bra, state_h)
    #expec_val2  = inner_product(state_, state0)
    c1 = get_transition_amplitude_with_obs(circuit_bra, state0_bra, op, state0)
    @show c1

    # second round => H^2 fitting
    op = jordan_wigner(ham * c1)
    state0_bra2 = create_hf_state(n_qubit, n_electron)
    circuit_bra2 = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=true)
    update_circuit_param!(circuit_bra2, rand(num_theta(circuit_bra2)))
    squared_norm = apply_ham!(op, state_h, circuit_bra2, state0_bra2, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )
    state_h2 = copy(state0_bra)
    update_quantum_state!(circuit_bra2, state_h2)
    c2 = get_transition_amplitude_with_obs(circuit_bra2, state0_bra2, op, state0)
    @show c2
    expec_val2  = inner_product(state_h2, state_h) * c2
    @show expec_val2
    expec_val2_  = inner_product(state_h2, state0_bra) * c2  
    @show expec_val2_
    #@show expect_val2
    @show expec_val2
    @show expec_val
    @test isapprox(expec_val,  expec_val2 , rtol=1e-1)
end



@testset "computation.fitting_ham^2_state_part2" begin
    nsite = 2 
    n_qubit = 2*nsite 
    U = 2.0
    t = 1.1
    μ = 1.0
    ham = generate_ham_1d_hubbard(t, U, nsite, μ)
    ham2 = ham^2
    op = jordan_wigner(ham)
    op2 = jordan_wigner(ham2)
    n_electron = 2

    state0 = create_hf_state(n_qubit, n_electron)
    expec_val = get_expectation_value(op2, state0) #<psi|H^2|psi>
    
    state0_bra = create_hf_state(n_qubit, n_electron)
    circuit_bra = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=true)
    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))

    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_ham!(op, state0, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )
    state_h = copy(state0_bra)
    update_quantum_state!(circuit_bra, state_h)
    #expec_val2  = inner_product(state_, state0)
    c1 = get_transition_amplitude_with_obs(circuit_bra, state0_bra, op, state0)
    @show c1

    # second round => H^2 fitting
    op = jordan_wigner(ham * c1)
    state0_bra2 = create_hf_state(n_qubit, n_electron)
    circuit_bra2 = uccgsd(n_qubit, orbital_rot=true)
    update_circuit_param!(circuit_bra2, rand(num_theta(circuit_bra2)))
    squared_norm = apply_ham!(op, state_h, circuit_bra2, state0_bra2, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )
    state_h2 = copy(state0_bra)
    update_quantum_state!(circuit_bra2, state_h2)
    c2 = get_transition_amplitude_with_obs(circuit_bra2, state0_bra2, op, state0)
    @show c2
    expec_val2  = inner_product(state_h2, state_h) * c2
    @show expec_val2
    expec_val2_  = inner_product(state_h2, state0) * c2  
    @show expec_val2_
    @show abs(expec_val)
    @show abs(expec_val2)
    @show abs(expec_val - expec_val2)
    @test isapprox(expec_val,  expec_val2 , rtol=1e-1)
end


@testset "computation.fitting_ham^3_state_part2" begin
    nsite = 2 
    n_qubit = 2*nsite 
    U = 2.0
    t = 0.1
    μ = 1.0
    ham = generate_ham_1d_hubbard(t, U, nsite, μ)
    ham3 = ham^3
    #ham2 = ham^2 
    op = jordan_wigner(ham)
    op3 = jordan_wigner(ham3)
    #op2 = jordan_wigner(ham2)
    n_electron = 2

    state0 = create_hf_state(n_qubit, n_electron)
    expec_val = get_expectation_value(op3, state0) #<psi|H^2|psi>
    
    state0_bra = create_hf_state(n_qubit, n_electron)
    circuit_bra = uccgsd(n_qubit, orbital_rot=true)
    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))

    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_ham!(op, state0, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )
    state_h = copy(state0_bra)
    update_quantum_state!(circuit_bra, state_h)
    #expec_val2  = inner_product(state_, state0)
    c1 = get_transition_amplitude_with_obs(circuit_bra, state0_bra, op, state0)

    # second round => H^2 fitting
    op = jordan_wigner(ham * c1)
    #state0_bra2 = create_hf_state(n_qubit, n_electron)
    #circuit_bra2 = uccgsd(n_qubit, orbital_rot=true, conserv_Sz_singles=false)
    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    squared_norm = apply_ham!(op, state_h, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )
    state_h2 = copy(state0_bra)
    update_quantum_state!(circuit_bra, state_h2)
    c2 = get_transition_amplitude_with_obs(circuit_bra, state0_bra, op, state0)


    #third round
    op = jordan_wigner(ham * c2 )
    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    squared_norm = apply_ham!(op, state_h2, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )
    state_h3 = copy(state0_bra)
    update_quantum_state!(circuit_bra, state_h3)
    c3 = get_transition_amplitude_with_obs(circuit_bra, state0_bra, op, state0)
    expec_val3_  = inner_product(state_h3, state_h2) * c3 
    #@show expec_val3_
    @show abs(expec_val - expec_val3_)
    @show abs(expec_val)
    @show abs(expec_val3_)
    @test isapprox(expec_val,  expec_val3_ , rtol=1)
end

#=
H^2 のフィッティングについて
H|state> = c1 |phi1*>
c = <phi1*|H|state>
H^2|state> = Hc1|phi1*> = c2|phi2*>
c2 = <phi2*|Hc1|state>
<state|H^2|state> = <phi1*| phi2*> * c2
=#

@testset "computation.fitting_ham^4_state" begin
    nsite = 2 
    n_qubit = 2*nsite 
    U = 2.0
    t = 0.1
    μ = 1.0
    ham = generate_ham_1d_hubbard(t, U, nsite, μ)
    ham4 = ham^4
    op = jordan_wigner(ham)
    op4 = jordan_wigner(ham4)
    n_electron = 2

    state0 = create_hf_state(n_qubit, n_electron)
    expec_val = get_expectation_value(op4, state0) #<psi|H^4|psi>
    
    # first round
    state0_bra = create_hf_state(n_qubit, n_electron)
    circuit_bra = uccgsd(n_qubit, orbital_rot=true)
    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))

    verbose = true
    maxiter = 300
    gtol = 1e-8
    squared_norm = apply_ham!(op, state0, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )
    state_h = copy(state0_bra)
    update_quantum_state!(circuit_bra, state_h)
    #expec_val1  = inner_product(state_h, state0)
    c1 = get_transition_amplitude_with_obs(circuit_bra, state0_bra, op, state0)
    # return c1, expec_val1, state_h

    # second round => H^2 fitting
    op = jordan_wigner(ham * c1)
    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    squared_norm = apply_ham!(op, state_h, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )
    state_h2 = copy(state0_bra)
    update_quantum_state!(circuit_bra, state_h2)
    c2 = get_transition_amplitude_with_obs(circuit_bra, state0_bra, op, state0)
    # expec_val2  = inner_product(state_h2, state_h) * c2
    #return c2, expect_val2, state_h2

    #third round
    op = jordan_wigner(ham * c2 )
    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    squared_norm = apply_ham!(op, state_h2, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )
    state_h3 = copy(state0_bra)
    update_quantum_state!(circuit_bra, state_h3)
    c3 = get_transition_amplitude_with_obs(circuit_bra, state0_bra, op, state0)
    expec_val3  = inner_product(state_h3, state_h2) * c3 
    #return c3, expec_val3, state_h3
    
    # forth round
    op = jordan_wigner(ham * c3 )
    update_circuit_param!(circuit_bra, rand(num_theta(circuit_bra)))
    squared_norm = apply_ham!(op, state_h3, circuit_bra, state0_bra, minimizer=QCMaterial.mk_scipy_minimize(
        options = Dict("disp" => verbose, "maxiter" => maxiter, "gtol" => gtol))
        )
    state_h4 = copy(state0_bra)
    update_quantum_state!(circuit_bra, state_h4)
    c4 = get_transition_amplitude_with_obs(circuit_bra, state0_bra, op, state0)
    expec_val4  = inner_product(state_h4, state_h3) * c4
    @show abs(expec_val - expec_val4)
    @test isapprox(expec_val,  expec_val4 , rtol=1e-2)
end
