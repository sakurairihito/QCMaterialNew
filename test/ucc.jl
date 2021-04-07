using Test
using LinearAlgebra
using QCMaterial

import PyCall: pyimport

@testset "ucc.uccgsd" begin
    # construct hamiltonian
    of = pyimport("openfermion")
    ofpyscf = pyimport("openfermionpyscf")
    qulacs = pyimport("qulacs")
    scipy_opt = pyimport("scipy.optimize")
    get_fermion_operator = of.transforms.get_fermion_operator
    jordan_wigner = of.transforms.jordan_wigner
    jw_get_ground_state_at_particle_number = of.linalg.sparse_tools.jw_get_ground_state_at_particle_number
    get_number_preserving_sparse_operator = of.linalg.get_number_preserving_sparse_operator
    FermionOperator = of.ops.operators.FermionOperator

    nsite = 2 
    n_qubit = 2*nsite 
    U = 1.0
    t = -0.01
    μ = 0

    ham = FermionOperator()
    for i in 1:nsite
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^ $(down)^ $(up) $(down)", -U) 
    end

    for i in 1:nsite-1
        ham += FermionOperator("$(up_index(i+1))^ $(up_index(i))", t) 
        ham += FermionOperator("$(up_index(i))^ $(up_index(i+1))", t) 
        ham += FermionOperator("$(down_index(i+1))^ $(down_index(i))", t) 
        ham += FermionOperator("$(down_index(i))^ $(down_index(i+1))", t) 
    end


    for i in 1:nsite
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^  $(up) ", -μ) 
        ham += FermionOperator("$(down)^ $(down)", -μ)
    end

    n_electron = 2　
    @assert mod(n_electron, 2) == 0
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron);　

    enes_ed = eigvals(sparse_mat.toarray());　

    jw_hamiltonian = jordan_wigner(ham);
    qulacs_hamiltonian = qulacs.observable.create_observable_from_openfermion_text(jw_hamiltonian.__str__())
    hfstate(n_qubit, n_electron) = parse(Int, repeat("0", n_qubit-n_electron) * repeat("1", n_electron), base=2)

    circuit, theta_offsets = uccgsd(n_qubit, n_electron÷2, (n_qubit-n_electron)÷2,true)
    function cost(theta_list)
        state = qulacs.QuantumState(n_qubit) 
        state.set_computational_basis(hfstate(n_qubit, n_electron))
        update_circuit_param!(circuit, theta_list, theta_offsets) 
        circuit.update_quantum_state(state) 
        qulacs_hamiltonian.get_expectation_value(state) 
    end

    theta_init = rand(size(theta_offsets)[1])
    cost_history = Float64[] 
    init_theta_list = theta_init
    push!(cost_history, cost(init_theta_list))

    method = "BFGS"
    options = Dict("disp" => true, "maxiter" => 200, "gtol" => 1e-5)
    callback(x) = push!(cost_history, cost(x))
    opt = scipy_opt.minimize(cost, init_theta_list, method=method, callback=callback)

    EigVal_min = minimum(enes_ed)
    #println("EigVal_min=",EigVal_min)
    #println("cost_history_end=",cost_history[end])
    @test abs(EigVal_min-cost_history[end]) < 1e-6 
end
