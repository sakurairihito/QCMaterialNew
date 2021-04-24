using QCMaterial
using PyCall
using LinearAlgebra

scipy_opt = pyimport("scipy.optimize")
struct Hamiltonian
    ham
    n_sites
end

hfstate(n_qubit, n_electron) = parse(Int, repeat("0", n_qubit-n_electron) * repeat("1", n_electron), base=2)

function generate_ham(nsite)
    # ## 2-site Hubbard modelのハミルトニアンを定義する。
    # 
    # $H=-t \sum_{\langle i, j>\sigma=(\uparrow, \downarrow)} \sum_{i, \sigma}\left(a_{i, \sigma}^{\dagger} a_{j, \sigma}+a_{j, \sigma}^{\dagger} a_{i, \sigma}\right)+U \sum_{i} n_{i, \uparrow} n_{i, \downarrow}-\mu \sum_{i} \sum_{\sigma=(\uparrow, \downarrow)} n_{i, \sigma}$
    # 
    # $1_\uparrow$->0,$1_\downarrow$->1, $2_\uparrow$->2, $2_\downarrow$->3
    # 
    # クーロン斥力(U=2): 2 * $n_{1,\uparrow}$ $n_{1,\downarrow}$ + 2 * $n_{2,\uparrow}$ $n_{2,\downarrow}$ 
    #                = 2 * $c^\dagger_0$ $c_0$ $c^\dagger_1$ $c_1$ + 2 * $c^\dagger_2$ $c_2$ $c^\dagger_3$ $c_3$
    #  
    # ホッピング項(t=-0.1): -0.1 * $c^\dagger_{1,\uparrow}$ $c_{2,\uparrow}$ -0.1 * $c^\dagger_{2,\uparrow}$ $c_{1,\uparrow}$
    #                    -0.1 * $c^\dagger_{1,\downarrow}$ $c_{2,\downarrow}$ -0.1 * $c^\dagger_{2,\downarrow}$ $c_{1,\downarrow}$
    #   = -0.1 * $c^\dagger_0$ $c_2$ + -0.1 * $c^\dagger_2$ $c_0$ + -0.1 * $c^\dagger_1$ $c_3$ + -0.1 * $c_3^\dagger$ $c_1$
    # 
    # ケミカルポテンシャル項(μ=U/2=1):-1 * $n_{1,\uparrow}$ - 1 * $n_{1,\downarrow}$ - 1 * $n_{2,\uparrow}$ - 1 * $n_{2,\downarrow}$ 
    #                = - 1 * $c^\dagger_0$ $c_0$ - 1 * $c^\dagger_1$ $c_1$ - 1 * $c^\dagger_2$ $c_2$ - 1 * $c^\dagger_3$ $c_3$
    
    #ハバードモデルハミルトニアンを定義し、対角化まで行う
    n_qubit = 2*nsite #量子ビットの数
    U = 10.0
    t = -1.0

    #ハーフフィリングを仮定(電子数 = サイトの数)するとケミカルポテンシャルはμ=U/2.マイナスがつくので以下。
    μ = U/2
    
    ham = FermionOperator()
    #斥力項
    for i in 1:nsite
        #up_index,down_indexの定義は、QC_materialを参照。
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^ $(down)^ $(up) $(down)", -U) #左側に生成演算子。右際に消滅演算子をもっていく過程で半交換関係が1回でマイナスをつける。
    end
    #ホッピング項
    for i in 1:nsite-1
        ham += FermionOperator("$(up_index(i+1))^ $(up_index(i))", t) 
        ham += FermionOperator("$(up_index(i))^ $(up_index(i+1))", t) 
        ham += FermionOperator("$(down_index(i+1))^ $(down_index(i))", t) 
        ham += FermionOperator("$(down_index(i))^ $(down_index(i+1))", t) 
    end

    #ケミカルポテンシャルの項
    for i in 1:nsite
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^  $(up) ", -μ) 
        ham += FermionOperator("$(down)^ $(down)", -μ)
    end

    Hamiltonian(ham, nsite)
end

"""
Evaluate energy
"""
function eval_energy(circuit, ham_jw, theta_list, n_qubit)
    state = create_hf_state(n_qubit, n_electron) #|0000> を準備
    update_circuit_param!(circuit, theta_list) #量子回路にパラメータをセット
    update_quantum_state!(circuit, state) #量子回路を状態に作用
    get_expectation_value(ham_jw, state) #ハミルトニアンの期待値
end


function construct_circuit(hamiltonian)
    ham = hamiltonian.ham
    n_qubit = 2*hamiltonian.n_sites

    @assert mod(n_electron, 2) == 0
    @assert n_electron <= hamiltonian.n_sites

    #JW変換
    ham_jw = jordan_wigner(ham)

    # Prepare a circuit
    circuit = uccgsd(n_qubit, orbital_rot=true)
    circuit, ham_jw
end

function solve(hamiltonian, n_electron;theta_init=nothing, comm=nothing)
    if comm === nothing
        rank = 0
    else
        rank = MPI.Comm_rank(comm)
    end

    ham = hamiltonian.ham
    n_qubit = 2*hamiltonian.n_sites

    println(ham)　#ハバードモデルハミルトニアンの表示。
    @assert mod(n_electron, 2) == 0
    @assert n_electron <= hamiltonian.n_sites

    # Compute exact ground-state energy
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron)　#行列の取得
    enes_ed = eigvals(sparse_mat.toarray())　#対角化を行う
    EigVal_min = minimum(enes_ed)
    if rank == 0
        println("Exact ground-state energy = $(EigVal_min)")
    end

    # Construct circuit
    circuit, ham_jw = construct_circuit(hamiltonian)
    if rank == 0
        println("Number of Qubits:", n_qubit)
        println("Number of Electrons:", n_electron)
    end

    # Define a cost function
    function cost(theta_list)
        eval_energy(circuit, ham_jw, theta_list, n_qubit)
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
        if rank == 0
            println("g: ", (t2-t1)*1e-9)
        end
        res
    end

    if theta_init === nothing
        theta_init = rand(size(circuit.theta_offsets)[1])
        if comm !== nothing
            # Make sure all processes use the same initial values
            MPI.Bcast!(theta_init, 0, comm)
        end
    end
    cost_history = Float64[] #コスト関数の箱
    init_theta_list = theta_init
    push!(cost_history, cost(init_theta_list))
    
    method = "BFGS"
    options = Dict("disp" => true, "maxiter" => 200, "gtol" => 1e-5)
    function callback(x)
        push!(cost_history, cost(x))
        if rank == 0
            println("iter ", length(cost_history), " ", cost_history[end])
        end
    end
    opt = scipy_opt.minimize(cost, init_theta_list, method=method, callback=callback, jac=grad_cost,
        options=options)

    return cost_history, circuit, EigVal_min, opt
end
