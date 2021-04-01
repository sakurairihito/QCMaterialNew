import QCMaterial: uccgsd, convert_openfermion_op, up_index, down_index
using PyCall
using LinearAlgebra

plt = pyimport("matplotlib.pyplot")
of = pyimport("openfermion")
ofpyscf = pyimport("openfermionpyscf")
qulacs = pyimport("qulacs")

scipy_opt = pyimport("scipy.optimize")
get_fermion_operator = of.transforms.get_fermion_operator
jordan_wigner = of.transforms.jordan_wigner
jw_get_ground_state_at_particle_number = of.linalg.sparse_tools.jw_get_ground_state_at_particle_number
get_number_preserving_sparse_operator = of.linalg.get_number_preserving_sparse_operator
FermionOperator = of.ops.operators.FermionOperator

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
function eval_energy(circuit, qulacs_hamiltonian, theta_list, theta_offsets, n_qubit)
    # FIXME: n_qubit can be obtained from qulcas_hamiltonian?
    state = qulacs.QuantumState(n_qubit) #|0000> を準備
    state.set_computational_basis(hfstate(n_qubit, n_electron))# |0011>　
    update_circuit_param!(circuit, theta_list, theta_offsets) #量子回路にパラメータをセット
    circuit.update_quantum_state(state) #量子回路を状態に作用
    qulacs_hamiltonian.get_expectation_value(state) #ハミルトニアンの期待値
end

"""
Compute partial derivative of a given function at a point x
"""
function numerical_grad(f, x::Vector{Float64}, dx=1e-8)
    deriv = similar(x)
    x_new1 = copy(x)
    x_new2 = copy(x)
    f_current = f(x)
    for i in eachindex(x)
        x_new1[i] += dx
        x_new2[i] -= dx
        deriv[i] = (f(x_new1) - f(x_new2))/(2*dx)
        x_new1[i] = x_new2[i] = x[i]
    end
    deriv
end

# Some test for numerical_grad
"""
f(x) = x[1] + 2*x[2]
deriv = numerical_grad(f, zeros(2))
println(deriv)
@assert deriv ≈ [1.0, 2.0]
"""

function update_circuit_param!(circuit::PyObject, theta_list, theta_offsets)
    for (idx, theta) in enumerate(theta_list)
        for ioff in 1:theta_offsets[idx][1]
            pauli_coef = theta_offsets[idx][3][ioff]
            circuit.set_parameter(theta_offsets[idx][2]+ioff-1, 
                                  theta*pauli_coef) #量子回路にパラメータをセット
        end
    end
end

function construct_circuit(hamiltonian)
    ham = hamiltonian.ham
    n_qubit = 2*hamiltonian.n_sites

    @assert mod(n_electron, 2) == 0
    @assert n_electron <= hamiltonian.n_sites

    #JW変換
    jw_hamiltonian = jordan_wigner(ham)
    
    #Qulacs用のハミルトニアンの作成
    qulacs_hamiltonian = qulacs.observable.create_observable_from_openfermion_text(jw_hamiltonian.__str__())

    # Prepare a circuit
    circuit, theta_offsets = uccgsd(n_qubit, n_electron÷2, (n_qubit-n_electron)÷2,true)
    circuit, theta_offsets, qulacs_hamiltonian
end

function solve(hamiltonian, n_electron, theta_init=nothing)
    ham = hamiltonian.ham
    n_qubit = 2*hamiltonian.n_sites

    println(ham)　#ハバードモデルハミルトニアンの表示。
    @assert mod(n_electron, 2) == 0
    @assert n_electron <= hamiltonian.n_sites

    # Compute exact ground-state energy
    sparse_mat = get_number_preserving_sparse_operator(ham, n_qubit, n_electron)　#行列の取得
    enes_ed = eigvals(sparse_mat.toarray())　#対角化を行う
    EigVal_min = minimum(enes_ed)
    println("Exact ground-state energy = $(EigVal_min)")

    # Construct circuit
    circuit, theta_offsets, qulacs_hamiltonian = construct_circuit(hamiltonian)
    println("Number of Qubits:", n_qubit)
    println("Number of Electrons:", n_electron)

    # Define a cost function
    cost(theta_list) = eval_energy(circuit, qulacs_hamiltonian, theta_list, theta_offsets, n_qubit)

    # Define the gradient of the cost function
    grad_cost(theta_list) = numerical_grad(cost, theta_list)

    if theta_init === nothing
        theta_init = rand(size(theta_offsets)[1])
    end
    cost_history = Float64[] #コスト関数の箱
    init_theta_list = theta_init
    push!(cost_history, cost(init_theta_list))
    
    method = "BFGS"
    options = Dict("disp" => true, "maxiter" => 100, "gtol" => 1e-5)
    callback(x) = push!(cost_history, cost(x))
    opt = scipy_opt.minimize(cost, init_theta_list, method=method, callback=callback, jac=grad_cost)

    return cost_history, circuit, EigVal_min, opt
end

"""
nsite = 2
ham = generate_ham(nsite)
n_electron = 2
cost_history, circuit, exact_gs_ene, opt = solve(ham, n_electron)
println("cost_history", cost_history)
println(opt["x"])

# TODO: save optimized parameters, i.e., theta_list
import PyPlot
PyPlot.plot(cost_history, color="red", label="VQE")
PyPlot.plot(1:length(cost_history), fill(exact_gs_ene, length(cost_history)),
    linestyle="dashed", color="black", label="Exact Solution")
PyPlot.xlabel("Iteration")
PyPlot.ylabel("Energy expectation value")
PyPlot.legend()
PyPlot.savefig("cost_history.pdf")
"""