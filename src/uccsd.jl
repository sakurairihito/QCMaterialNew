export num_theta,
    num_pauli,
    pauli_coeff,
    theta_offset,
    UCCQuantumCircuit,
    add_parametric_circuit_using_generator!
export update_circuit_param!, update_quantum_state!, gen_t1, gen_p_t2, gen_t1_kucj
export uccgsd
export gen_t2_kucj, kucj, kucj2, gen_t2_kucj_2
export sparse_ansatz

################################################################################
############################# QUANTUM  CIRCUIT #################################
################################################################################
struct UCCQuantumCircuit <: VariationalQuantumCircuit
    circuit::ParametricQuantumCircuit
    thetas::Vector{Float64}
    # Vector of (num_term_count::Int64, ioff::Int64, pauli_coeffs::Float64)
    theta_offsets::Vector{Tuple{Int64,Int64,Vector{Float64}}}
end

#function get_n_qubit(circuit::UCCQuantumCircuit)
#get_n_qubit(circuit.circuit)
#end

function num_theta(circuit::UCCQuantumCircuit)
    size(circuit.theta_offsets)[1]
end

function get_thetas(circuit::UCCQuantumCircuit)
    copy(circuit.thetas)
end

function num_pauli(circuit::UCCQuantumCircuit, idx_theta::Int)
    circuit.theta_offsets[idx_theta][1]
end

function pauli_coeff(circuit::UCCQuantumCircuit, idx_theta::Int, idx_pauli::Int)
    circuit.theta_offsets[idx_theta][3][idx_pauli]
end

function theta_offset(circuit::UCCQuantumCircuit, idx_theta::Int)
    circuit.theta_offsets[idx_theta][2]
end

function UCCQuantumCircuit(n_qubit::Int)
    UCCQuantumCircuit(QulacsParametricQuantumCircuit(n_qubit), [], [])
end

function Base.copy(uc::UCCQuantumCircuit)
    UCCQuantumCircuit(copy(uc.circuit), copy(uc.thetas), copy(uc.theta_offsets))
end

function add_parametric_circuit_using_generator!(
    circuit::UCCQuantumCircuit,
    generator::QubitOperator,
    theta::Float64;
)
    pauli_coeffs = Float64[]
    #println("terms_dict(generator)=", terms_dict(generator))
    for (pauli_str, pauli_coef) in terms_dict(generator)
        pauli_index_list, pauli_id_list = parse_pauli_str(pauli_str)
        #println("pauli_index_list=", pauli_index_list)
        #println("pauli_id_list=", pauli_id_list)
        if !all(1 .<= pauli_index_list)
            error("Pauli indies are out of range!")
        end

        if length(pauli_index_list) == 0
            continue
        end
        # println(pauli_str, pauli_coef)
        pauli_coef = imag(pauli_coef) 
        # coef should be pure imaginary
        # println("pauli_coeff=", pauli_coeff)
        push!(pauli_coeffs, pauli_coef)
        
        add_parametric_multi_Pauli_rotation_gate!(
            circuit.circuit,
            pauli_index_list,
            pauli_id_list,
            pauli_coef * theta,
        )
        #println("add_parametric_multi_Pauli_rotation_gate!",
        #    add_parametric_multi_Pauli_rotation_gate!(
        #        circuit.circuit,
        #        pauli_index_list,
        #        pauli_id_list,
        #        pauli_coef * theta,
        #    )
        #)
    end
    if length(pauli_coeffs) == 0 # isempty(pauli_coeffs) が true とな
        return
        # return parameterinfo
    end
    
    num_thetas = num_theta(circuit)
    # print the position of all the parameters
    println("position of theta=", num_thetas)
    
    push!(parameterinfo, ("x", num_thetas)) # この "x" どうやって決める？

    #=
    初回呼び出し Method1
    julia> parameterinfo = Tuple{String,Int}[]
    julia> add_parametric_circuit_using_generator!(args..., ;parameterinfo)
    julia> @show parameterinfo # 何か入ってればOK

    初回呼び出し Method2
    julia> parameterinfo = add_parametric_circuit_using_generator!(args...,)
    julia> push!(parametricinfo_hairetu, parametricinfo)
    julia> @show parameterinfo_hairetu # 何か入ってればOK
    =#
    #dic = Dict()
    #dic["a"] = num_thetas
    
    ioff =
        num_thetas == 0 ? 0 :
        theta_offset(circuit, num_thetas) + num_pauli(circuit, num_thetas)
    push!(circuit.thetas, theta)
    # push!(circuit.theta)
    push!(circuit.theta_offsets, (get_term_count(generator), ioff, pauli_coeffs))
    #return parameterinfo #length(circuit.thetas)
end

"""
Update circuit parameters
thetas wil be copied.
"""
function update_circuit_param!(circuit::UCCQuantumCircuit, thetas::Vector{Float64})
    if num_theta(circuit) != length(thetas)
        error("Invalid length of thetas!")
    end
    #println("before update_circuit_param!")
    #println("num_theta()=", num_theta(circuit))
    #println("enumerate(thetas)=", enumerate(thetas)) # OK 初期パラメータ
    for (idx, theta) in enumerate(thetas)
        #println("idx=", idx)
        #println("theta=", theta)
        #println("a")
        #println("num_pauli(circuit, idx)=", num_pauli(circuit, idx))
        #println("", )
        for ioff = 1:num_pauli(circuit, idx)
            #println("ioff", ioff)
            pauli_coef = pauli_coeff(circuit, idx, ioff) # ここにIはそもそも存在してない。だから本来４つあったのが３つになっている。
            #println("pauli_coeff=", pauli_coef)
            #println("before set_parameter")
            set_parameter!(
                circuit.circuit,
                theta_offset(circuit, idx) + ioff,
                theta * pauli_coef,
            )
            #println("after set_parameter") #OK
        end
        #println("end of the for ioff") #OK
    end
    #println("before circuit.thetas") # NG
    circuit.thetas .= thetas
    #println("circuit.thetas", circuit.thetas)
end


"""
Update a state using a circuit
"""
function update_quantum_state!(ucccirc::UCCQuantumCircuit, state::QulacsQuantumState)
    update_quantum_state!(ucccirc.circuit, state)
end


"""
Generate single excitations
"""
function gen_t1(a, i)
    #a^\dagger_a a_i (excitation)
    generator = FermionOperator([(a, 1), (i, 0)], 1.0)
    #-a^\dagger_i a_a (de-exciation)
    generator += FermionOperator([(i, 1), (a, 0)], -1.0)
    #JW-transformation of a^\dagger_a a_i - -a^\dagger_i a_a
    jordan_wigner(generator)
end

"""
Generate single excitations for kucj
"""
function gen_t1_kucj(a, i; sgn=1.0)
    #a^\dagger_a a_i (excitation)
    generator = FermionOperator([(a, 1), (i, 0)], sgn)
    # JW-transformation of a^\dagger_a a_i = 
    jordan_wigner(generator)
end

"""
Generate pair dobule excitations
"""
function gen_p_t2(aa, ia, ab, ib)
    generator = FermionOperator([(aa, 1), (ab, 1), (ib, 0), (ia, 0)], 1.0)
    generator += FermionOperator([(ia, 1), (ib, 1), (ab, 0), (aa, 0)], -1.0)
    jordan_wigner(generator)
end

"""
Generate pair double excitations for kucj
"""

function gen_t2_kucj(aa, ab)
    generator = FermionOperator([(aa, 1), (ab, 1), (ab, 0), (aa, 0)], 1.0im)
    jordan_wigner(generator)
end

"""
Generate pair dobule excitations
"""

function gen_t2_kucj_2(aa, ia, ab, ib)
    generator = FermionOperator([(aa, 1), (ab, 1), (ib, 0), (ia, 0)], 1.0im)
    #generator = jordan_wigner(generator)
    println("jordanwigener(generator)=", jordan_wigner(generator))
    println("jordanwigener(generator).pyobj=", jordan_wigner(generator).pyobj)
    println("jordanwigener(generator).pyobj=", jordan_wigner(generator).pyobj.terms)
    return jordan_wigner(generator)
end

"""
Returns UCCGSD circuit.
"""
function uccgsd(
    n_qubit;
    nocc=-1,
    orbital_rot=false,
    conserv_Sz_doubles=true,
    conserv_Sz_singles=true,
    Doubles=true,
    uccgsd=true,
    p_uccgsd=false
)
    if n_qubit <= 0 || n_qubit % 2 != 0
        error("Invalid n_qubit: $(n_qubit)")
    end
    if !orbital_rot && nocc < 0
        error("nocc must be given when orbital_rot = false!")
    end
    circuit = UCCQuantumCircuit(n_qubit)

    norb = n_qubit ÷ 2
    cr_range = orbital_rot ? (1:norb) : (1+nocc:norb)
    anh_range = orbital_rot ? (1:norb) : (1:nocc)

    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)
    sz = [1, -1]
    parameterinfo = []
    # Singles
    for (a_spatial, i_spatial) in (Iterators.product(cr_range, anh_range))
        for ispin1 = 1:2, ispin2 = 1:2
            if conserv_Sz_singles && sz[ispin1] + sz[ispin2] != 0
                continue
            end
            #Spatial Orbital Indices
            a_spin_orbital = so_idx(a_spatial, ispin1)
            i_spin_orbital = so_idx(i_spatial, ispin2)
            #t1 operator
            generator = gen_t1(a_spin_orbital, i_spin_orbital)
            #Add t1 into the circuit
            #parameterinfo = add~
            # 
            #add_parametric_circuit_using_generator!(circuit, generator, 0.0)
            #a = (a_spin_orbital, i_spin_orbital) => length(circuit.thetas) # Pair という型があるよ
            # Dict("a"=>1)
            #a = Dict([("a_spin_orbital+i_spin_orbital", length(circuit.thetas))])
            #push!(parameterinfo, a)
            
        end
    end

    if Doubles
        if uccgsd
            #Doubles
            for (spin_a, spin_i, spin_b, spin_j) in Iterators.product(1:2, 1:2, 1:2, 1:2)
                for (a, i, b, j) in Iterators.product(1:norb, 1:norb, 1:norb, 1:norb)
                    if conserv_Sz_doubles &&
                       sz[spin_a] - sz[spin_i] + sz[spin_b] - sz[spin_j] != 0
                        continue
                    end
                    #Spatial Orbital Indices
                    aa = so_idx(a, spin_a)
                    ia = so_idx(i, spin_i)
                    bb = so_idx(b, spin_b)
                    jb = so_idx(j, spin_j)
                    #perform loop only if ia>jb && aa>bb               
                    if aa <= bb || ia <= jb
                        continue
                    end
                    #t2 operator
                    generator = gen_p_t2(aa, ia, bb, jb)
                    #Add p-t2 into the circuit
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                end
            end
        end

        if p_uccgsd
            spin_a = 1
            spin_b = 2
            for (a, i) in Iterators.product(1:norb, 1:norb)
                aa = so_idx(a, spin_a)
                ia = so_idx(i, spin_a)
                bb = so_idx(a, spin_b)
                jb = so_idx(i, spin_b)
                generator = gen_p_t2(aa, ia, bb, jb)
                #Add p-t2 into the circuit
                add_parametric_circuit_using_generator!(circuit, generator, 0.0)
            end
        end
    end
    println("num_thetas=", num_theta(circuit))
    circuit
end





"""
Returns k-ucj circuit.
"""
function kucj(n_qubit; conserv_Sz_doubles=true, k=1, sparse=true)
    if n_qubit <= 0 || n_qubit % 2 != 0
        error("Invalid n_qubit: $(n_qubit)")
    end

    circuit = UCCQuantumCircuit(n_qubit)

    norb = n_qubit ÷ 2

    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)
    sz = [1, -1]
    
    for i = 1:k
        parameterinfo = []
        # exp(-K) where K is an orbital rotation operator
        for (a_spatial, i_spatial) in (Iterators.product(1:norb, 1:norb))
            for ispin1 = 1:2
                #if conserv_Sz_singles 
                #    continue
                #end
                #Spatial Orbital Indices
                # 
                a_spin_orbital = so_idx(a_spatial, ispin1) #a_spatial => p, #a_spatial=1, ispin1=1, res= 1
                i_spin_orbital = so_idx(i_spatial, ispin1) #i_spatial => q, #i_spatial=2, ispin1=1, res= 3

                #a_spin_orbital = 1
                #i_spin_orbital = 3

                #a_spin_orbital = 3
                #i_spin_orbital = 1
                #t1 operator
                generator = gen_t1_kucj(a_spin_orbital, i_spin_orbital, sgn=-1.0)
                #Add t1 into the circuit

                add_parametric_circuit_using_generator!(circuit, generator, 0.0) # 回転角度をマイナスをしたい！generatorにマイナス符号をつける
                
                #a = (a_spin_orbital, i_spin_orbital) => length(circuit.thetas) # Pair という型がある
                a = (a_spin_orbital, i_spin_orbital) => num_theta(circuit) 
                
                #=
                paraminfo = []
                ref_orbits = []
                for (a, i) in [(1, 2), (1, 3), (2, 3)]
                    push!(ref_orbits, (a, i))
                    add_parametric_circuit_using_generator!（てきとーな儀式）
                    # circuit.thetas が作られる
                    memo = (a, i) => num_theta(circuit)
                    push!(paraminfo, memo)
                end

                # この時点で paraminfo は
                # [(1, 2) => 1), (1, 3) => 2), (2, 3) => 3)]

                for (a, i) in [(2, 1), (3, 1), (3, 2)]
                    mukasi_a = i # 1 if i == 1
                    mukasi_i = a # 2 if a == 2
                    add_parametric_circuit_using_generator!（てきとーな儀式）
                    # circuit.thetas が作られる
                    push!(paraminfo, ((mukasi_a, mukasi_i) =>  num_theta(circuit))
                end

                # この時点で paraminfo は
                # [(1, 2) => 1), (1, 3) => 2) (2, 3) => 3), (1, 2) => 4), (1, 3) => 5), (2, 3) => 6)]
                update(paraminfo) を実行
                # [(1, 2) => 1), (1, 3) => 2) (2, 3) => 3), (1, 2) => 1), (1, 3) => 2), (2, 3) => 3)]
                # if possible
                # [(1, 2) => 1), (1, 3) => 2) (2, 3) => 3), (2, 1) => 1), (3, 1) => 2), (3, 2) => 3)]
                =#

                # Dict("a"=>1)
                #a = Dict([("a_spin_orbital+i_spin_orbital", length(circuit.thetas))])
                push!(parameterinfo, a)
            end
        end
        
        #parameter_position = Dict(parameterinfo)
        # spin2length = Dict(parameterinfo)
        # v12 = spin2length[(1,2)]
        # v13 = spin2length[(1,3)]

        #Doubles
        for (spin_a, spin_i, spin_b, spin_j) in Iterators.product(1:2, 1:2, 1:2, 1:2)
            for (a, i, b, j) in Iterators.product(1:norb, 1:norb, 1:norb, 1:norb)
                if conserv_Sz_doubles && sz[spin_a] - sz[spin_i] + sz[spin_b] - sz[spin_j] != 0
                    continue
                end

                #Spatial Orbital Indices
                aa = so_idx(a, spin_a)
                ia = so_idx(i, spin_i)
                bb = so_idx(b, spin_b)
                jb = so_idx(j, spin_j)

                #perform loop only if ia>jb && aa>bb
                if aa <= bb || ia <= jb
                    continue
                end

                #perform loop only if aa=ia && bb=jb
                if aa != ia || bb != jb
                    continue
                end

                #A = [aa ia bb jb]
                #if sparse && in(1, A) == false && in(2, A) == false
                #    continue
                #end
                #if sparse && 
                #t2 operator
                generator = gen_t2_kucj_2(aa, ia, bb, jb)
                # remove rm_Identity
                #println("generator_before_remove_identity=")
                #println("generator=", generator)
                generator = rm_identity(generator)
                #println("generator_remove_identity=")
                #println("generator_remove_identity=", generator)
                #Add p-t2 into the circuit
                add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                #println("generator=", generator)
            end
        end

        # exp(K) where K is an orbital rotation operator
        for (a_spatial, i_spatial) in (Iterators.product(1:norb, 1:norb))
            for ispin1 = 1:2
                #Spatial Orbital Indices
                a_spin_orbital = so_idx(a_spatial, ispin1)
                i_spin_orbital = so_idx(i_spatial, ispin1)
                #t1 operator
                generator = gen_t1_kucj(a_spin_orbital, i_spin_orbital)
                #Add t1 into the circuit
                # ((1,2), 1), ((1,3),2),,,,,, ((1,2), 100->1), ((1,3), 101->2)
                add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                #parameter_position = parameter_position[(a_spin_orbital, i_spin_orbital)]
                # 
                b = (a_spin_orbital, i_spin_orbital) => num_theta(circuit) 
                # ((1,2), 1), ((1,3), 2), ((1,2), 3), ((1,3), 4)
                # thetas = [1.2, 1.3, -1.3, 0.2]
                # ((1,2), 1.2),((1,3), 1.3), ()
                # => thetas_after = [1.2, 1.3,1.2, 1.3]
                push!(parameterinfo, b)
                #circuit.thetas[end] = parameter_position
                #parameter position = spin2length[(a_spin_orbital, i_spin_orbital)]
                #v13 = spin2length[(1,3)]
            end
        end
    end
    println("num_thetas=", num_theta(circuit))
    circuit
end

"""
Returns sparse circuit based on impurity model.
"""
function sparse_ansatz(
    n_qubit;
    nocc=-1,
    orbital_rot=false,
    conserv_Sz_doubles=true,
    conserv_Sz_singles=true,
    Doubles=true,
    uccgsd=true,
    p_uccgsd=false
)
    if n_qubit <= 0 || n_qubit % 2 != 0
        error("Invalid n_qubit: $(n_qubit)")
    end
    if !orbital_rot && nocc < 0
        error("nocc must be given when orbital_rot = false!")
    end
    circuit = UCCQuantumCircuit(n_qubit)

    norb = n_qubit ÷ 2
    cr_range = orbital_rot ? (1:norb) : (1+nocc:norb)
    anh_range = orbital_rot ? (1:norb) : (1:nocc)

    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)
    sz = [1, -1]

    # Singles
    for (a_spatial, i_spatial) in (Iterators.product(cr_range, anh_range))
        for ispin1 = 1:2, ispin2 = 1:2
            if conserv_Sz_singles && sz[ispin1] + sz[ispin2] != 0
                continue
            end
            #Spatial Orbital Indices
            a_spin_orbital = so_idx(a_spatial, ispin1)
            i_spin_orbital = so_idx(i_spatial, ispin2)
            #t1 operator
            generator = gen_t1(a_spin_orbital, i_spin_orbital)
            #Add t1 into the circuit
            add_parametric_circuit_using_generator!(circuit, generator, 0.0)
        end
    end

    if Doubles
        if uccgsd
            #Doubles
            for (spin_a, spin_i, spin_b, spin_j) in Iterators.product(1:2, 1:2, 1:2, 1:2)
                for (a, i, b, j) in Iterators.product(1:norb, 1:norb, 1:norb, 1:norb)
                    #if conserv_Sz_doubles &&
                    #   sz[spin_a] + sz[spin_i] + sz[spin_b] + sz[spin_j] != 0
                    #    continue
                    #end
                    if conserv_Sz_doubles &&
                       sz[spin_a] - sz[spin_i] + sz[spin_b] - sz[spin_j] != 0
                        continue
                    end
                    #Spatial Orbital Indices
                    aa = so_idx(a, spin_a)
                    ia = so_idx(i, spin_i)
                    bb = so_idx(b, spin_b)
                    jb = so_idx(j, spin_j)
                    #perform loop only if ia>jb && aa>bb               
                    if aa <= bb || ia <= jb
                        continue
                    end
                    A = [aa ia bb jb]
                    #if in(1, A) == false && in(2, A) == false
                    #    continue
                    #end
                    if count(i -> (1 <= i <= 2), A) <= 1 #in(1, A) == false && in(2, A) == false
                        #println("Aの中に2つ以上のimpurityのスピン軌道が含まれる（４サイトの場合）.")
                        continue
                    end
                    #t2 operator
                    generator = gen_p_t2(aa, ia, bb, jb)
                    #Add p-t2 into the circuit
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                end
            end
        end

        if p_uccgsd
            spin_a = 1
            spin_b = 2
            for (a, i) in Iterators.product(1:norb, 1:norb)
                aa = so_idx(a, spin_a)
                ia = so_idx(i, spin_a)
                bb = so_idx(a, spin_b)
                jb = so_idx(i, spin_b)
                generator = gen_p_t2(aa, ia, bb, jb)
                #Add p-t2 into the circuit
                add_parametric_circuit_using_generator!(circuit, generator, 0.0)
            end
        end
    end
    println("num_thetas=", num_theta(circuit))
    circuit
end

