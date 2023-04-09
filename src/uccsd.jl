export num_theta,
    num_pauli,
    pauli_coeff,
    theta_offset,
    UCCQuantumCircuit,
    add_parametric_circuit_using_generator!
export update_circuit_param!, update_quantum_state!, gen_t1, gen_p_t2, gen_t1_kucj_real, gen_t1_kucj_imag
export uccgsd
export gen_t2_kucj, kucj, kucj2, gen_t2_kucj_2
export sparse_ansatz
export gen_t1_diag_kucj, orb_rot

################################################################################
############################# QUANTUM  CIRCUIT #################################
################################################################################
struct UCCQuantumCircuit <: VariationalQuantumCircuit
    circuit::ParametricQuantumCircuit
    thetas::Vector{Float64}
    # Vector of (num_term_count::Int64, ioff::Int64, pauli_coeffs::Float64)
    theta_offsets::Vector{Tuple{Int64,Int64,Vector{Float64}}}
end

"""
julia> qpqc = QulacsParametricQuantumCircuit(適当な引数与えてね)
julia> dosome(qpqc, idx) # qpqc.circuit.pyobj.add_H_gate!(circuit, idx) とおなじ
"""

function add_H_gate!(vqc::VariationalQuantumCircuit, idx::Int)
    vqc.circuit.pyobj.add_H_gate(idx-1)
end

function add_X_gate!(vqc::VariationalQuantumCircuit, idx::Int)
    vqc.circuit.pyobj.add_X_gate(idx-1)
end


function add_Sdag_gate!(vqc::VariationalQuantumCircuit, idx::Int)
    vqc.circuit.pyobj.add_Sdag_gate(idx-1)
end

function add_S_gate!(vqc::VariationalQuantumCircuit, idx::Int)
    vqc.circuit.pyobj.add_S_gate(idx-1)
end

function get_gate_count(uc::UCCQuantumCircuit)
    uc.circuit.pyobj.get_gate_count()
end

function get_gate(uc::UCCQuantumCircuit, idx_gate::Int)
    QulacsGate(uc.circuit.pyobj.get_gate(idx_gate-1))
end 
#function get_n_qubit(circuit::UCCQuantumCircuit)
#    get_n_qubit(circuit.circuit)
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
    #println("position of theta=", num_thetas)

    #push!(parameterinfo, ("x", num_thetas)) # この "x" どうやって決める？

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
we separate imaginary part and real part of parameters
sgn1 and sgn2 means sign for single particle excitation operator e^{-K}
"""

function gen_t1_diag_kucj(a; sgn=1.0im)
    #a^\dagger_a a_i (excitation)
    generator = FermionOperator([(a, 1), (a, 0)], sgn)
    generator += FermionOperator([(a + 1, 1), (a + 1, 0)], sgn)
    # JW-transformation of a^\dagger_a a_i = 
    jordan_wigner(generator)
end


function gen_t1_kucj_real(a, i; sgn1=1.0, sgn2=-1.0)
    #a^\dagger_a a_i (excitation)
    generator = FermionOperator([(a, 1), (i, 0)], sgn1)
    generator += FermionOperator([(i, 1), (a, 0)], sgn2)
    # JW-transformation of a^\dagger_a a_i = 
    jordan_wigner(generator)
end

function gen_t1_kucj_imag(a, i; sgn1=1.0im, sgn2=-1.0im)
    #a^\dagger_a a_i (excitation)
    generator = FermionOperator([(a, 1), (i, 0)], sgn1)
    generator += FermionOperator([(i, 1), (a, 0)], sgn2)
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

#function gen_t2_kucj(aa, ab)
#    generator = FermionOperator([(aa, 1), (ab, 1), (ab, 0), (aa, 0)], 1.0im)
#    jordan_wigner(generator)
#end

"""
Generate pair dobule excitations
"""

```
Two particle excitation ops for unitary cluster jastrow

c^i ci c^j cj
```

function gen_t2_kucj_2(aa, ab)
    generator = FermionOperator([(aa, 1), (aa, 0), (ab, 1), (ab, 0)], 1.0im)
    return jordan_wigner(generator)
end


"""
Returns UCCGSD circuit.
"""
function uccgsd(
    n_qubit;
    nocc=-1,
    orbital_rot=true,
    conserv_Sz_doubles=true,
    conserv_Sz_singles=true,
    Doubles=true,
    uccgsd=true,
    p_uccgsd=false,
    nx=0
)
    if n_qubit <= 0 || n_qubit % 2 != 0
        error("Invalid n_qubit: $(n_qubit)")
    end
    if !orbital_rot && nocc < 0
        error("nocc must be given when orbital_rot = false!")
    end
    circuit = UCCQuantumCircuit(n_qubit)
    
    for i in 1:nx
        add_X_gate!(circuit, i)
    end

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
            add_parametric_circuit_using_generator!(circuit, generator, 0.0)
            #a = (a_spin_orbital, i_spin_orbital) => length(circuit.thetas) # Pair という型があるよ
            # Dict("a"=>1)
            #a = Dict([("a_spin_orbital+i_spin_orbital", length(circuit.thetas))])
            #push!(parameterinfo, a)
        end
    end
    println("num_thetas_orbrot=", num_theta(circuit))
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

function kucj(n_qubit; 
    conserv_Sz_singles=true, 
    k=1, 
    ucj=true, 
    orbrot=true, 
    sparse=false, 
    oneimp=true, 
    twoimp=false,
    onsite_bathsite = false,
    nx=0)
    if n_qubit <= 0 || n_qubit % 2 != 0
        error("Invalid n_qubit: $(n_qubit)")
    end

    circuit = UCCQuantumCircuit(n_qubit)
    for i in 1:nx
        add_X_gate!(circuit, i)
    end
    
    norb = n_qubit ÷ 2
    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)
    parameterinfo = []
    sz = [1, -1]

    if orbrot
        for (a_spatial, i_spatial) in (Iterators.product(1:norb, 1:norb))
            for ispin1 = 1:2, ispin2 = 1:2
                if conserv_Sz_singles && sz[ispin1] + sz[ispin2] != 0
                    continue
                end
                #Spatial Orbital Indices
                a_spin_orbital = so_idx(a_spatial, ispin1)
                i_spin_orbital = so_idx(i_spatial, ispin2)
                #t1 operator
                generator = gen_t1(a_spin_orbital, i_spin_orbital)
                add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                #orbital_rotation=-1
                orbital_rotation = (-1, num_theta(circuit)) # ここは全て独立なパラメータ
                push!(parameterinfo, orbital_rotation)
            end
        end
    end
    println("num_thetas(orbrot)=", num_theta(circuit))
    if ucj
        for i = 1:k
            # exp(-K) where K is an orbital rotation operator
            # K = \sum_{pq, \sigma} K_{pq} a_{p \sigma}^{\dagger}a_{q \sigma}
            # diagonal part of K, p=q
            # K_{p p} is a pure imaginary number
            # K_{p p} a_{p up}^{\dagger} a_{p up} a_{p down}^{\dagger} a_{p down} 
            for a_spatial in 1:norb
                for ispin1 = 1:2
                    a_spin_orbital = so_idx(a_spatial, ispin1)
                    if a_spin_orbital % 2 == 0
                        continue
                    end
                    # a_spin_orbital = 1, 3, 5,,,
                    generator = gen_t1_diag_kucj(a_spin_orbital, sgn=-1.0im) # exp(-K)
                    generator = rm_identity(generator)
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                    paraminfo_t1_diag = (a_spin_orbital, i)
                    push!(parameterinfo, paraminfo_t1_diag)
                end
            end
            # non-diagonal part of t1
            # K = \sum_{p/=q, \sigma} K_{pq} a_{p \sigma}^{\dagger}a_{q \sigma}
            # real part, K_{p q} is a real parameter
            # K_{p q} a_{p \sigma}^{\dagger}a_{q \sigma} + a_{q \sigma}^{\dagger}a_{p \sigma}
            for (a_spatial, i_spatial) in (Iterators.product(1:norb, 1:norb))
                for ispin1 = 1:2
                    a_spin_orbital = so_idx(a_spatial, ispin1) #if ispin1 = 1, then a_spin_orbital=1, 3, 5,, #if ispin1 = 2, then a_spin_orbital=2, 4, 6,,
                    i_spin_orbital = so_idx(i_spatial, ispin1) #
                    #perform only loof if a_spin_orbital < i_spin_orbital
                    if a_spin_orbital >= i_spin_orbital
                        continue
                    end
                    # Ex) ispin=1 => (a_spin_orbital =1, i_spin_orbital=3) or  (a_spin_orbital =1, i_spin_orbital=5) 
                    # Ex) ispin=2 => (a_spin_orbital =2, i_spin_orbital=4) or  (a_spin_orbital =2, i_spin_orbital=6) 
                    generator = gen_t1_kucj_real(a_spin_orbital, i_spin_orbital, sgn1=-1.0, sgn2=1.0)
                    #Add t1 into the circuit
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0) # 回転角度をマイナスをしたい！generatorにマイナス符号をつける
                    #a = (a_spin_orbital, i_spin_orbital) => length(circuit.thetas) # Pair という型がある
                    #paraminfo_t1_nondiag_real_up = ((a_spatial, ispatial, real=-2, k), num_theta(circuit)) 
                    # non-diag real
                    paraminfo_t1_nondiag_real = (a_spatial, i_spatial, -2, i)
                    push!(parameterinfo, paraminfo_t1_nondiag_real)
                end

                # imaginary part
                # K_{p q} is a pure imaginary parameter 
                # K_{p q} a_{p \sigma}^{\dagger}a_{q \sigma} + a_{q \sigma}^{\dagger}a_{p \sigma}
                for ispin1 = 1:2
                    a_spin_orbital = so_idx(a_spatial, ispin1)
                    i_spin_orbital = so_idx(i_spatial, ispin1)
                    if a_spin_orbital >= i_spin_orbital
                        continue
                    end
                    generator = gen_t1_kucj_imag(a_spin_orbital, i_spin_orbital, sgn1=-1.0im, sgn2=1.0im)
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                    # paraminfo_t1_nondiag_imag = ((a_spatial, i_spatial, imag=-3, i), num_theta(circuit))
                    paraminfo_t1_nondiag_imag = (a_spatial, i_spatial, -3, i)
                    push!(parameterinfo, paraminfo_t1_nondiag_imag)
                end
            end

            #Doubles
            #diagonal with respect to spatial orbitals a, b
            
            for a in 1:norb
                #if sparse 
                    #if onsite_bathsite
                        # オンサイトのバスサイトを削る。
                        #if oneimp
                            # 一つのバスサイトに関するパラメータを削る.1は不純物
                        #    if a >= 2
                        #        continue
                        #    end
                        #end

                        #if twoimp
                            # 一つのバスサイトに関するパラメータを削る。1,2は不純物
                        #    if a >=3
                        #        continue
                        #    end
                        #end
                    #end
                #end
                        
                aa = so_idx(a, 1) # spin_a = up
                bb = so_idx(a, 2) # spin_b = down
                # それ以外は、0（スピンに関して対角) or 同じ値(spin_a=2, spin_b = 1)
                generator = gen_t2_kucj_2(aa, bb)
                generator = rm_identity(generator)
                add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                paraminfo_t2_diag = (a, a, 1, 2, i)
                push!(parameterinfo, paraminfo_t2_diag)

                aa = so_idx(a, 2) # spin_a = down
                bb = so_idx(a, 1) # spin_b = up
                # それ以外は、0（スピンに関して対角) or 同じ値(spin_a=2, spin_b = 1)
                generator = gen_t2_kucj_2(aa, bb)
                generator = rm_identity(generator)
                add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                paraminfo_t2_diag = (a, a, 1, 2, i)
                push!(parameterinfo, paraminfo_t2_diag)

                #aa = so_idx(a, 1) # spin_a = up
                #bb = so_idx(a, 1) # spin_b = up
                # それ以外は、0（スピンに関して対角) or 同じ値(spin_a=2, spin_b = 1)
                #generator = gen_t2_kucj_2(aa, bb)
                #generator = rm_identity(generator)
                #add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                #paraminfo_t2_diag = (a, a, 1, 1, k)
                #push!(parameterinfo, paraminfo_t2_diag)

                #aa = so_idx(a, 2) # spin_a = down
                #bb = so_idx(a, 2) # spin_b = down
                # それ以外は、0（スピンに関して対角) or 同じ値(spin_a=2, spin_b = 1)
                #generator = gen_t2_kucj_2(aa, bb)
                #generator = rm_identity(generator)
                #add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                #paraminfo_t2_diag = (a, a, 1, 1, k)
                #push!(parameterinfo, paraminfo_t2_diag)
            end
            # non-diagonal with respect to spatial orbitals a, b
            # diagonal with respect to spin_orbitals
            for a in 1:norb
                if sparse
                    if oneimp
                        a = 1 #One impurity spatial orbitals (special case)
                    end
                    if twoimp
                        if a >= 3 #a=1,2
                            continue
                        end
                    end
                end
                for b in 1:norb
                    # a<bのみループが回るようにしたい
                    if a >= b
                        continue
                    end
                    aa = so_idx(a, 1)
                    bb = so_idx(b, 1)
                    generator = gen_t2_kucj_2(aa, bb)
                    generator = rm_identity(generator)
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                    paraminfo_t2_nondiag_spin_diag1 = (a, b, 1, 1, i)
                    push!(parameterinfo, paraminfo_t2_nondiag_spin_diag1)

                    # a, b , spin_a=2, spin_b=2
                    aa = so_idx(a, 2)
                    bb = so_idx(b, 2)
                    generator = gen_t2_kucj_2(aa, bb)
                    generator = rm_identity(generator)
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                    paraminfo_t2_nondiag_spin_diag2 = (a, b, 1, 1, i)
                    push!(parameterinfo, paraminfo_t2_nondiag_spin_diag2)
                    # non-diagonal with respect to spin_orbitals
                    # a, b , spin_a=1, spin_b=2
                    aa = so_idx(a, 1)
                    bb = so_idx(b, 2)
                    generator = gen_t2_kucj_2(aa, bb)
                    generator = rm_identity(generator)
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                    paraminfo_t2_nondiag_spin_diag2 = (a, b, 1, 2, i)
                    push!(parameterinfo, paraminfo_t2_nondiag_spin_diag2)
                    # a, b , spin_a=2, spin_b=1
                    aa = so_idx(a, 2)
                    bb = so_idx(b, 2)
                    generator = gen_t2_kucj_2(aa, bb)
                    generator = rm_identity(generator)
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                    paraminfo_t2_nondiag_spin_diag2 = (a, b, 1, 2, i)
                    push!(parameterinfo, paraminfo_t2_nondiag_spin_diag2)
                end
            end

            # exp(K) where K is an orbital rotation operator
            # diagonal part of K
            for a_spatial in (1:norb)
                for ispin1 = 1:2
                    a_spin_orbital = so_idx(a_spatial, ispin1)
                    if a_spin_orbital % 2 == 0
                        continue
                    end
                    generator = gen_t1_diag_kucj(a_spin_orbital)
                    generator = rm_identity(generator)
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                    paraminfo_t1_diag = (a_spin_orbital, i)
                    push!(parameterinfo, paraminfo_t1_diag)
                end
            end
            # non-diagonal part of K
            for (a_spatial, i_spatial) in (Iterators.product(1:norb, 1:norb))
                # real 
                for ispin1 = 1:2
                    a_spin_orbital = so_idx(a_spatial, ispin1) #if ispin1 = 1, then 1, 3, 5,,
                    i_spin_orbital = so_idx(i_spatial, ispin1) #if ispin1 = 2, then 2, 4, 6,,
                    #perform only loof if a_spin_orbital < i_spin_orbital
                    if a_spin_orbital >= i_spin_orbital
                        continue
                    end
                    generator = gen_t1_kucj_real(a_spin_orbital, i_spin_orbital)
                    #Add t1 into the circuit
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                    #a = (a_spin_orbital, i_spin_orbital) => length(circuit.thetas) # Pair という型がある
                    #paraminfo_t1_nondiag_real_up = ((a_spatial, ispatial, real=-2, k), num_theta(circuit)) 
                    paraminfo_t1_nondiag_real = (a_spatial, i_spatial, -2, i)
                    push!(parameterinfo, paraminfo_t1_nondiag_real)
                end

                # imaginary
                for ispin1 = 1:2
                    a_spin_orbital = so_idx(a_spatial, ispin1)
                    i_spin_orbital = so_idx(i_spatial, ispin1)
                    if a_spin_orbital >= i_spin_orbital
                        continue
                    end
                    generator = gen_t1_kucj_imag(a_spin_orbital, i_spin_orbital)
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                    # paraminfo_t1_nondiag_imag = ((a_spatial, i_spatial, imag=-3, i), num_theta(circuit))
                    paraminfo_t1_nondiag_imag = (a_spatial, i_spatial, -3, i)
                    push!(parameterinfo, paraminfo_t1_nondiag_imag)
                end
            end
        end
    end
    println("num_thetas(redundant)=", num_theta(circuit))
    return circuit, parameterinfo
end


"""
Returns sparse circuit based on impurity model.
"""
function sparse_ansatz(
    n_qubit;
    nocc=-1,
    orbital_rot=true,
    conserv_Sz_doubles=true,
    conserv_Sz_singles=true,
    oneimp = true,
    twoimp = false,
    twoimp_bb = false,
    nx = 0
)
    if n_qubit <= 0 || n_qubit % 2 != 0
        error("Invalid n_qubit: $(n_qubit)")
    end
    if !orbital_rot && nocc < 0
        error("nocc must be given when orbital_rot = false!")
    end
    circuit = UCCQuantumCircuit(n_qubit)
    for i in 1:nx
        add_X_gate!(circuit, i)
    end

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

            #A = [
            #    
            #]
            #if in(1, A) == false && in(2, A) == false
            #    continue
            #end
            if oneimp 
                if count(i -> (1 <= i <= 2), A) <= 1 #in(1, A) == false && in(2, A) == false
                    #println("Aの中に2つ以上のimpurityのスピン軌道が含まれる（４サイトの場合）.")
                    continue
                end
            end
            # Impurity = 2
            #    ◯(5,6)            ◯(11,12)
            #     \               / 
            # ◯(7,8) -- ◯(1,2) -- ◯(3,4) -- ◯(13,14)
            #     /               \ 
            #    ◯(9,10)            ◯(15,16)
            #
            if twoimp
                if count(i -> (1 <= i <= 4), A) <= 1 #in(1, A) == false && in(2, A) == false
                    #println("Aの中に2つ以上のimpurityのスピン軌道が含まれる（４サイトの場合）.")
                    continue
                elses
                    maxcnt = 0
                    for i in 1:4
                        maxcnt += length(findall(a -> a == i, A))
                    end
                    #@show maxcnt
                    @assert maxcnt >= 2 # assertエラーが起きたら、上の実装がおかしい。maxintが１だったらimpurityが一つなのでなんか変？
                end
            end

            if twoimp_bb 
                if count(i -> (5 <= i <= 10), A) == 1 && count(i -> (11 <= i <= 16), A) == 1
                    continue
                end
            end
            #if twoimp
            #    if count(i -> (5 <= i <= 16), A) => 4 #bathの数が４つあったら捨てる。
            #        continue
            #    end
                # if 2 & (3,4,5) の間で粒子の交換があったらサボる？
                # if でも今のでうまくいかなかったら、UCCGSDでもうまくいくのかわからない。。
            #end

            #t2 operator
            generator = gen_p_t2(aa, ia, bb, jb)
            #Add p-t2 into the circuit
            add_parametric_circuit_using_generator!(circuit, generator, 0.0)
        end
    end
    println("num_thetas=", num_theta(circuit))
    circuit
end


function orb_rot(
    n_qubit;
    conserv_Sz_singles=true
)
    if n_qubit <= 0 || n_qubit % 2 != 0
        error("Invalid n_qubit: $(n_qubit)")
    end

    norb = n_qubit ÷ 2
    circuit = UCCQuantumCircuit(n_qubit)
    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)
    sz = [1, -1]
    # Singles
    for (a_spatial, i_spatial) in (Iterators.product(1:norb, 1:norb))
        for ispin1 = 1:2, ispin2 = 1:2
            if conserv_Sz_singles && sz[ispin1] + sz[ispin2] != 0
                continue
            end
            #Spatial Orbital Indices
            a_spin_orbital = so_idx(a_spatial, ispin1)
            i_spin_orbital = so_idx(i_spatial, ispin2)
            #t1 operator
            generator = gen_t1(a_spin_orbital, i_spin_orbital)
            add_parametric_circuit_using_generator!(circuit, generator, 0.0)
        end
    end
    println("num_thetas=", num_theta(circuit))
    circuit
end

