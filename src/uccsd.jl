export num_theta, num_pauli, pauli_coeff, theta_offset, UCCQuantumCircuit, add_parametric_circuit_using_generator!
export update_circuit_param!, update_quantum_state!, gen_t1, gen_p_t2
export uccgsd

################################################################################
############################# QUANTUM  CIRCUIT #################################
################################################################################
struct UCCQuantumCircuit <: VariationalQuantumCircuit
    circuit::ParametricQuantumCircuit
    thetas::Vector{Float64}
    # Vector of (num_term_count::Int64, ioff::Int64, pauli_coeffs::Float64)
    theta_offsets::Vector{Tuple{Int64, Int64, Vector{Float64}}}
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

function add_parametric_circuit_using_generator!(circuit::UCCQuantumCircuit,
    generator::QubitOperator, theta::Float64) 
    pauli_coeffs = Float64[]
    for (pauli_str, pauli_coef) in terms_dict(generator)
        pauli_index_list, pauli_id_list = parse_pauli_str(pauli_str)
        if !all(1 .<= pauli_index_list)
            error("Pauli indies are out of range!")
        end
        if length(pauli_index_list) == 0
            continue
        end
        #println(pauli_str, pauli_coef)
        pauli_coef = imag(pauli_coef) #coef should be pure imaginary
        push!(pauli_coeffs, pauli_coef)
        add_parametric_multi_Pauli_rotation_gate!(
            circuit.circuit, pauli_index_list, pauli_id_list, pauli_coef*theta)
    end
    if length(pauli_coeffs) == 0
        return
    end
    num_thetas = num_theta(circuit)
    ioff = num_thetas == 0 ? 0 : theta_offset(circuit, num_thetas) + num_pauli(circuit, num_thetas)
    push!(circuit.thetas, theta)
    push!(circuit.theta_offsets, (get_term_count(generator), ioff, pauli_coeffs))
end



"""
Update circuit parameters

thetas wil be copied.
"""
function update_circuit_param!(circuit::UCCQuantumCircuit, thetas::Vector{Float64})
    if num_theta(circuit) != length(thetas)
        error("Invalid length of thetas!")
    end
    for (idx, theta) in enumerate(thetas)
        for ioff in 1:num_pauli(circuit, idx)
            pauli_coef = pauli_coeff(circuit, idx, ioff)
            set_parameter!(circuit.circuit,
                theta_offset(circuit, idx)+ioff, theta*pauli_coef) 
        end
    end
    circuit.thetas .= thetas
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
Generate pair dobule excitations
"""
function gen_p_t2(aa, ia, ab, ib)
    generator = FermionOperator([
        (aa, 1),
        (ab, 1),
        (ib, 0),
        (ia, 0)],
        1.0)
    generator += FermionOperator([
        (ia, 1),
        (ib, 1),
        (ab, 0),
        (aa, 0)],
        -1.0)
    jordan_wigner(generator)
end


"""
Returns UCCGSD circuit.
"""
function uccgsd(n_qubit; nocc=-1, orbital_rot=false, conserv_Sz_doubles=true, conserv_Sz_singles=true, Doubles=true, uccgsd = true, p_uccgsd=false)
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
        for ispin1 in 1:2, ispin2 in 1:2
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
                    if conserv_Sz_doubles && sz[spin_a] + sz[spin_i] + sz[spin_b] + sz[spin_j] != 0
                        continue
                    end
                    #Spatial Orbital Indices
                    aa = so_idx(a, spin_a)
                    ia = so_idx(i, spin_i)
                    bb = so_idx(b, spin_b)
                    jb = so_idx(j, spin_j)
                    #perform loop only if ia>jb && aa>bb               
                　
                    if aa<=bb || ia <= jb
                        continue
                    end
                         
                    #t2 operator
                    generator = gen_p_t2(aa, ia, bb, jb)
                    #Add p-t2 into the circuit
                    add_parametric_circuit_using_generator!(circuit, generator, 0.0)
                end
            end
            circuit
        end

        if p_uccgsd
            spin_a = 1
            spin_b = 2
            for (a,i) in Iterators.product(1:norb, 1:norb)
                aa = so_idx(a, spin_a)
                ia = so_idx(i, spin_a)
                bb = so_idx(a, spin_b)
                jb = so_idx(i, spin_b)  

                generator = gen_p_t2(aa, ia, bb, jb)
                #Add p-t2 into the circuit
                add_parametric_circuit_using_generator!(circuit, generator, 0.0)
            end
            circuit
        end
    end
end 
    



