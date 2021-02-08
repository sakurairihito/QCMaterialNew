export uccsd1, convert_openfermion_op, convert_openfermion_op_debug

import PyCall: pyimport, PyVector

#pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__) # Add cwd to path

up_index(i) = 2*(i-1)
down_index(i) = 2*(i-1)+1

"""
Generate single excitations
"""
function gen_t1(a, i)
    ofermion = pyimport("openfermion")
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__) # Add cwd to path
    util = pyimport("util")
    #a^\dagger_a a_i (excitation)
    generator = ofermion.ops.FermionOperator((
                    (a, 1),
                    (i, 0)),
                    1.0)
    #-a^\dagger_i a_a (de-exciation)
    generator += ofermion.ops.FermionOperator((
                    (i, 1),
                    (a, 0)),
                    -1.0)
    #JW-transformation of a^\dagger_a a_i - -a^\dagger_i a_a
    util.qulacs_jordan_wigner(generator)
end

"""
Generate pair dobule excitations
"""
function gen_p_t2(aa, ia, ab, ib)
    ofermion = pyimport("openfermion")
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__) # Add cwd to path
    util = pyimport("util")
    generator = ofermion.ops.FermionOperator((
        (aa, 1),
        (ab, 1),
        (ib, 0),
        (ia, 0)),
        1.0)
    generator += ofermion.ops.FermionOperator((
        (ia, 1),
        (ib, 1),
        (ab, 0),
        (aa, 0)),
        -1.0)
    util.qulacs_jordan_wigner(generator)
end


function add_theta_value_offset!(theta_offsets, generator, ioff)
    pauli_coef_lists = Float64[]
    for i in 0:generator.get_term_count()-1
        pauli = generator.get_term(i)
        push!(pauli_coef_lists, imag(pauli.get_coef())) #coef should be pure imaginary
    end
    push!(theta_offsets, [generator.get_term_count(), ioff, pauli_coef_lists])
    ioff = ioff + generator.get_term_count()
    return theta_offsets, ioff
end


"""
Returns UCCSD1 circuit.
"""
function uccsd1(n_qubit, nocc, nvirt)
    qulacs = pyimport("qulacs")
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__) # Add cwd to path
    util = pyimport("util")

    theta_offsets = []
    circuit = qulacs.ParametricQuantumCircuit(n_qubit)
    ioff = 0

    # Singles
    spin_index_functions = [up_index, down_index]
    for (i_t1, (a, i)) in enumerate(Iterators.product(1:nvirt, 1:nocc))
        a_spatial = a + nocc
        i_spatial = i
	for ispin in 1:2
            #Spatial Orbital Indices
            so_index = spin_index_functions[ispin]
            a_spin_orbital = so_index(a_spatial)
            i_spin_orbital = so_index(i_spatial)
            #t1 operator
            qulacs_generator = gen_t1(a_spin_orbital, i_spin_orbital)
            #Add t1 into the circuit
            theta = 0.0
            theta_offsets, ioff = add_theta_value_offset!(theta_offsets,
                                                      qulacs_generator,
                                                      ioff)
            circuit = util.add_parametric_circuit_using_generator(circuit,
                                                   qulacs_generator,
                                                   theta,i_t1,1.0)
        end
    end


    # Dobules
    for (i_t2, (a, i, b, j)) in enumerate(Iterators.product(1:nvirt, 1:nocc, 1:nvirt, 1:nocc))
            a_spatial = a + nocc
            i_spatial = i
            b_spatial = b + nocc
            j_spatial = j

            #Spatial Orbital Indices
            aa = up_index(a_spatial)
            ia = up_index(i_spatial)
            bb = down_index(b_spatial)
            jb = down_index(j_spatial)
            #t1 operator
            qulacs_generator = gen_p_t2(aa, ia, bb, jb)
            #Add p-t2 into the circuit
            theta = 0.0
            theta_offsets, ioff = add_theta_value_offset!(theta_offsets,
                                                   qulacs_generator,
                                                   ioff)
            circuit = util.add_parametric_circuit_using_generator(circuit,
                                                   qulacs_generator,
                                                   theta,i_t2,1.0)
    end
    circuit, theta_offsets
end

"""convert_openfermion_op

Args:
    n_qubit (:class:`int`)
    openfermion_op (:class:`openfermion.ops.QubitOperator`)
Returns:
    :class:`qulacs.Observable`
"""
function convert_openfermion_op(n_qubit, openfermion_op)
    qulacs = pyimport("qulacs")
    ret = qulacs.Observable(n_qubit)
    for (pauli_product, coef) in openfermion_op.terms
        pauli_string = ""
        for pauli_operator in pauli_product
            pauli_string *= pauli_operator[2] * " $(pauli_operator[1]) "
        end
        ret.add_operator(real(coef), pauli_string[1:end-1])
    end
    ret
end
