export uccsd1, convert_openfermion_op, add_parametric_circuit_using_generator!, add_parametric_multi_Pauli_rotation_gate!

"""
Generate single excitations
"""
function gen_t1(a, i)
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
    qulacs_jordan_wigner(generator)
end

"""
Generate pair dobule excitations
"""
function gen_p_t2(aa, ia, ab, ib)
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
    qulacs_jordan_wigner(generator)
end


"""
Returns UCCGSD circuit.
"""
function uccgsd(n_qubit, nocc, nvirt, orbital_rot=false, conserv_Sz_doubles=true, conserv_Sz_singles=true)
    theta_offsets = []
    circuit = qulacs.ParametricQuantumCircuit(n_qubit)
    ioff = 0

    norb = nvirt + nocc
    cr_range = orbital_rot ? (1:norb) : (1+nocc:norb)
    anh_range = orbital_rot ? (1:norb) : (1:nocc)

    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)
    sz = [1, -1]
    
    # Singles
    spin_index_functions = [up_index, down_index]
    for (i_t1, (a_spatial, i_spatial)) in enumerate(Iterators.product(cr_range, anh_range))
	    for ispin1 in 1:2, ispin2 in 1:2
            if conserv_Sz_singles && sz[ispin1] + sz[ispin2] != 0
                continue
            end
            #Spatial Orbital Indices
            a_spin_orbital = so_idx(a_spatial, ispin1)
            i_spin_orbital = so_idx(i_spatial, ispin2)
            #t1 operator
            qulacs_generator = gen_t1(a_spin_orbital, i_spin_orbital)
            #Add t1 into the circuit
            theta = 0.0
            theta_offsets, ioff = add_theta_value_offset!(theta_offsets,
                                                      qulacs_generator,
                                                      ioff)
            add_parametric_circuit_using_generator!(circuit,
                                                   qulacs_generator,
                                                   theta)
        end
    end


<<<<<<< HEAD
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

            #t2 operator
=======
    # Dobules (different spins)
#    for (i_t2, (a, i, b, j)) in enumerate(Iterators.product(1:nvirt, 1:nocc, 1:nvirt, 1:nocc))
#            a_spatial = a + nocc
#            i_spatial = i
#            b_spatial = b + nocc
#            j_spatial = j
#
#            #Spatial Orbital Indices
#            aa = up_index(a_spatial)
#            ia = up_index(i_spatial)
#            bb = down_index(b_spatial)
#            jb = down_index(j_spatial)
##            #t1 operator
#            qulacs_generator = gen_p_t2(aa, ia, bb, jb)
#            #Add p-t2 into the circuit
#            theta = 0.0
#            theta_offsets, ioff = add_theta_value_offset!(theta_offsets,
#                                                   qulacs_generator,
#                                                   ioff)
#            add_parametric_circuit_using_generator!(circuit,
#                                                   qulacs_generator,
#                                                   theta)
#    end
    
        # Dobules (different spins)
    for (i_t2, (a, i, b, j)) in enumerate(Iterators.product(1:nvirt, 1:nocc, 1:nvirt, 1:nocc))
            a_spatial = a 
            i_spatial = i
            b_spatial = b + nocc
            j_spatial = j + nocc

            #Spatial Orbital Indices
            aa = down_index(a_spatial)
            ia = up_index(i_spatial)
            bb = up_index(b_spatial)
            jb = down_index(j_spatial)
            #t1 operator
>>>>>>> ba3651bbebd402248339b84f6113c1c769f64db2
            qulacs_generator = gen_p_t2(aa, ia, bb, jb)
            #Add p-t2 into the circuit
            theta = 0.0
            theta_offsets, ioff = add_theta_value_offset!(theta_offsets,
                                                   qulacs_generator,
                                                   ioff)
            add_parametric_circuit_using_generator!(circuit,
                                                   qulacs_generator,
                                                   theta)
        end
    end

    
    circuit, theta_offsets
end