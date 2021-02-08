# Qulacs
import qulacs
from qulacs import Observable, GeneralQuantumOperator
from qulacs.observable import create_observable_from_openfermion_text
from qulacs import QuantumState, ParametricQuantumCircuit
# Openfermion
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermion.chem import MolecularData
from openfermion.ops import FermionOperator
from openfermion.utils import up_index, down_index
# Openfermion-PySCF
from openfermionpyscf import run_pyscf
# miscs.
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
#
from pyscf import fci

def parse_of_general_operators(num_qubits, openfermion_operators):
    """convert openfermion operator for generic cases (non-Hermitian operators)
    Args:
        n_qubit (:class:`int`)
        openfermion_op (:class:`openfermion.ops.QubitOperator`)
    Returns:
        :class:`qulacs.GeneralQuantumOperator`
    """
    ret = GeneralQuantumOperator(num_qubits)

    for pauli_product in openfermion_operators.terms:
        coef = openfermion_operators.terms[pauli_product]
        pauli_string = ''
        for pauli_operator in pauli_product:
            pauli_string += pauli_operator[1] + ' ' + str(pauli_operator[0])
            pauli_string += ' '
        ret.add_operator(coef, pauli_string[:-1])
    return ret


def qulacs_jordan_wigner(fermion_operator, n_qubits=None):
    """
    wrapper for openfermion.jordan_wigner which directly converts 
    openfermion.FermionOperator to qulacs.GeneralQuantumOperator
    Args:
        fermion_operator (openfermion.FermionOperator)
        n_qubits (int):
            # of qubits (if not given, n_qubits is assumed to be 
            the number of orbitals which appears in the given fermion operator)
    Return:
        qulacs.GeneralQuantumOperator
    """
    def count_qubit_in_qubit_operator(op):
        n_qubits = 0
        for pauli_product in op.terms:
            for pauli_operator in pauli_product:
                if n_qubits < pauli_operator[0]:
                    n_qubits = pauli_operator[0]
        return n_qubits+1

    qubit_operator = jordan_wigner(fermion_operator)
    _n_qubits = count_qubit_in_qubit_operator(qubit_operator) if n_qubits is None else n_qubits
    qulacs_operator = parse_of_general_operators(_n_qubits, qubit_operator)
    return qulacs_operator


def gen_t1(a, i):      
    # Generate single excitations
                #a^\dagger_a a_i (excitation)
                generator = FermionOperator((
                    (a, 1),
                    (i, 0)),
                    1.0)
                #-a^\dagger_i a_a (de-exciation)
                generator += FermionOperator((
                    (i, 1),
                    (a, 0)),
                    -1.0)
                #JW-transformation of a^\dagger_a a_i - -a^\dagger_i a_a
                qulacs_generator = qulacs_jordan_wigner(generator)
                return qulacs_generator
            
            
def gen_p_t2(aa, ia, ab, ib):      
    # Generate pair dobule excitations
                generator = FermionOperator((
                    (aa, 1),
                    (ab, 1),
                    (ib, 0),
                    (ia, 0)),
                    1.0)
                generator += FermionOperator((
                    (ia, 1),
                    (ib, 1),
                    (ab, 0),
                    (aa, 0)),
                    -1.0)
                qulacs_generator = qulacs_jordan_wigner(generator)
                return qulacs_generator


def add_parametric_multi_Pauli_rotation_gate(circuit, indices, 
                                             pauli_ids, theta, 
                                             parameter_ref_index=None, 
                                             parameter_coef=1.0):
    circuit.add_parametric_multi_Pauli_rotation_gate(indices, 
                                                     pauli_ids, theta)
    return circuit


def add_parametric_circuit_using_generator(circuit,
                                           generator, theta,
                                           param_index, coef=1.0):
    for i_term in range(generator.get_term_count()):
        pauli = generator.get_term(i_term)
        pauli_id_list = pauli.get_pauli_id_list()
        pauli_index_list = pauli.get_index_list()
        pauli_coef = pauli.get_coef().imag #coef should be pure imaginary
        circuit = add_parametric_multi_Pauli_rotation_gate(circuit, 
                        pauli_index_list, pauli_id_list,
                        theta, parameter_ref_index=param_index,
                        parameter_coef=coef*pauli_coef)
    return circuit


def add_theta_value_offset(theta_offsets, generator, ioff):
    pauli_coef_lists = []
    for i in range(generator.get_term_count()):
        pauli = generator.get_term(i)
        pauli_coef_lists.append(pauli.get_coef().imag) #coef should be pure imaginary
    if isinstance(theta_offsets, np.ndarray):
        np.append(theta_offsets, [generator.get_term_count(), ioff, 
                          pauli_coef_lists])
    else:
        theta_offsets.append([generator.get_term_count(), ioff, 
                          pauli_coef_lists])
    ioff = ioff + generator.get_term_count()
    return theta_offsets, ioff


"""
def update_circuit_param(circuit, theta_list, theta_offsets):
    for idx, theta in enumerate(theta_list):
        for ioff in range(theta_offsets[idx][0]):
            pauli_coef = theta_offsets[idx][2][ioff]
            circuit.set_parameter(theta_offsets[idx][1]+ioff, 
                                  theta*pauli_coef) #量子回路にパラメータをセット
            #print (theta_offsets[idx][1]+ioff)
    return circuit
"""

def UCCSD1(n_qubit, nocc, nvirt):
    """UCCSD1
    Returns UCCSD1 circuit.

    """
    theta_offsets = []
    circuit = ParametricQuantumCircuit(n_qubit)
    ioff = 0

    # Singles
    spin_index_functions = [up_index, down_index]
    for i_t1, (a, i) in enumerate(
        itertools.product(range(nvirt), range(nocc))):
        a_spatial = a + nocc
        i_spatial = i
        for ispin in range(2):
            #Spatial Orbital Indices
            so_index = spin_index_functions[ispin]
            a_spin_orbital = so_index(a_spatial)
            i_spin_orbital = so_index(i_spatial)
            #t1 operator
            qulacs_generator = gen_t1(a_spin_orbital,
                                      i_spin_orbital)
            #Add t1 into the circuit
            theta = 0.0
            theta_offsets, ioff = add_theta_value_offset(theta_offsets,
                                                      qulacs_generator,
                                                      ioff)
            circuit = add_parametric_circuit_using_generator(circuit,
                                                   qulacs_generator,
                                                   theta,i_t1,1.0)



    # Dobules
    for i_t2, (a, i, b, j) in enumerate(
            itertools.product(range(nvirt), range(nocc), range(nvirt), range(nocc))):

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
            qulacs_generator = gen_p_t2(aa, ia,
                                      bb, jb)
            #Add p-t2 into the circuit
            theta = 0.0
            theta_offsets, ioff = add_theta_value_offset(theta_offsets,
                                                      qulacs_generator,
                                                      ioff)
            circuit = add_parametric_circuit_using_generator(circuit,
                                                   qulacs_generator,
                                                   theta,i_t2,1.0)

    return circuit, theta_offsets

def convert_openfermion_op(n_qubit, openfermion_op):
    """convert_openfermion_op

    Args:
        n_qubit (:class:`int`)
        openfermion_op (:class:`openfermion.ops.QubitOperator`)
    Returns:
        :class:`qulacs.Observable`
    """
    ret = Observable(n_qubit)

    for pauli_product in openfermion_op.terms:
        coef = float(np.real(openfermion_op.terms[pauli_product]))
        pauli_string = ''
        for pauli_operator in pauli_product:
            pauli_string += pauli_operator[1] + ' ' + str(pauli_operator[0])
            pauli_string += ' '
        ret.add_operator(coef, pauli_string[:-1])

    return ret
