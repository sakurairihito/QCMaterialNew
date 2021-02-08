# Qulacs
#import qulacs
from qulacs import Observable, GeneralQuantumOperator
#from qulacs.observable import create_observable_from_openfermion_text
#from qulacs import QuantumState, ParametricQuantumCircuit
# Openfermion
from openfermion.transforms import jordan_wigner
#from openfermion.linalg import get_sparse_operator
#from openfermion.chem import MolecularData
#from openfermion.ops import FermionOperator
#from openfermion.utils import up_index, down_index
# Openfermion-PySCF
#from openfermionpyscf import run_pyscf
import numpy as np
#
#from pyscf import fci

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
