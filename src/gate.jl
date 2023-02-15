abstract type Gate end
abstract type QuantumGateMatrix end
# 何をsub abstract typeにすれば良いのか？
# 

#include("quantum_state.jl")

struct QulacsGate <: Gate
    pyobj::PyObject
end
struct QulacsQuantumGateMatrix <: QuantumGateMatrix
    pyobj::PyObject
end

function RX(target::Int, angle::Real)
    QulacsGate(qulacs.gate.RX(target-1, angle))
end

function RY(target::Int, angle::Real)
    QulacsGate(qulacs.gate.RY(target-1, angle))
end

function RZ(target::Int, angle::Real)
    QulacsGate(qulacs.gate.RZ(target-1, angle))
end

# gateっていうクラスの中で定義されたメンバー関数の中からget_matrixを取り出している。
function get_matrix(gate::QulacsGate)
    gate.pyobj.get_matrix()
end

# qulacs.gateの中の名前空間からto_matrix_gateを呼び出している。
function to_matrix_gate(gate::QulacsGate)
    QulacsQuantumGateMatrix(qulacs.gate.to_matrix_gate(gate.pyobj) )
end

function H(target::Int)
    QulacsGate(qulacs.gate.H(target-1))
end

function Sdag(target::Int)
    QulacsGate(qulacs.gate.Sdag(target-1))
end

function S(target::Int)
    QulacsGate(qulacs.gate.S(target-1))
end

function X(target::Int)
    QulacsGate(qulacs.gate.X(target-1))
end

function Y(target::Int)
    QulacsGate(qulacs.gate.Y(target-1))
end

function Z(target::Int)
    QulacsGate(qulacs.gate.Z(target-1))
end

function Pauli(target_list::Vector{Int}, pauli_index::Vector{Int})
    QulacsGate(qulacs.gate.Pauli(target_list.-1, pauli_index))
end

#function update_quantum_state!(gate::QulacsGate, state::QulacsQuantumState)
#    gate.pyobj.update_quantum_state(state)
    #update_quantum_state(gate.pyobj, state)
#end


#function to_matrix_gate(gate::QulacsGate)
#    qulacs.gate.to_matrix_gate()
#end


#function H_update_quantum_state!(gate::QulacsGate, target::Int, state::QulacsQuantumState)
#    gate.pyobj.H(target).update_quantum_state(state)
#end

#abstract type 