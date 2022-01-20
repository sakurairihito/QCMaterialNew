abstract type Gate end

struct QulacsGate <: Gate
    pyobj::PyObject
end

function RX(target::Int, angle::Float64)
    QulacsGate(qulacs.gate.RX(target, angle))
end

function RY(target::Int, angle::Float64)
    QulacsGate(qulacs.gate.RY(target, angle))
end

function RZ(target::Int, angle::Float64)
    QulacsGate(qulacs.gate.RZ(target, angle))
end

function get_matrix(gate::QulacsGate)
    gate.pyobj.get_matrix()
end