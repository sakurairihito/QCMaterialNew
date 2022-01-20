using PyCall

qulacs = pyimport("qulacs")


struct Goma # QulacsQuantumCircuit
    pyobj::PyObject
    nakigoe
end

struct Nya 
    yaoobj::YaoObject
end

function Base.getproperty(g::Nya, s::Symbol) #構造体を引数に保つ関数！
    if (s in fieldnames(Goma))
        getfield(g, s)
    else
        s(g.yaoobj)
    end
end

n_qubit = 2
g = Goma(qulacs.QuantumCircuit(n_qubit), "kyu")
g.pyobj
g.add_X_gate
g.nakigoe

function Base.getproperty(g::Goma, s::Symbol) #構造体を引数に保つ関数！
    if (s in fieldnames(Goma))
        getfield(g, s)
    else
        getproperty(g.pyobj, s)
    end
end

println(g.add_X_gate(1))

c = qulacs.QuantumCircuit(n_qubit)
c.add_X_gate(1)

g = Goma(qulacs.QuantumCircuit(n_qubit)) #Goma()でGomaというオブジェクトを作る。
g.add_Y_gate(1)
