using PyCall

numpy = pyimport("numpy")

struct Nya 
    pyobj::PyObject
end


#yaoobjを省きたい

#=
g = Nya(numpy) # qo1 たちは適当に与えられる. 
g.sakurai は getproperty(g, :sakurai) と同じ
g.pyobj.sakurai は (g.pyobj).sakurai

g.pyobj は getfield(g, :pyobj) と同じで qo1 という値が得られる.
(g.pyobj).sakurai は getproperty((g.pyobj), :sakurai) になる

# g.add_X_gate のことで g.pyobj.add_X_gate とみなしたい
# g.pyobj は g.pyobj.pyobj
=#



function Base.getproperty(g::Nya, s::Symbol) #構造体を引数に保つ関数！
    # if (s in fieldnames(Nya))
    if s == :pyobj
        getfield(g, s)
    else
        getproperty(getfield(g, :pyobj), s)
    end
end


nya = Nya(numpy)
# nya.sqrt(2) は getproperty(nya, :sqrt) と同じ. 内部で numpy.sqrt が得られる.
# sは、sqrt
# getfield(g, :pyobj)は、Nyaのなかのnumpy に相当する。

# getfield(g, :pyobj)はnumpyだと思う。
# 実際に、getfield(nya, :pyobj)で
# PyObject <module 'numpy' from '/usr/local/lib/python3.8/dist-packages/numpy/__init__.py'>

# getfield(nya, :pyobj) => numpy は、nya= Nya(numpy) の中のpyobjに対応する。
# PyObject <module 'numpy' from '/usr/local/lib/python3.8/dist-packages/numpy/__init__.py'> 


"numpy.arange(10)"

struct Goma
    pyobj::PyObject
    nakigoe::String
end

qulacs = pyimport("qulacs")
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

g = Goma(qulacs.QuantumCircuit(n_qubit), "aaaaaaaaaa") #Goma()でGomaというオブジェクトを作る。
g.add_Y_gate(1)

