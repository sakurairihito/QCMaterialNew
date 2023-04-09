export generate_ham_1d_hubbard
export generate_impurity_ham_with_1imp_multibath
export generate_impurity_ham_with_2imp_multibath
export generate_impurity_ham_with_1imp_3bath_dmft
export generate_impurity_ham_with_1imp_3bath_dmft_U8
function generate_ham_1d_hubbard(t::Float64, U::Float64, nsite::Integer, μ::Float64)
    #up_index,down_indexの定義は、QC_materialを参照。
    #up = up_index(i)
    #down = down_index(i)

    ham = FermionOperator()

    #斥力項
    for i in 1:nsite
        ham += FermionOperator("$(up_index(i))^ $(down_index(i))^ $(up_index(i)) $(down_index(i))", -U) #左側に生成演算子。右際に消滅演算子をもっていく過程で半交換関係が1回でマイナスをつける。
    end
    #ホッピング項
    for i in 1:nsite-1
        ham += FermionOperator("$(up_index(i+1))^ $(up_index(i))", t) 
        ham += FermionOperator("$(up_index(i))^ $(up_index(i+1))", t) 
        ham += FermionOperator("$(down_index(i+1))^ $(down_index(i))", t) 
        ham += FermionOperator("$(down_index(i))^ $(down_index(i+1))", t) 
    end

    #ケミカルポテンシャルの項
    for i in 1:nsite
        ham += FermionOperator("$(up_index(i))^  $(up_index(i)) ", -μ) 
        ham += FermionOperator("$(down_index(i))^ $(down_index(i))", -μ)
    end
    ham
end


function generate_impurity_ham_with_1imp_multibath(U::Float64, V::Float64, μ::Float64, ε::Vector{Float64}, nsite::Integer)
    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)
    ham = FermionOperator()

    #Coulomb   
    ham += FermionOperator("$(up_index(1))^ $(down_index(1))^ $(up_index(1)) $(down_index(1))", -U)

    for ispin in [1, 2]
        for i in 2:nsite
            ham += FermionOperator("$(so_idx(1, ispin))^ $(so_idx(i, ispin))", -V)
            ham += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(1, ispin))", -V)
        end
    end

    #chemical potential
    for ispin in [1, 2]
        ham += FermionOperator("$(so_idx(1, ispin))^ $(so_idx(1, ispin))", -μ)
    end

    for ispin in [1, 2]
        for i in 2:nsite
            ham += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(i, ispin))", ε[i])
        end
    end
    ham
end


function generate_impurity_ham_with_2imp_multibath(U::Float64, V::Float64, μ::Float64, ε::Vector{Float64}, t::Float64, nsite::Integer,numbath::Integer)
    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)
    ham = FermionOperator()
    #Coulomb   
    ham += FermionOperator("$(up_index(1))^ $(down_index(1))^ $(up_index(1)) $(down_index(1))", -U)
    ham += FermionOperator("$(up_index(2))^ $(down_index(2))^ $(up_index(2)) $(down_index(2))", -U)
    for ispin in [1, 2]
        for i in 3:2+numbath #3:5
            ham += FermionOperator("$(so_idx(1, ispin))^ $(so_idx(i, ispin))", -V)
            ham += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(1, ispin))", -V)
        end
    end
    for ispin in [1, 2]
        for i in 2+numbath+1:nsite #bath+1=6,nsite=8
            ham += FermionOperator("$(so_idx(2, ispin))^ $(so_idx(i, ispin))", -V)
            ham += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(2, ispin))", -V)
        end
    end

    #ham += J * QubitOperator("X1 X2") + J * QubitOperator("Z1 Z2") + J * QubitOperator("X1 X2")
    #ham += OFQubitOperator("Z1 Z2") + OFQubitOperator("X1 X2") + OFQubitOperator("Z1 Z2")
    #chemical potential
    for ispin in [1,2]
        ham += FermionOperator("$(so_idx(1, ispin))^ $(so_idx(2, ispin))", -t)
        ham += FermionOperator("$(so_idx(2, ispin))^ $(so_idx(1, ispin))", -t)
    end
    for i in [1, 2]
        for ispin in [1, 2]
            ham += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(i, ispin))", -μ)
        end
    end
    
    #bath energy
    #[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
    for ispin in [1, 2]
        for i in 3:nsite
            ham += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(i, ispin))", ε[i-2])
        end
    end
    ham
end


function generate_impurity_ham_with_1imp_3bath_dmft(U::Float64, μ::Float64,  nsite::Integer)
    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)

    ham_op1 = FermionOperator()
    ham_op1 += FermionOperator("$(up_index(1))^ $(down_index(1))^ $(up_index(1)) $(down_index(1))", -U)

    #chemical potential
    for ispin in [1, 2]
        ham_op1 += FermionOperator("$(so_idx(1, ispin))^ $(so_idx(1, ispin))", -μ)
    end

    #bath energy level
    ε = [0.0, 1.11919, 0.00000, -1.11919]
    for ispin in [1, 2]
        for i in 2:nsite
            ham_op1 += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(i, ispin))", ε[i])
        end
    end

    #hybridization
    V = [0.0, -1.26264, 0.07702, -1.26264]

    for ispin in [1, 2]
        for i in 2:nsite
            ham_op1 += FermionOperator("$(so_idx(1, ispin))^ $(so_idx(i, ispin))", V[i])
            ham_op1 += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(1, ispin))", V[i])
        end
    end
    ham_op1
end


function generate_impurity_ham_with_1imp_3bath_dmft_U8(U::Float64, μ::Float64,  nsite::Integer)
    # up_index = 2*(i-1) + 1
    # down_index = 2*i 
    # spin_index_functions = [up_index, down_index]
    spin_index_functions = [up_index, down_index]
    so_idx(iorb, ispin) = spin_index_functions[ispin](iorb)

    ham_op1 = FermionOperator()
    ham_op1 += FermionOperator("$(up_index(1))^ $(down_index(1))^ $(up_index(1)) $(down_index(1))", -U)

    #chemical potential
    for ispin in [1, 2]
        ham_op1 += FermionOperator("$(so_idx(1, ispin))^ $(so_idx(1, ispin))", -μ)
    end

    #bath energy level
    ε = [0.0, 1.17395, -0.20238, -3.04852]
    for ispin in [1, 2]
        for i in 2:nsite
            ham_op1 += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(i, ispin))", ε[i])
        end
    end

    #hybritization
    V = [0.0, 1.16862, -0.54186, 1.24341]

    for ispin in [1, 2]
        for i in 2:nsite
            if i == 3
                ham_op1 += FermionOperator("$(so_idx(1, 1))^ $(so_idx(i, 1))", -0.54186)
                ham_op1 += FermionOperator("$(so_idx(i, 1))^ $(so_idx(1, 1))", -0.54186)
                ham_op1 += FermionOperator("$(so_idx(1, 2))^ $(so_idx(i, 2))", 0.54186)
                ham_op1 += FermionOperator("$(so_idx(i, 2))^ $(so_idx(1, 2))", 0.54186)
            end
            if i != 3
                ham_op1 += FermionOperator("$(so_idx(1, ispin))^ $(so_idx(i, ispin))", V[i])
                ham_op1 += FermionOperator("$(so_idx(i, ispin))^ $(so_idx(1, ispin))", V[i])
        
            end
        end
    end
    ham_op1
end


