export generate_ham_1d_hubbard
export generate_impurity_ham_with_1imp_multibath

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


function generate_impurity_ham_with_1imp_multibath(U::Float64, V::Float64, μ::Float64, ε::Float64, nsite::Integer)
    ham = FermionOperator()

    #Coulomb   
    ham += FermionOperator("$(up_index(1))^ $(down_index(1))^ $(up_index(1)) $(down_index(1))", -U)

    #hybridization
    for i in 1:nsite-1
        ham += FermionOperator("$(up_index(i+1))^ $(up_index(i))", -V)
        ham += FermionOperator("$(up_index(i))^ $(up_index(i+1))", -V)
        ham += FermionOperator("$(down_index(i+1))^ $(down_index(i))", -V)
        ham += FermionOperator("$(down_index(i))^ $(down_index(i+1))", -V)
    end

    #chemical potential
    ham += FermionOperator("$(up_index(1))^ $(up_index(1))", -μ)
    ham += FermionOperator("$(down_index(1))^ $(down_index(1))", -μ)

    #bath energy level
    for i in 2:nsite
        ham += FermionOperator("$(up_index(i))^ $(up_index(i))", ε)
        ham += FermionOperator("$(down_index(i))^ $(down_index(i))", ε)
    end

    ham
end