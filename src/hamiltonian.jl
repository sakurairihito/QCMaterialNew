export generate_ham_1d_hubbard

function generate_ham_1d_hubbard(t::Float64, U::Float64, nsite::Integer, μ::Float64)
    ham = FermionOperator()
    
    #斥力項
    for i in 1:nsite
        #up_index,down_indexの定義は、QC_materialを参照。
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^ $(down)^ $(up) $(down)", -U) #左側に生成演算子。右際に消滅演算子をもっていく過程で半交換関係が1回でマイナスをつける。
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
        up = up_index(i)
        down = down_index(i)
        ham += FermionOperator("$(up)^  $(up) ", -μ) 
        ham += FermionOperator("$(down)^ $(down)", -μ)
    end
    ham
end