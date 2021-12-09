using Test
using QCMaterial

@testset "hamiltonian.generate_ham_1d_hubbard" begin
    nsite = 2 
    n_qubit = 2*nsite 
    U = 1.0
    t = -0.01
    μ = 0.0

    ham = generate_ham_1d_hubbard(t, U, nsite, μ)
    
    
    ham_ref = FermionOperator()
    up1 = up_index(1)
    down1 = down_index(1)
    up2 = up_index(2)
    down2 = down_index(2)

    #クーロン斥力
    ham_ref += FermionOperator("$(up1)^ $(down1)^ $(up1) $(down1)", -U) 
    ham_ref += FermionOperator("$(up2)^ $(down2)^ $(up2) $(down2)", -U) 
    
    #ホッピング積分
    ham_ref += FermionOperator("$(up2)^ $(up1)", t) 
    ham_ref += FermionOperator("$(up1)^ $(up2)", t) 
    ham_ref += FermionOperator("$(down2)^ $(down1)", t) 
    ham_ref += FermionOperator("$(down1)^ $(down2)", t) 

    #ケミカルポテンシャルの項
    ham_ref += FermionOperator("$(up1)^  $(up1) ", -μ) 
    ham_ref += FermionOperator("$(down1)^ $(down1)", -μ)
    ham_ref += FermionOperator("$(up2)^  $(up2) ", -μ) 
    ham_ref += FermionOperator("$(down2)^ $(down2)", -μ)

    @test ham == ham_ref
end

@testset "hamiltonian.generate_impurity_dimer" begin
    nsite = 2  
    U = 1.0
    V = 1.0
    μ = 1.0
    ε = 1.0

    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite)

    ham_ref = FermionOperator()
    up1 = up_index(1)
    down1 = down_index(1)
    up2 = up_index(2)
    down2 = down_index(2)

    #Coulomb
    ham_ref += FermionOperator("$(up1)^ $(down1)^ $(up1) $(down1)", -U) 
    
    #hybridization
    ham_ref += FermionOperator("$(up2)^ $(up1)", -V) 
    ham_ref += FermionOperator("$(up1)^ $(up2)", -V) 
    ham_ref += FermionOperator("$(down2)^ $(down1)", -V) 
    ham_ref += FermionOperator("$(down1)^ $(down2)", -V) 

    #chemical potential
    ham_ref += FermionOperator("$(up1)^  $(up1) ", -μ) 
    ham_ref += FermionOperator("$(down1)^ $(down1)", -μ)

    #bath energy level
    ham_ref += FermionOperator("$(up2)^  $(up2) ", ε) 
    ham_ref += FermionOperator("$(down2)^ $(down2)", ε)

    @test ham == ham_ref
end

@testset "hamiltonian.generate_impurity_ham_with_1imp_3bath" begin
    nsite = 4
    U = 1.0
    V = 1.0
    μ = 1.0
    ε = 1.0

    ham = generate_impurity_ham_with_1imp_multibath(U, V, μ, ε, nsite)

    ham_ref = FermionOperator()
    up1 = up_index(1)
    down1 = down_index(1)
    up2 = up_index(2)
    down2 = down_index(2)
    up3 = up_index(3)
    down3 = down_index(3)
    up4 = up_index(4)
    down4 = down_index(4)

    #Coulomb
    ham_ref += FermionOperator("$(up1)^ $(down1)^ $(up1) $(down1)", -U) 
    
    #hybridization
    ham_ref += FermionOperator("$(up2)^ $(up1)", -V) 
    ham_ref += FermionOperator("$(up1)^ $(up2)", -V) 
    ham_ref += FermionOperator("$(down2)^ $(down1)", -V) 
    ham_ref += FermionOperator("$(down1)^ $(down2)", -V) 

    ham_ref += FermionOperator("$(up3)^ $(up1)", -V) 
    ham_ref += FermionOperator("$(up1)^ $(up3)", -V) 
    ham_ref += FermionOperator("$(down3)^ $(down1)", -V) 
    ham_ref += FermionOperator("$(down1)^ $(down3)", -V) 

    ham_ref += FermionOperator("$(up4)^ $(up1)", -V) 
    ham_ref += FermionOperator("$(up1)^ $(up4)", -V) 
    ham_ref += FermionOperator("$(down4)^ $(down1)", -V) 
    ham_ref += FermionOperator("$(down1)^ $(down4)", -V) 

    #chemical potential
    ham_ref += FermionOperator("$(up1)^  $(up1) ", -μ) 
    ham_ref += FermionOperator("$(down1)^ $(down1)", -μ)

    #bath energy level
    ham_ref += FermionOperator("$(up2)^  $(up2) ", ε) 
    ham_ref += FermionOperator("$(down2)^ $(down2)", ε)

    ham_ref += FermionOperator("$(up3)^  $(up3) ", ε) 
    ham_ref += FermionOperator("$(down3)^ $(down3)", ε)

    ham_ref += FermionOperator("$(up4)^  $(up4) ", ε) 
    ham_ref += FermionOperator("$(down4)^ $(down4)", ε)

    @test ham == ham_ref
end