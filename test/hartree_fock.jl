@testset "hartree_fock.extract_tij_Uijlk" begin
    ofermion = pyimport("openfermion")
    FermionOperator = ofermion.ops.operators.FermionOperator
    ham = FermionOperator("1^ 0^ 1 0") + FermionOperator("0^ 0") + FermionOperator("1^ 1", -1.0)
    tij, Uijlk = extract_tij_Uijlk(ham)

    tij_ref = [1.0 0.0; 0.0 -1.0]
    Uijlk_ref = zeros(Float64, 2, 2, 2, 2)
    Uijlk_ref[2, 1, 2, 1] = 2.0
    @test all(tij .== tij_ref)
    @test all(Uijlk .== Uijlk_ref)
end