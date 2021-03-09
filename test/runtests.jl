using QCMaterial
using QCMaterial.HartreeFock
using Test
import PyCall: pyimport

@testset "util.topylist" begin
    org_array = [1, 2, 3.0]
    pylist = topylist(org_array)
    @test all(org_array .== pylist)
end

@testset "util.doublefunc" begin
    org_array = [1, 2, 3]
    @test all(doublefunc(org_array) .== 2 .* org_array)
end

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