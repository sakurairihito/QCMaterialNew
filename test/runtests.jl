using QCMaterial
using Test

@testset "util.topylist" begin
    org_array = [1, 2, 3.0]
    pylist = topylist(org_array)
    @test all(org_array .== pylist)
end

@testset "util.doublefunc" begin
    org_array = [1, 2, 3]
    @test all(doublefunc(org_array) .== 2 .* org_array)
end


@testset "uccsd.uccgsd" begin
    # Construct 2-site Hubbard model
    # Solve VQE
    # Compare the results with the exact one
end
