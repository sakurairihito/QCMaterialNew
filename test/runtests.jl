using QCMaterial
using Test

@testset "util.topylist" begin
    org_array = [1, 2, 3.0]
    pylist = topylist(org_array)
    @test all(org_array .== pylist)
end
