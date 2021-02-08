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
