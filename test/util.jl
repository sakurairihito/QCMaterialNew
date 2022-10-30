using Test
using QCMaterial

@testset "util.topylist" begin
    org_array = [1, 2, 3.0]
    pylist = topylist(org_array)
    @test all(org_array .== pylist)
end

@testset "util.doublefunc" begin
    org_array = [1, 2, 3]
    @test all(doublefunc(org_array) .== 2 .* org_array)
end

@testset "util.numerical_grad" begin
    # Some test for numerical_grad
    f(x) = x[1] + 2*x[2]
    deriv = numerical_grad(f, zeros(2))
    @test deriv ≈ [1.0, 2.0]
end


@testset "util.generate_numerical_grad" begin
    # Some test for numerical_grad
    N = 20
    f(x) = sum(collect(1:N) .* x)
    x = 1. .* collect(1:N)
    deriv = numerical_grad(f, x)
    grad = QCMaterial.generate_numerical_grad(f)
    @show grad
    @show grad(x)
    @test isapprox(grad(x), collect(1:N), rtol=1e-5)
end

@testset "util.fit_svd" begin
    #test for truncated svd
    #y = Ax
    #A = U∑V^{dagger}
    #x = V∑^{-1}(U^{dagger}y)
    N = 10
    A = randn(Float64, (N, N))
    y = ones(Float64, N)
    x = QCMaterial.fit_svd(y, A, 1e-10)
    @test y ≈ A * x
end

@testset "util.tikhonov" begin
    N = 10
    A = rand(Float64, (N,N))
    y = ones(Float64, N)
    x = QCMaterial.tikhonov(y, A, 1e-15)
    @test y ≈ A * x 
end

@testset "util.ParamInfo" begin
    ukeys = [(1,2), (1,3), (1,1,2), (1,1,2), (1,2,1), (1,2,1), (1,2), (1,3)]
    pinfo = ParamInfo(ukeys)
    @test pinfo.nparam == 4
    @test pinfo.nparamlong == 8
    @test pinfo.mapping == [1, 2, 3, 3, 4, 4, 1, 2]
end

@testset "expand" begin
    ukeys = [(1,2), (1,3), (1,1,2), (1,1,2), (1,2,1), (1,2,1), (1,2), (1,3)]
    pinfo = ParamInfo(ukeys)
    θunique = [-1.0, -2.0, -3.0, -4.0]
    theta_expand = expand(pinfo, θunique)
    @test theta_expand == [-1.0, -2.0, -3.0, -3.0, -4.0, -4.0, -1.0, -2.0]
end

#@testset "write_to_txt_1" begin
#    params = [1.0, 2.0]
#    write_to_txt_1(test, params) 
#    @test params == [1.0, 2.0]
#end

@testset "compact_paraminfo" begin
    keys = [(1,2), (1,3), (1,1,2), (1,1,2), (1,2,1), (1,2,1), (1,2), (1,3)]
    res = compact_paraminfo(keys) 
    @test res == Dict{Any, Any}((1, 2, 1) => 5, (1, 2) => 1, (1, 3) => 2, (1, 1, 2) => 3)
end


@testset "make_compact_params" begin
    thetas = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
    keys = [(1,2), (1,3), (1,1,2), (1,1,2), (1,2,1), (1,2,1), (1,2), (1,3)]
    res = make_compact_params(thetas, keys) 
    @test res == [-1.0, -2.0, -3.0, -5.0] 
end

@testset "make_long_param_form_compact" begin
    keys = [(1,2), (1,3), (1,1,2), (1,1,2), (1,2,1), (1,2,1), (1,2), (1,3)]
    thetas = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
    res = make_long_param_from_compact(keys,thetas)
    @test res == [-1.0, -2.0, -3.0, -3.0, -5.0, -5.0, -1.0, -2.0]
end