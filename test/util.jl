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