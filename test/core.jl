using Test
using QCMaterial

using PyCall
of = pyimport("openfermion")

@testset "core.jordan_wigner" begin
    #jw = of.transforms.jordan_wigner(
        #of.ops.FermionOperator(((0, 1), (0, 0)))
    #)
    #her, antiher = divide_real_imag_openfermion(jw)
    #println(jw, jw.__class__, her.__class__)
#
    #jw2 = jordan_wigner(FermionOperator([(0, 1), (0, 0)]))
    #println(jw2, jw2.pyobj.__class__)
end