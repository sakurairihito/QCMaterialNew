using Test

using MPI

# Call MPI.Init() before loading QCMaterial.mpi
MPI.Init()

using QCMaterial.mpi

@testset "mpi.distribute" begin
    size = 11
    comm_size = 3

    #rank = 0 
    @test mpi.distribute(size, comm_size, 0) == (1, 4)

    #rank = 1
    @test mpi.distribute(size, comm_size, 1) == (5, 4)

    #rank = 2
    @test mpi.distribute(size, comm_size, 2) == (9, 3)
end
