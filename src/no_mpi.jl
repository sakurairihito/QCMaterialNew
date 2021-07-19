export MPI_rank, MPI_size

MPI_rank = 0
MPI_size = 1
MPI_COMM_WORLD = nothing

mpirank(comm) = 0
mpisize(comm) = 1

function distribute(size, comm_size, rank)
    return 1, size
end

function Allreduce(data, mpi_sum, comm)
    return data
end
