export MPI_rank, MPI_size

MPI_rank = 0
MPI_size = 1
MPI_COMM_WORLD = nothing

function distribute(size, comm_size, rank)
    return 1, size
end

function Allreduce(data, mpi_sum, comm)
    return data
end
