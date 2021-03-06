import MPI

export distribute, MPI_rank, MPI_size

if !MPI.Initialized()
    MPI.Init()
end

MPI_rank = MPI.Comm_rank(MPI.COMM_WORLD)
MPI_size = MPI.Comm_size(MPI.COMM_WORLD)
MPI_COMM_WORLD = MPI.COMM_WORLD

mpirank(comm::MPI.Comm) = MPI.Comm_rank(comm)
mpisize(comm::MPI.Comm) = MPI.Comm_size(comm)

function distribute(size::Int, comm_size::Int, rank::Int)
    """
    Compute the first element and size for a given rank

    Parameters
    ----------
    size : Int
        Length of array
    comm_size : Int
        Number of MPI  processes
    rank: Int
        MPI rank

    Returns
    -------
    size : Int
        Size
    offset : Int
        Offsets for the MPI process
    """
    if comm_size > size
        if rank + 1 <= size
            return rank + 1, 1
        else
            return 1, 0
        end
    end

    base = div(size, comm_size)
    leftover = size % comm_size
    if rank+1 <= leftover
        size = base+1
        start = rank * (base+1) + 1
    else
        size = base
        start = (base+1) * leftover + (rank-leftover) * base + 1
    end
    return start, size
end

function Allreduce(data, mpi_sum, comm)
    if comm === nothing
        return data
    end
    return MPI.Allreduce(data, mpi_sum, comm)
end