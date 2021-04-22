import MPI

export distribute, MPI_rank, MPI_size

if !MPI.Initialized()
    MPI.Init()
end

MPI_rank = MPI.Comm_rank(MPI.COMM_WORLD)
MPI_size = MPI.Comm_size(MPI.COMM_WORLD)

function distribute(size, comm_size, rank)
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
        error("comm_size is larger than size!")
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
