import numpy as np
from mpi4py import MPI


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    fc_layer: str,
    in_dim: int,
    out_dim: int,
):
    """
    Prepare necessary information for later communications in forward and backward passes.

    Parameters
    ----------
    comm : Communicator
        The global MPI communicator.
    rank : int
        The global rank of the process.
    mp_size : int
        Model Parallel size.
    dp_size : int
        Data Parallel size.
    fc_layer : str
        Identifier for the fully-connected layer. It must be one of:
        'fc_q', 'fc_k', 'fc_v', or 'fc_o'.
        - For 'fc_q', 'fc_k', and 'fc_v', the partitioning is along the output dimension.
        - For 'fc_o', the partitioning is along the input dimension.
    in_dim : int
        Original input feature dimension.
    out_dim : int
        Original output feature dimension.

    Returns
    -------
    mp_idx : int
        Model parallel index (position within a data parallel replica).
    dp_idx : int
        Data parallel index (which replica this process belongs to).
    mp_comm : Communicator
        The model parallel communicator (all processes in one data parallel replica).
    dp_comm : Communicator
        The data parallel communicator (all processes holding the same weight shard).
    part_in_dim : int
        The partitioned input dimension for the FC layer.
    part_out_dim : int
        The partitioned output dimension for the FC layer.
    """
    # Compute the model parallel and data parallel indices
    mp_idx = rank % mp_size # mp id within dp group (position within a DP group)
    dp_idx = rank // mp_size # dp group id (which DP group this process belongs to)

    # Create communicators for MP and DP group (for intra-group communication)
    # mp_comm: All processes within the same DP group (same dp_idx) shared the same MP communicator
    # dp_comm: All processes with the same MP group (same mp_idx) shared the same DP communicator
    mp_comm = comm.Split(color=dp_idx, key=rank)  # All MP nodes in the same DP group
    dp_comm = comm.Split(color=mp_idx, key=rank)  # All DP nodes across MP groups

    # Determine the partitioned dimensions
    if fc_layer in ['fc_q', 'fc_k', 'fc_v']:
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size # shard along out_dim
    elif fc_layer == 'fc_o':
        part_in_dim = in_dim // mp_size # shard along in_dim
        part_out_dim = out_dim
    else:
        raise ValueError("Invalid fc_layer. Must be one of ['fc_q', 'fc_k', 'fc_v', 'fc_o'].")

    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim

def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward inputs from all model-parallel nodes.

    Each node holds a piece of the full input with shape:
      (batch_size, seq_length, part_in_dim)
    After gathering, the full input should have shape:
      (batch_size, seq_length, part_in_dim * mp_size)
    """
    batch_size, seq_length, part_in_dim = x.shape

    # Ensure input is contiguous
    x = np.ascontiguousarray(x)

    # Prepare buffer for gathering full input tensor
    gathered_x = None  # Non-root ranks should pass None in Gather()
    if mp_comm.rank == 0:
        gathered_x = np.empty((mp_size, batch_size, seq_length, part_in_dim), dtype=x.dtype)

    # Gather all parts at rank 0
    mp_comm.Gather(sendbuf=x, recvbuf=gathered_x, root=0)

    # Reconstruct the full tensor at rank 0
    if mp_comm.rank == 0:
        collected_x = np.concatenate(gathered_x, axis=-1)  # Merge slices along last axis
    else:
        collected_x = np.empty((batch_size, seq_length, part_in_dim * mp_size), dtype=x.dtype)

    # Broadcast full tensor to all MP ranks
    mp_comm.Bcast(collected_x, root=0)

    return collected_x

def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward outputs from all model-parallel nodes.

    Each node holds a piece of the full output with shape:
      (batch_size, seq_length, part_out_dim)
    After gathering, the full output should have shape:
      (batch_size, seq_length, part_out_dim * mp_size)
    """
    batch_size, seq_length, part_out_dim = out.shape

    # Ensure output is contiguous
    out = np.ascontiguousarray(out)

    # Prepare buffer for gathering full output tensor
    gathered_out = None
    if mp_comm.rank == 0:
        gathered_out = np.empty((mp_size, batch_size, seq_length, part_out_dim), dtype=out.dtype)

    # Gather all parts from MP ranks
    mp_comm.Gather(sendbuf=out, recvbuf=gathered_out, root=0)

    # Reconstruct the full tensor at rank 0
    if mp_comm.rank == 0:
        collected_out = np.concatenate(gathered_out, axis=-1)  # Merge slices along last axis
    else:
        collected_out = np.empty((batch_size, seq_length, part_out_dim * mp_size), dtype=out.dtype)

    # Broadcast the full output tensor to all MP ranks
    mp_comm.Bcast(collected_out, root=0)

    return collected_out

def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """
    Collect the fc output layer's output gradient for the local MP node.
    
    In our setup, the full output_grad is a 3-D tensor of shape 
        (batch_size, seq_length, out_dim),
    and the fully connected layer's weight is partitioned along out_dim.
    Therefore, we split output_grad along axis=2 into mp_size parts and
    return the part corresponding to mp_group_idx.
    
    Parameters
    ----------
    output_grad : np.ndarray
        The full output gradient from fc_o with shape 
        (batch_size, seq_length, out_dim).
    mp_group_idx : int
        The current model parallel node's index.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_output_grad : np.ndarray
        The local output gradient for this MP node with shape 
        (batch_size, seq_length, out_dim // mp_size).
    """
    batch_size, seq_length, out_dim = output_grad.shape

    # Compute the start and end indices for slicing
    part_size = out_dim // mp_size
    start_idx = mp_group_idx * part_size
    end_idx = start_idx + part_size

    # Slice the output gradient to get the local part
    collected_output_grad = output_grad[:, :, start_idx:end_idx]

    return collected_output_grad


def naive_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Use reduce-scatter / all-to-all to combine the contributions for grad_x from all nodes
    and scatter the reduced result along the input feature dimension.
    
    The grad_x tensor (gradient with respect to fc_o's input) has shape
        (batch_size, seq_length, in_dim),
    and the fc_o's weight matrix is sharded along the in_dim axis. In the 
    backward pass, each node computes a local grad_x and then these must be 
    summed across nodes. Instead of summing the full tensor and then slicing,
    we perform a reduce-scatter / all-to-all.
    
    Parameters
    ----------
    grad_x : np.ndarray
        The locally computed grad_x for fc_o, of shape 
        (batch_size, seq_length, in_dim).
    mp_comm :
        The model parallel communicator. It is assumed to expose methods such as reduce-scatter / all-to-all.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_grad_x : np.ndarray
        The reduced and scattered grad_x with shape 
        (batch_size, seq_length, in_dim // mp_size).
    """
    batch_size, seq_length, out_dim = grad_x.shape

    # Ensure in_dim is divisible by mp_size
    assert out_dim % mp_size == 0, "in_dim is not divisible by mp_size"

    # Compute the size of each shard
    part_size = out_dim // mp_size

    # Prepare buffer for Reduce-Scatter
    send_buf = np.empty((mp_size, batch_size, seq_length, part_size), dtype=grad_x.dtype)
    collected_grad_x = np.empty((batch_size, seq_length, part_size), dtype=grad_x.dtype)

    # Distribute grad_x among mp_size chunks before Reduce-Scatter
    for i in range(mp_size):
        send_buf[i] = grad_x[:, :, i * part_size : (i + 1) * part_size]

    # Perform Reduce-Scatter across model parallel ranks
    mp_comm.Reduce_scatter(sendbuf=send_buf, recvbuf=collected_grad_x)

    return collected_grad_x