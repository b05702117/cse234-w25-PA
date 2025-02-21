from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size() # number of processes

    def Get_rank(self):
        return self.comm.Get_rank() # rank of the current process

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1) # each process sends its data to all other processes and receives data from all other processes
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        #TODO: Your code here
        nprocs = self.comm.Get_size() # number of processes
        rank = self.comm.Get_rank() # rank of the current process
        root = 0 # root process

        # calculate the number of bytes being transferred
        src_array_byte = src_array.itemsize * src_array.size

        # Reduce - All non-root processes send their data to the root process
        self.comm.Reduce(src_array, dest_array, op=op, root=root)

        # Broadcast - The root process sends the reduced data back to all processes
        self.comm.Bcast(dest_array, root=root)

        # calculate the number of bytes being transferred
        if rank == root:
            # Root receives from (size-1) other processes and sends to (size-1) other processes
            self.total_bytes_transferred += src_array_byte * 2 * (nprocs - 1)
        else:
            # Non-root processes send to root and receive from root
            self.total_bytes_transferred += src_array_byte * 2 

    def myAlltoall_naive(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()
        root = 0

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # calculate the number of bytes in one segment
        seg_size = src_array.size // nprocs
        send_seg_bytes = src_array.itemsize * seg_size
        recv_seg_bytes = dest_array.itemsize * seg_size

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        for peer in range(nprocs):
            send_offset = peer * seg_size
            recv_offset = peer * seg_size

            if peer == rank:
                # Directly copy the local segment (self-to-self transfer)
                dest_array[recv_offset:recv_offset + seg_size] = src_array[send_offset:send_offset + seg_size]
            else:
                # Exchange segments with peer using Sendrecv
                self.comm.Sendrecv(
                    sendbuf=src_array[send_offset:send_offset + seg_size],
                    dest=peer,
                    recvbuf=dest_array[recv_offset:recv_offset + seg_size],
                    source=peer
                )

                # Update the total bytes transferred
                self.total_bytes_transferred += send_seg_bytes + recv_seg_bytes
        

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()
        root = 0

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # calculate the number of bytes in one segment
        seg_size = src_array.size // nprocs
        send_seg_bytes = src_array.itemsize * seg_size
        recv_seg_bytes = dest_array.itemsize * seg_size

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        requests = []
        for peer in range(nprocs):
            send_offset = peer * seg_size
            recv_offset = peer * seg_size

            if peer == rank:
                # Directly copy the local segment (self-to-self transfer)
                dest_array[recv_offset:recv_offset + seg_size] = src_array[send_offset:send_offset + seg_size]
            else:
                # Non-blocking send
                send_req = self.comm.Isend(
                    src_array[send_offset:send_offset + seg_size],
                    dest=peer
                )
                requests.append(send_req)

                # Non-blocking receive
                recv_req = self.comm.Irecv(
                    dest_array[recv_offset:recv_offset + seg_size],
                    source=peer
                )
                requests.append(recv_req)

                # Update the total bytes transferred
                self.total_bytes_transferred += send_seg_bytes + recv_seg_bytes

        # Wait for all non-blocking operations to complete
        MPI.Request.Waitall(requests)

    def myAlltoall_ring(self, src_array, dest_array):
        """
        Optimized all-to-all using a ring-based approach to minimize network congestion.
        
        - Reduces contention compared to full P-to-P message passing.
        - Each process exchanges data only with its neighbor at each step.
        """
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()

        seg_size = src_array.size // nprocs
        send_seg_bytes = src_array.itemsize * seg_size

        # Initial copy (self-to-self)
        dest_array[rank * seg_size:(rank + 1) * seg_size] = src_array[rank * seg_size:(rank + 1) * seg_size]

        for step in range(1, nprocs):
            send_to = (rank + step) % nprocs
            recv_from = (rank - step) % nprocs

            self.comm.Sendrecv(
                sendbuf=src_array[send_to * seg_size:(send_to + 1) * seg_size], dest=send_to,
                recvbuf=dest_array[recv_from * seg_size:(recv_from + 1) * seg_size], source=recv_from
            )

            # Update total bytes transferred
            self.total_bytes_transferred += send_seg_bytes * 2