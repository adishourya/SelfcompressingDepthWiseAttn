import torch
import cutlass
from functools import partial
import cutlass.cute as cute

@cute.kernel
def naive_elem_add_kernel(
    A:cute.Tensor,
    B:cute.Tensor,
    C:cute.Tensor
):
    ltid,_,_ = cute.arch.thread_idx()
    bid,_,_ = cute.arch.block_idx()
    bdim,_,_ = cute.arch.block_dim()

    thread_id = (bid*bdim) + ltid
    m,n = A.shape
    # the launched is a 1d set of threads.
    # we need to map these 1d threads to 2d indices
    # we will settle for row-major layout for now
    # assume atleast as many threads as elements
    col = thread_id % n
    row = thread_id // n

    # read
    a_val = A[row,col]
    b_val = B[row,col]

    # store
    C[row,col] = a_val + b_val


@cute.jit
def naive_elem_add(
    dA:cute.Tensor,
    dB:cute.Tensor,
    dC:cute.Tensor
):
    threads_per_block = 256
    m,n, = dA.shape
    # this does not run it.. just inits it
    kernel = naive_elem_add_kernel(dA,dB,dC)
    # this launches it
    kernel.launch(
    grid = ((m*n)// threads_per_block,1,1),
    block = (threads_per_block,1,1)
    )

def impl1():
    M,N = 2048,2048

    a = torch.ones(M,N,device="cuda",dtype=torch.float16)
    b = torch.ones(M,N,device="cuda",dtype=torch.float16)
    c = torch.zeros(M,N,device="cuda",dtype=torch.float16)

    # convert to cute tensors... using dl_pack
    a_ =cute.runtime.from_dlpack(a,assumed_align=16)
    b_ =cute.runtime.from_dlpack(b,assumed_align=16)
    c_ =cute.runtime.from_dlpack(c,assumed_align=16)

    # compile and run
    compiled = cute.compile(naive_elem_add, a_,b_,c_)
    compiled(a_,b_,c_)
    # print(c) # note its stored in c and not c_
    torch.testing.assert_close(c,a+b)
    benchmark(
              partial(compiled, a_,b_,c_),
              a_elems=M*N,
              num_warmups=5,
              num_iterations=100,
              )
def impl2(A:cute.Tensor,B:cute.Tensor,C:cute.Tensor):
    # Little's Law
    # number of items in system = Bandwidth * Latency
    # Bandwidth : Data transfer rate b/n memory and compute units
    # Latency : Round-trip delay of memory requests

    # to improve latency we need to improve number of items.
    # i.e get each thread to load more elements.. vectorized load and store
    # Ampere supports 128 bit per thread.. so for a fp32 a thread can handle 4
    # elements at a time

    # this vectorization can be made simpler with tiling operations
    # i.e we can partition the input tensor into groups of tiler blocks (1,4)
    # contiguous elements in the same row)
    # i.e suppose if you have 2048,2048: 2048 ,1 # row major 
    # then we divide this layout with a tiler (1,4) # 4 elems from same row
    # then you get ((1,4) , (2048,512)) : ((0,1) , (2048,4))
    # i.e size per thread , num of tiles
    pass

def benchmark(func,a_elems,* , num_warmups, num_iterations):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event= torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        func()

    # after warmup
    start_event.record(torch.cuda.current_stream())
    for _ in range(num_iterations):
        func()
    end_event.record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    elapsed = start_event.elapsed_time(end_event)
    avg_time = elapsed/num_iterations
    print(f"Avg Exec Time: {avg_time:.5f}ms")
    # read and write 3 matrices of size M,N 
    total_elem = 3*2*a_elems
    print(f"Throughput: {total_elem/ (avg_time/1000)/ 1e9:.5f} GB/s")

if __name__ == "__main__":
    impl1()
