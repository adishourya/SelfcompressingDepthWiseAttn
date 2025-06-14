from cutlass.cute.arch import thread_idx
import torch
import cutlass
import cutlass.cute as cute

@cute.kernel
def naive_elemadd(A:cute.Tensor,
                  B:cute.Tensor,
                  C:cute.Tensor):

    tx,_,_ = cute.arch.thread_idx()
    bidx,_,_ = cute.arch.thread_idx()
    bdx,_,_ = cute.arch.thread_idx()
    thread_id = (bidx * bdx) + tx

    m,n = A.shape
    i,j = thread_id // m , thread_id %n
    C[i,j] = A[i,j] + B[i,j]
    return C

