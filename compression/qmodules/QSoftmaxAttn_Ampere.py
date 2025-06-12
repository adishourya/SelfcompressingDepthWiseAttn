# Modified From : https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
# For Self Compressing Routing Attention

# From :  https://arxiv.org/pdf/2003.05997v5
# Attention-based models can be problematic for long sequences. For a sequence
# of length n, the full attention matrix A, as introduced in Section 3, is n
# × n-dimensional and can be prohibitive to instantiate. This motivates
# sparse attention models, i.e. models relying on attention matrices which have
# a majority of zero entries.

# Self routing Attn: https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_TRAR_Routing_the_Attention_Spans_in_Transformer_for_Visual_Question_ICCV_2021_paper.pdf
# SAR(X) = sum_i [ α[i] * softmax((X @ Wq).T @ (X @ Wk) / sqrt(d)) @ Di @ (X @ Wv) ]
# we will replace alpha with normalized bit depth.
# where we will tie a quantization function per head.


import torch
from cuda.bindings.driver import cuda
import cutlass
import cutlass.torch as cutorch
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp,cpasync
from cutlass.cute.runtime import from_dlpack
# remember H100.. has hopper architecutre and not ampere.
from cutlass.utils.ampere_helpers import SMEM_CAPACITY


print(torch.cuda.get_device_name(0))
ampere_capability = torch.cuda.get_device_capability(0)  # returns SM Version (major, minor)
ampere_capacity = f"sm{ampere_capability[0]}{ampere_capability[1]}" 
# print(SMEM_CAPACITY)
print(f"Ampere Capacity: {SMEM_CAPACITY[ampere_capacity]}")
