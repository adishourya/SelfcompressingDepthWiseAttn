import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cltorch

# Enable console logging (default: False)
# export CUTE_DSL_LOG_TO_CONSOLE=1

# Log to file instead of console (default: False)
# export CUTE_DSL_LOG_TO_FILE=my_cute_log.txt

# Control log verbosity (0, 10, 20, 30, 40, 50, default: 10)
# export CUTE_DSL_LOG_LEVEL=20

@cute.jit
def static_dynamic(x_static:cutlass.Int32,
        x_dyn: cutlass.Constexpr):
    # constexpr needs to be known at compile time
    print(f"{x_dyn=}")
    print(f"{x_static=}")
    cute.printf(f"{x_dyn=}")
    cute.printf(f"{x_static=}")

static_dynamic(x_static=2, x_dyn=2)

@cute.jit
def known_layout(x:cute.Tensor):
    print(f"tensor.layout: {x.layout}")  # Prints tensor layout at compile time
    cute.printf("tensor: {}", x)         # Prints tensor values at runtime

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
a_pack = cute.runtime.from_dlpack(a)
print(f"{a_pack=}") # this does not have the a.data... just the tensor layout, mem_pointer , element_type

compiled_func = cute.compile(known_layout, a_pack)
compiled_func(a_pack)


# first kernel
@cute.kernel
def say_hello_device():
    tid,_,_ = cute.arch.thread_idx()
    if tid == 0:
        cute.printf("hello from the device")

@cute.jit
def say_hello():
    # calls the device function
    #print: Can only show static values at compile time
    #cute.printf: Can display both static and dynamic values at runtime

    cute.printf("Even jitted function can use cute.printf")
    launch_config = dict(grid = (1,1,1), block = (32,1,1)) # grid and blockdim
    say_hello_device().launch(**launch_config)

# before launching any kernel we need to init context
cutlass.cuda.initialize_cuda_context()
# jit hello
say_hello()
# precompiled hello
compiled_hello = cute.compile(say_hello)
compiled_hello()

# making tensors and tensor layouts

@cute.jit
def create_tensor_ones(ptr: cute.Pointer):
    our_layout = cute.make_layout(shape = (8,5),
                                  stride = (5,1))
    generated_tensor = cute.make_tensor(iterator=ptr,
                                        layout= our_layout)
    generated_tensor.fill(1)
    cute.print_tensor(generated_tensor)
    return generated_tensor

a = torch.randn(8,5,dtype=cltorch.dtype(cutlass.Float32))
ptr_a = cute.runtime.make_ptr(cutlass.Float32, a.data_ptr())
print(ptr_a)
print("Generating ones tensor")
out = create_tensor_ones(ptr_a)
cute.print_tensor(out)

