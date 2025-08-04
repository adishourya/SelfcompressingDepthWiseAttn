from cutlass.cute import kernel
import torch
import torchvision.io as io
import torchvision.transforms as transforms
from functools import partial

def benchmark(func, input_tensor, output_tensor, weight_tensor, *, num_warmups, num_iterations):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    for _ in range(num_warmups):
        func()

    torch.cuda.synchronize()
    start_event.record(torch.cuda.current_stream())
    for _ in range(num_iterations):
        func()
    end_event.record(torch.cuda.current_stream())
    torch.cuda.synchronize()

    elapsed = start_event.elapsed_time(end_event)  # milliseconds
    avg_time = elapsed / num_iterations

    print(f"Avg Exec Time: {avg_time:.5f} ms")

    total_bytes = (
        input_tensor.numel() + output_tensor.numel() + weight_tensor.numel()
    ) * 4  # float32 → 4 bytes
    throughput_gbs = total_bytes / (avg_time / 1000) / 1e9

    print(f"Throughput: {throughput_gbs:.5f} GB/s")


# --- Config ---
B = 256# batch size
H=W=64
dummy_dict = dict(
in_channels = 3,
out_channels = 32,
kernel_size = 3,
padding = 1,
groups = 1
)
print(f"Resolution = {H}x{W}")

ungrouped_dict = dict(
    in_channels = dummy_dict["out_channels"],
    out_channels = dummy_dict["out_channels"],
    kernel_size = dummy_dict["kernel_size"],
    stride=2,
    padding=1,
    groups = 1
)

depth_wise_dict = dict(
    in_channels = dummy_dict["out_channels"],
    out_channels = dummy_dict["out_channels"],
    kernel_size = dummy_dict["kernel_size"],
    stride=2,
    padding = 1,
    groups = dummy_dict["out_channels"]
)
# --------------

# Load and preprocess image
img = io.read_image("./dolphin.jpg")  # (3, H, W), uint8
resizer = transforms.Resize((H, W))
input_imgs = resizer(img).float() / 255.0  # (3, H, W), float32
input_imgs = input_imgs.unsqueeze(0).repeat(B, 1, 1, 1).to("cuda")  # (B, 3, 224, 224)

# Conv layer
dummy_conv = torch.nn.Conv2d(**dummy_dict).to("cuda")
dummy_conv = torch.compile(dummy_conv)

# normal_conv
ungrouped_convT = torch.nn.ConvTranspose2d(**ungrouped_dict).to("cuda")
ungrouped_convT = torch.compile(ungrouped_convT)

# depthwise
depthwise_convT = torch.nn.ConvTranspose2d(**depth_wise_dict).to("cuda")
depthwise_convT = torch.compile(depthwise_convT)

# Forward once to get output shape
with torch.no_grad():
    dummy_out = dummy_conv(input_imgs)
    ungrouped_out = ungrouped_convT(dummy_out)
    depthwise_out = depthwise_convT(dummy_out)
    print("input: ",input_imgs.shape)
    print("init: ",dummy_out.shape)
    print("ungroupd:",ungrouped_out.shape)
    print("depth_out",depthwise_out.shape)
    print("=======")
    print("dummy weight",dummy_conv.weight.shape)
    print("ungrouped weight",ungrouped_convT.weight.shape)
    print("depthwise weight",depthwise_convT.weight.shape)

# Benchmark function
def run_dummy():
    dummy_conv(input_imgs)

def run_ungrouped():
    ungrouped_convT(dummy_out)

def run_depthwise():
    depthwise_convT(dummy_out)


# Run benchmark
# print("Benchmarking Dummy")
# benchmark(
#     partial(run_dummy),
#     input_tensor=input_imgs,
#     output_tensor=dummy_out,
#     weight_tensor=dummy_conv.weight,
#     num_warmups=10,
#     num_iterations=100
# )

print("Benchmarking ungrouped")
benchmark(
    partial(run_ungrouped),
    input_tensor=dummy_out,
    output_tensor=ungrouped_out,
    weight_tensor=ungrouped_convT.weight,
    num_warmups=10,
    num_iterations=100
)

print("Benchmarking depthwise")
benchmark(
    partial(run_depthwise),
    input_tensor=dummy_out,
    output_tensor=depthwise_out,
    weight_tensor=depthwise_convT.weight,
    num_warmups=10,
    num_iterations=100
)


