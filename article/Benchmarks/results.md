~/code/um/sem4/thesis/article/benchmarks (master*) » python bench_conv.py
input:  torch.size([256, 3, 224, 224])
init:  torch.size([256, 32, 224, 224])
ungroupd: torch.size([256, 32, 224, 224])
depth_out torch.size([256, 32, 224, 224])
dummy weight torch.size([32, 3, 3, 3])
ungrouped weight torch.size([32, 32, 3, 3])
depthwise weight torch.size([32, 1, 3, 3])
benchmarking dummy
avg exec time: 24.75369 ms
throughput: 72.64822 gb/s
benchmarking ungrouped
avg exec time: 53.81551 ms
throughput: 61.10452 gb/s
benchmarking depthwise
avg exec time: 35.02566 ms
throughput: 93.88360 gb/s


~/code/um/sem4/thesis/article/Benchmarks (master*) » python bench_conv.py
Resolution = 32x32
input:  torch.Size([256, 3, 32, 32])
init:  torch.Size([256, 32, 32, 32])
ungroupd: torch.Size([256, 32, 32, 32])
depth_out torch.Size([256, 32, 32, 32])
dummy weight torch.Size([32, 3, 3, 3])
ungrouped weight torch.Size([32, 32, 3, 3])
depthwise weight torch.Size([32, 1, 3, 3])
Benchmarking ungrouped
Avg Exec Time: 1.00554 ms
Throughput: 66.77597 GB/s
Benchmarking depthwise
Avg Exec Time: 0.65548 ms
Throughput: 102.38256 GB/s

~/code/um/sem4/thesis/article/Benchmarks (master*) » python bench_conv.py
Resolution = 224x224
input:  torch.Size([256, 3, 224, 224])
init:  torch.Size([256, 32, 224, 224])
ungroupd: torch.Size([256, 32, 224, 224])
depth_out torch.Size([256, 32, 224, 224])
dummy weight torch.Size([32, 3, 3, 3])
ungrouped weight torch.Size([32, 32, 3, 3])
depthwise weight torch.Size([32, 1, 3, 3])
Benchmarking ungrouped
Avg Exec Time: 53.39827 ms
Throughput: 61.58198 GB/s
Benchmarking depthwise
Avg Exec Time: 35.21302 ms
Throughput: 93.38409 GB/s


~/code/um/sem4/thesis/article/Benchmarks (master*) » python bench_convT.py
Resolution = 64x64
input:  torch.Size([256, 3, 64, 64])
init:  torch.Size([256, 32, 64, 64])
ungroupd: torch.Size([256, 32, 127, 127])
depth_out torch.Size([256, 32, 127, 127])
=======
dummy weight torch.Size([32, 3, 3, 3])
ungrouped weight torch.Size([32, 32, 3, 3])
depthwise weight torch.Size([32, 1, 3, 3])
Benchmarking ungrouped
Avg Exec Time: 14.51271 ms
Throughput: 45.66822 GB/s
Benchmarking depthwise
Avg Exec Time: 12.51806 ms
Throughput: 52.94222 GB/s
