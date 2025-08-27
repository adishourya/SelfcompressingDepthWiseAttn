= Experiments <section_experiment>

=== Setup <section_setup>

*Datasets*: We evaluate our model on three standard dense prediction tasks *classification*, *segmentation*, and *optical flow* problems. 

  + For *classification*, we use CUB-200-2011 @wah_branson_welinder_perona_belongie_2022 and Country211 @country211. CUB-200-2011 contains 11,788 images (≈6k train, 6k test) of 200 bird species, collected from online repositories between 2009–2011, with an average resolution around 500×500 pixels and fine-grained species labels. Country211 is a geo-location dataset derived from YFCC100M @thomee2016yfcc100m, constructed by sampling 150 training, 50 validation, and 100 test images per country for 211 countries (\~63k total). The raw images are drawn from Wikimedia Commons @wikimediacommons and other Creative Commons sources that vary in resolution; we resize them to 448×448 pixels for training. Together, these datasets test models both fine-grained species recognition (local biases) and global geographic locations (global biases).  

  + For *segmentation*, we use ADE20K @zhou2017ade20k, downscaled to match the average resolution of CUB-200-2011. ADE20K provides 20k training and 2k validation images with dense pixel-level annotations across 150 semantic categories, offering a broad benchmark for general scene understanding.  

  + For *optical flow*, we evaluate on HD1K @kondermann2016hd1k, which contains 1,066 high-definition video frames (1920×1080) with dense ground truth flow fields for real-world driving scenarios. To ensure consistent input resolution with our classification setting, we resize all frames to 448×448 pixels. This benchmark stresses robustness to large displacements, motion blur, and realistic driving conditions.  

All datasets are publicly available, we follow standard train/validation/test splits without additional filtering, and apply only normalization to the images, we downsample high resolution images to 448×448 for all the tasks.


*Throughput Measurement*: The compression penalty terms in our training objective are derived by measuring module throughput on our target device (@nvidia-4070-mobile) at FP16 precision while we train our model on a single H100 @nvidia-h100. For compute-bound operations, such as linear attention, we modify existing fused kernels @yang2024fla to encorporate our quantization approach. For memory-bound operations during training time, such as top-$k$ selection and cross-entropy, we rely on off-the-shelf implementations @daoailab-quack. Throughput is reported using the total data flight time, including both read and write operations.

*Model Implementation*: Our models are implemented in PyTorch @paszke2019pytorch (fused operations in @cutlass-cute and @triton2022openai) and trained using the AdamW optimizer @loshchilov2019decoupled. We adopt a cosine learning rate decay schedule for model parameters and a cosine ramp-up schedule for the compression coefficient ($gamma$ in @equation_loss), ensuring stable convergence while progressively increasing the effect of the quantization penalty.

All models are trained from random initialization, with each module initialized at 4-bit precision for quantization experiments. We report Top-1 and Top-5 accuracy for classification, mean Intersection over Union (mIoU) @everingham2010pascal for semantic segmentation, and average End-Point Error (EPE) @baker2011database for optical flow, following standard evaluation practice.

=== Ablation Studies <section_ablation>
All ablation studies to study the design of our transformer module are conducted on a model with 2.13M parameters, evaluated on the Country211 for classification @country211 prediction task.

*Pixel Shuffling vs Transposed Convolution*: We evaluate the use of pixel shuffling @shi2016realtimesingleimagevideo as an alternative to transposed convolution for the upscaling stage. Pixel shuffling is a non-learnable operation that increases spatial resolution by rearranging the input tensor. While this approach reduces computational overhead, it couples projection features across channels with different bit-depths thereby limiting the granularity of learned representations. As a result, pruning of projection convolutions is largely ineffective: channels remain coupled, and in our experiments, we observed no pruning of projection convolutions when pixel shuffling was applied.

*Pointwise Projection*:   We investigate pointwise convolutions (1×1 kernels) as the projection stage preceding the linear attention module. Although computationally efficient, quantized 1×1 kernels tend to collapse representations toward zero at intialization, particularly when low bit-depths are assigned at initializtion. To address this, we adopt 3×3 kernels for the projection stage instead of pointwise convolutions. In our experiments, we observe no degradation in training loss when comparing quantized 3×3 kernels with unquantized 1×1 kernels, indicating that slightly larger kernels maintain performance while supporting effective quantization and compression.

*Linear Attention in Parallel with Convolution Kernels*: To further improve throughput, we test running linear attention modules in parallel with convolutional kernels (similar to @Zhang_2025). This design aims to offload global receptive field computation without sequential bottlenecks. However, we observe poor GPU occupancy (at inference time) when many convolution kernels are pruned.


#let prune_fiure = figure(
  grid(columns: 3,
  include "../graphs/prune_heads.typ",
  include "../graphs/prune_upscalers.typ",
  include "../graphs/prune_conv.typ",
  ),
  scope: "parent",
  placement:bottom,
  caption:[*Module Pruning*: The number of active modules (attention heads, upscalers, and convolution kernels) is tracked over training epochs. We observe a gradual reduction in active modules, indicating that the model prunes redundant modes of tensor as training progresses.]
  )

#let bit_depth_figure = figure(
grid(columns:(0.65fr,0.4fr),

  include "../graphs/pseudobitdepth_layers.typ",
  include "../graphs/pseudo_decay.typ",
),
  scope: "parent",
  placement: bottom,
  caption:[*Average Bit Depth Across Model Layers*: The bit depth for each module is computed as the mean across all its quantized parameters. While the embedding layer maintains higher precision than initalization (4bits), deeper layers (L) progressively converge toward lower values. The average bit depth of our modules across all layers drops to 3bits.
  ]
)
 
#figure(
  grid(rows:2,
  [#prune_fiure <figure_prune>],
  [#bit_depth_figure <figure_bitdepth>]
  ),
  scope:"parent",
  placement: bottom
)

