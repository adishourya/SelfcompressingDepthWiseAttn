= Method <section_method>
// ------------- ------------- -------------
// Approach Overview with a labeled diagram (no multi-stage)
// Components
// Implementation Details:
  // 1.Method Implementation
  // 2.Architectural Choices, algorithms and details
// Datasets:
// Show criteria and counts, Dataset Description
// acess details in footnotes or appendix.
// 
// 
// ------------- ------------- -------------
// Plan:
// method : modifications in effective vit module with self compressing blocks
// Transformer block , complete
// Throughput Graph

#figure(
  grid(rows: 2,
  row-gutter: 10pt,
  include "../graphs/module_graph.typ",
  include "../graphs/architecture.typ"
  ),
  scope: "parent",
  placement:top,
  caption:[*Model architecture*.  Each transformer block consists of an expansion stage using depthwise convolution (ConvD), an upscaling stage using transposed convolution (ConvT), another depthwise convolution, a projection stage, and a lightweight linear attention module. The output is passed through an MLP head to produce logits for the dense prediction task.]
) <figure_architecture>

#figure(
  grid(columns: 3,
  include "../graphs/throughput_conv.typ",
  include "../graphs/throughput_convt.typ",
  include "../graphs/throughput_attn.typ",
  ),
  scope: "parent",
  placement:top,
  caption:[*Module Throughput*: Throughput evaluation of convolutional, transposed convolutional, depthwise convolutional, and attention-based modules across varying input resolutions on an NVIDIA 4070 @nvidia-4070-mobile. The results demonstrate that depthwise operations and linear attention sustain significantly higher throughput as resolution increases, reflecting their computational efficiency compared to their ungrouped or softmax-based counterparts.]
) <figure_throughput>

// ------------
// 1. Model Architecture.
// - we take ideas from efficient vit @cai2024efficientvitmultiscalelinearattention and @Zhang_2025 in designing our architecture.
// - The image goes through a embedding where each embedding dimension is associated with a convolution kernel
// - the transformer block is a sequential operation of convolution and transpose convolution operation as learned upscaling
// - and then finally apply linear attention to capture global receptive field.

// - we quantize each of the module and penalize the module proportionally to the size of the size of the matrices and inversely proportional to the throughput of each module by adding the penalizing factor as hinge loss to cross entropy

// Loss = cross_entropy + penalty

// we now go into more detail on module details :


We design our model by adapting ideas from recent vision models @cai2024efficientvitmultiscalelinearattention and @Zhang_2025. Our architecture follows a decoder–only transformer structure. The input image is first projected into a latent sequence of embeddings, where each embedding dimension corresponds to a learnable convolution kernel. Each transformer block is then implemented as a sequential composition of convolutional and transpose convolution operations @zeiler2013visualizingunderstandingconvolutionalnetworks, consisting of channel expansion (convolution), learned upscaling (grouped transposed convolution), depthwise convolution, and channel projection (convolution). After these local operations, a linear attention module is applied to capture global receptive fields as in @figure_architecture.

We quantize all matrices associated to the compute modules as the primary method to quantize and prune modes of weights. Specifically, the penalty of operations is designed to be proportional to the size of the parameter matrices and inversely proportional to the measured throughput on target device for each module. This penalty is incorporated into the training objective as a hinge–style regularizer, resulting in the overall loss function where $L_o$ is the orignal loss of the model and $gamma$ acts as the compression factor over the module size normalizing constant $"C"$ as in @equation_loss.
$ "Loss" = L_0 + gamma/"C" sum_(k=1)^"all modules" alpha_k "size"_k $ <equation_loss>

For, our experiments use quantization as used in @cséfalvay2023selfcompressingneuralnetworks to quantize weights in the forward pass and use symmetric differentiable number Q8A format @micikevicius2022fp8formatsdeeplearning given in @equation_quantization. The bit-depth $b$ and the scaling factor $s$ is shared per mode of parameters in $W$.

$ Q(W,b,s)_k = 2^e  #sym.floor.l "clip"(W/2^s ,-2^(b-1), 2^(b-1)-1 ) #sym.ceil.r $ <equation_quantization>

The $#sym.floor.l dot #sym.ceil.r$ acts as a straight through estimator @bengio2013estimatingpropagatinggradientsstochastic for the rounding function which returns the identity of upstream gradient during backward pass.  

In the following sections, we use similar methods from @cséfalvay2023selfcompressingneuralnetworks and derive the penalty cost for modules in our transformer block.

// Convolution:
// Need for convolution : local inductive bias. full softmax attention captures both local inductive bias and global receptive field. but full softmax attention is costlier.so we use convolution kernels excessively in our transformer block.
// we can make further effort in optimizing the convolution operation by quantizing it. we tie bit width to output mode. and define the size of the module . we then scale this inversely proportional to throughput and proportional to the weight of the layers.

// + *Convolution*: Convolutions are the most frequent operation inside our transformer block, so to reduce memory usage we quantize weights of every convolutional kernel. Let the weight tensor of a convolutional for some layer have shape $(O, I, H, W)$, where $O$ and $I$ denote the output and input modes, and $H, W$ the spatial dimensions. Convolution kernels provide a strong local inductive bias in inputs, with a computational cost of $O(O I H W k^2)$, where $k$ is the kernel size. This contrasts with full softmax attention @vaswani2023attentionneed, which operates on embedding dimension $d$ and patch size $p$ with cost $O((H W)^2 d p^(-2))$, making it quadratic in the spatial resolution. So we leverage convolution kernels inside of our vision tower instead of the traditional full softmax attention.

//   Each output channel $O_i$ is associated with a learnable bit–depth parameter $b_{i,l}$ and an exponent parameter $e_{i,l}$. The quantized weight is then obtained by scaling the raw kernel as in @equation_quantization. To capture the effective compression cost of a quantized convolution, we define the size of layer $l$ as

//   $ "size"(l) = I H W sum_(i=1)^O max(0, ceil(b_(i,l))) $ <eq_size>

//   This formulation penalizes kernels with higher bit allocations per channel, thereby encouraging compact representations that prefer fewer effective filter kernels. To further bias training toward efficient configurations, we scale the penalty inversely with the measured throughput of each layer. Layers that execute faster therefore incur a smaller compression penalty, aligning optimization with both compactness and runtime efficiency:

//   $ lambda_l = 1 / T_l $ <eq_lambda>

//   $ L_("pen",l) = lambda_l "size"(l) $ <eq_pen_layer>


+ *Convolution*: are most frequent operation in our transformer block because they provide a strong local inductive bias. While full softmax attention @vaswani2023attentionneed can capture both local patterns and long–range dependencies, its computational cost is much higher. Attention over an image of height $H$ and width $W$, with patch size $p$ and embedding dimension $d$, scales as $O((H W)^2 d p^(-2))$, making it quadratic in the spatial resolution. In contrast, a convolutional kernel of shape $(O, I, H, W)$, with kernel size $k$, has cost $O(O I H W k^2)$.

  We further optimize the convolution operation through quantization. Each output channel $O_i$ is tied to a learnable bit–depth parameter $b_{i,l}$ and an exponent parameter $e_{i,l}$. The quantized kernel weights are obtained by scaling the raw kernel as in @equation_quantization. To formalize the compression cost, we define the effective size of layer $l$ as

  $ "size"(l) = I H W sum_(i=1)^O max(0, ceil(b_(i,l))) $ <eq_size>

  This penalizes convolution module with higher bit allocations per output mode, pushing the weight towards more sparse filters in the filter bank. To align compression with runtime efficiency, we scale this penalty $lambda_l$ inversely with the measured throughput $T_l$ on the target device of each layer. Faster modules thus incur a smaller penalty, ensuring the optimization objective favors both compactness and speed:

  $ lambda_l = 1 / T_l $ <eq_lambda>
  $ L_("pen",l) = lambda_l "size"(l) $ <eq_pen_layer>

+ *Depthwise Convolution*: One way to improve throughput in convolutional layers is to apply strided convolution, which increases throughput linearly with spatial resolution, though at the expense of reducing the effective local receptive field. Another option is grouped convolution, where the input channels are divided into $g$ groups. In this case, the convolutional kernel has shape $(O, I/g, k^2)$, with $g$ denoting the number of groups. A special case of this is depthwise convolution, where the number of groups $g$ equals the number of input modes $I$. As shown in @figure_throughput, depthwise convolution @Zhang_2025 has a higher throughput than ungrouped convolution when spatial resolution increases.  

  In our design, we employ depthwise convolutions in the expansion phase of the vision tower. This ensures that the computational cost does not scale with the embedding dimension, unlike ungrouped convolutions. To regularize these layers, we apply the same heuristic penalty formulation as in the ungrouped case, thereby encouraging efficient yet expressive expansion blocks.
  
+ *Transposed Convolution*:  For the upscaling stage, we employ transposed convolution after the projection block, inspired by @shocher2020discretecontinuousconvolutionlayers, to generate upscaled features from expanded convolution outputs. Although transposed convolutions are more expensive than standard convolutions (see @figure_throughput), we opt for ungrouped transposed operation since they provide richer local features than depthwise alternatives. To mitigate cost, we restrict the number of output channels rather than performing full ungrouped upscaling, and we apply a stronger penalty to the module size. This strategy follows the bottleneck principle in ResNet @he2016resnet and the reduced-channel decoding of U-Net @ronneberger2015unet, where only a limited set of upscaled features is sufficient to recover fine spatial details. In the subsequent stages, we apply one final composition of depthwise convolution and strided convolution to efficiently downscale the features for linear attention module.

// ideas:
// like mentioned before softmax attention provides strong long and global receptive filed.
// we offload local receptive field to previous composition of convolution operations and only focus on global receptive field here.
// unlike softmax attention linear attention can capture this relation in linear time of spatial resolution.
// we apply @qt_vit , @katharopoulos2020transformersrnnsfastautoregressive.
// since the weight quantization can be broadcasted to each heads before any computation we can use existing mechanisms to still use flash linear attention @beck2025tiledflashlinearattention to further improve throughput.
// 

+ *Linear Attention*: While softmax attention captures both local and global dependencies, local context is already handled by composition of convolution functions in the previous stages of our transformer block. The linear attention module therefore focuses exclusively on modeling global receptive fields. Unlike standard attention, which scales quadratically with spatial resolution, linear attention computes global interactions in time linear with respect to spatial resolution, following @qt_vit and @katharopoulos2020transformersrnnsfastautoregressive.  

    We leverage per-head quantization of queries, keys, values, and outputs, which can be broadcasted prior to computation. This design enables integration with Flash Linear Attention @beck2025tiledflashlinearattention, improving runtime throughput while preserving linear complexity in token count.  

    Our implementation of the linear attention module reshapes spatial feature maps into a token sequence, applies the linear attention, and restores the original tensor layout. A residual connection with the learned output tensor is applied, followed by group normalization to ensure numerical stability.
