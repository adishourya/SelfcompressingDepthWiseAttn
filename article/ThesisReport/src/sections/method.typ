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
  caption:[Model Architecture]
) <figure_architecture>

#figure(
  grid(columns: 3,
  include "../graphs/throughput_conv.typ",
  include "../graphs/throughput_convt.typ",
  include "../graphs/throughput_attn.typ",
  ),
  scope: "parent",
  placement:top,
  caption:[Module Throughput]
)

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

We quantize all matrices associated to the compute modules as the primary method to quantize and prune modes of weights. Specifically, the penalty of operations is designed to be proportional to the size of the parameter matrices and inversely proportional to the measured throughput on target device for each module. This penalty is incorporated into the training objective as a hinge–style regularizer, resulting in the overall loss function where $L_o$ is the orignal loss of the model and $gamma$ acts as the compression factor over the normalizing constant $"C"$ as in @equation_loss.
$ "Loss" = L_0 + gamma/"C" sum_(k=1)^"all modules" alpha_k "size"_k $ <equation_loss>

For, our experiments use quantization as used in @cséfalvay2023selfcompressingneuralnetworks to quantize weights in the forward pass and use symmetric differentiable number Q8A format @micikevicius2022fp8formatsdeeplearning given in @equation_quantization. The bit-depth $b$ and the scaling factor $s$ is shared per mode of parameters in $W$.

$ Q(W,b,s)_k = 2^e  #sym.floor.l "clip"(W/2^s ,-2^(b-1), 2^(b-1)-1 ) #sym.ceil.r $ <equation_quantization>

The $#sym.floor.l dot #sym.ceil.r$ acts as a straight through estimator @bengio2013estimatingpropagatinggradientsstochastic for the rounding function which returns the identity of upstream gradient during backward pass.  

In the following sections, we derive the penalty cost for modules in our transformer block.
+ *Convolution*:#lorem(100)
+ *Transposed Convolution*:#lorem(100)
+ *Linear Attention*:#lorem(100)

