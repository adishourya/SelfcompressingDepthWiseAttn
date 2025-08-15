= Introduction <section_introduction>
// ================================================================================
// Problem Statement: Quantized and Prunable Moduels by making Bit depth
// trainable for all the large compute units such as linear attn weights
// , convolution weights.
//
// Significance: Througphut Based Applications like
// autofocusing in Wildlife Photography, autonomous drones.
//
// Challenges: Many
// Methods need hardware aware stratergy to compress. we present a unified way
// to compress modules.
//
// Brief Description:Key Contribution: Present a Backbone
// model for dense prediction , segmenentation and optical flow problems. We
// write compute bounded kernels like linear attention for increasing throughput
// for Ada Lovelace based to improve on native torch compiled kernels.
//
// Thesis Structure: Describe Sections This will be for the end
// ================================================================================

High-resolution dense prediction remains a critical task in computer vision. The effectiveness of a vision model is often evaluated by its performance both in accuracy and latency on large-scale dense prediction benchmarks.

Low-compute vision models enable real-time applications such as autonomous drones, on-device image processing, and computational photography. In domains like sports or wildlife photography, throughput can be just as important as accuracy. However, most state-of-the-art vision models rely on deep, sequential architectures, which are  unsuitable for deployment on consumer-grade devices with limited computational resources. In addition to their role in standalone vision tasks, vision towers that perform well on dense prediction problems are also considered as candidates for the visual encoders in vision-language models specially in small Vision Language Models @beyer2024paligemmaversatile3bvlm.<check_significance>

Recent research suggest that the deeper layers of  models may contribute less to overall accuracy than earlier layers, which indicates over-parameterization of networks and challenges the need for high computational cost in deeper layers @dao2025hybrid, @gromov2025unreasonableineffectivenessdeeperlayers. For instance, MiniMax-O1 @minimax2025minimax01scalingfoundationmodels employs full softmax attention ($O(N^2)$) only in few initial layers, while adopting more compute efficient alternatives such as linear attention ($O(N)$) in the deeper layers to reduce total compute requirements.

Inspired by these insights, we propose a fully prunable and quantizable vision model as a backbone where all major learnable components such as convolution and attention weights and linear layers can be compressed and pruned to improve model storage efficiency and improve accuracy during low precision inference for dense prediction tasks.<check_brief_description>

A key hypothesis we examine in this work is the need for uniform allocation of model precision and compute cost across model depth. In a non self-compressing models @mobile_vit, @tiny_vit, weight precision and compute budgets are assigned identically to all layers regardless of their relative contribution to final performance. This uniform treatment can result in overparameterization and inefficient use of compute resources. <check_research_question> We extensive evaluate our model on image classification , segmentation and optical flow problems. Our model matches accuracy over existing state of the art method. and present the following key contribution: <check_contribution>
  + *Unified Compression Strategy*: We propose a unified framework that to quantize and compress the weights of modules proportionally to their computational cost, enabling consistent compression across all major learnable components Convolutional , Attention Weights, and Linear Layers via pruning and quantization.

  + *Efficient Deployment *: Our compressed model sustains accuracy levels comparable to the FP32 baseline during FP8 inference, while significantly improving throughput during inference.

