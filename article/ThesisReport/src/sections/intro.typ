= Introduction <section_introduction>
// ================================================================================
// Problem Statement: Quantized and Prunable Moduels by making Bit depth trainable for all the large compute units such as linear attn weights , convolution weights.
// Significance: Througphut Based Applications like autofocusing in Wildlife Photography, autonomous drones.
// Challenges: Many Methods need hardware aware stratergy to compress. we present a unified way to compress modules.
// Brief Description:
// Key Contribution: Present a Backbone model for dense prediction , segmenentation and optical flow problems. We write compute bounded kernels like linear attention for increasing throughput for Ada Lovelace based to improve on native torch compiled kernels.
// Thesis Structure: Describe Sections This will be for the end
// ================================================================================

// examples : high resolution dense prediction is an important task.the capabitility of the backbone is often tested on how well (accuracy/latency) they perform on large scale classification. many of the vision towers selected for a vision language model does this.

// The low computational cost of vision tower improves the efficiency of applications such as wildlife photography , autonomous drone applications.
// However most of the vision tower model are associated with deep layers (sequential computation).
// Which makes them not suitable for most of the consumer available devices.

// There is a paper that suggests that the final few layers of the model does not carry their weight. i.e the first few layers are way more influential than the final few layers.[need to find this paper]

// Also one of the newer techniques is to use full softmax attn in the first few layers and low cost linear attn in the final layers in llm.[find paper].
// So we present a completely prunable vision tower where all the learnable weights (significant sized,, so something like learnabale layernorm will not be pruned or quantized) can be pruned and quantized


High-resolution dense prediction remains a critical task in computer vision. The effectiveness of a vision model is often evaluated by its performance both in accuracy and latency on large-scale classification benchmarks. Many of the vision towers chosen for Vision Language models follow this principle.

Low-compute vision towers can significantly improve the efficiency of real-time applications such as autonomous drones, on-device image analysis, or wildlife photography. However, most existing vision towers rely heavily on deep, sequential computation, making them unsuitable for deployment on consumer-grade devices with limited compute budgets.

Recent studies suggest that the deeper layers of language models may contribute less to final performance compared to the early ones, challenging the need  for computational cost specially in the deeper layers of the network @gromov2025unreasonableineffectivenessdeeperlayers. For example [citation] where full softmax attention is used only in early layers, while lighter alternatives like linear attention are used in deeper layers to reduce cost.

Inspired by these insights, we propose a fully prunable and quantizable vision model as a backbone where all major learnable components—such as convolution and attention weights—can be compressed. 

#lorem(500)

