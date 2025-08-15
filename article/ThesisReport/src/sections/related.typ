= Related Study <section_related>
// ------------ ------------ ------------
// Objectives Related Studies section
// 1.Summarize SOTA
// 2. Discuss strengths and limitations (pseudo bit depth)
// 3.Gaps (all modueles compressible)
// 4.How do we tackle gaps (key contribution 1)
// ------------ ------------ ------------
//  Plan:
// 5 studies.. efficient vit for selecting moduels efficiently to get high dense prediciton accuracy
// psuedo bit depth does quantizable pseudo compression through ste but not do pruning .. i.e needs atleast 1 bit per weight or weight group
// adaqat : another adaptive quantized aware training
// self compressing does not suffer from this .. however it penalizes the size of the matrices to improve throughput.. (we do parallel compute cost like flash attention is not exactly O(n^3) when done parallely)
// LayerDrop offers to stochastically turn on/off (prune) modules to gain high accuracy.


// efficient vit (sota) :
// 1. selects modules most effectively to perform dense prediction task.
// for example: offloads local inductive bias calculation to convolution kernels instead of using more expensive full softmax attention. and gets global receptive from linear attention
// another example : to get rich local inductive bias it performs a more cheaper (grouped) depth wise convolution.
// 2. although the careful selection of module improves throughput and accuracy. but a lot more could be done if it was also pseudo quantized
//
// 
// diffq (compression bit depth no pruning) and self compression( both compression and pruning)
// diffq basic introduction and adds noise for emulatiion quantization. self compression does it directly
// targets more abstract group of weights... instead coleasced group of memory.

// ------ Abstracts
// efficient vit
// High-resolution dense prediction enables many appealing real-world applications, such as computational photography, autonomous driving, etc. However, the vast computational cost makes deploying state-of-the-art high-resolution dense prediction models on hardware devices difficult. This work presents EfficientViT, a new family of high-resolution vision models with novel multi-scale linear attention. Unlike prior high-resolution dense prediction models that rely on heavy softmax attention, hardware-inefficient large-kernel convolution, or complicated topology structure to obtain good performances, our multi-scale linear attention achieves the global receptive field and multi-scale learning (two desirable features for high-resolution dense prediction) with only lightweight and hardware-efficient operations. As such, EfficientViT delivers remarkable performance gains over previous state-of-the-art models with significant speedup on diverse hardware platforms, including mobile CPU, edge GPU, and cloud GPU. Without performance loss on Cityscapes, our EfficientViT provides up to 13.9× and 6.2× GPU latency reduction over SegFormer and SegNeXt, respectively. For super-resolution, EfficientViT delivers up to 6.4x speedup over Restormer while providing 0.11dB gain in PSNR. For Segment Anything, EfficientViT delivers 48.9x higher throughput on A100 GPU while achieving slightly better zero-shot instance segmentation performance on COCO
//
// - differentiable
//  We propose DiffQ a differentiable method for model compression for quantizing model parameters without gradient approximations (e.g., Straight Through Estimator). We suggest adding independent pseudo quantization noise to model parameters during training to approximate the effect of a quantization operator. DiffQ is differentiable both with respect to the unquantized weights and the number of bits used. Given a single hyper-parameter balancing between the quantized model size and accuracy, DiffQ optimizes the number of bits used per individual weight or groups of weights, in end-to-end training. We experimentally verify that our method is competitive with STE based quantization techniques on several benchmarks and architectures for image classification, language modeling, and audio source separation. For instance, on the ImageNet dataset, DiffQ compresses a 12 layers transformer-based model by more than a factor of 8, (lower than 4 bits precision per weight on average), with a loss of 0.3% in model accuracy

// - self compressing
// In our experiments we compare with the related method of Défossez et al. (2022), as described in more detail in the Experiments section below. The following differences with our method should be noted: 1. We allow bit depths to reduce to zero, eliminating some weights, instead of limiting minimum com- pression to 1 bit. 2. We define the quantization function in such a way that it is fully differentiable with respect to all pa- rameters, including the number format parameters (scale/exponent and bit depth (Jacob et al. 2017)). Importantly, this turns all number format parame- ters into network parameters that can be trained di- rectly as if they were weights. 3. We use the basic STE for all training instead of us- ing pseudo-quantization noise. 4. We use a coarser grouping of weights: instead of using groups of 4, 8 or 16 weights, we group all weights in a channel, achieving greater stability and less forgetting during training. This also allows for a significant reduction in compute requirements without requiring specialized hardware by a com- plete elimination of channels.

// layerdrop
// Overparameterized transformer networks have obtained state of the art results in various natural language processing tasks, such as machine translation, language modeling, and question answering. These models contain hundreds of millions of parameters, necessitating a large amount of computation and making them prone to overfitting. In this work, we explore LayerDrop, a form of structured dropout, which has a regularization effect during training and allows for efficient pruning at inference time. In particular, we show that it is possible to select sub-networks of any depth from one large network without having to finetune them and with limited impact on performance. We demonstrate the effectiveness of our approach by improving the state of the art on machine translation, language modeling, summarization, question answering, and language understanding benchmarks. Moreover, we show that our approach leads to small BERT-like models of higher quality compared to training from scratch or using distillation.


// adaqat
// Large-scale deep neural networks (DNNs) have achieved remarkable success in many application scenarios. However, high computational complexity and energy costs of modern DNNs make their deployment on edge devices challenging. Model quantization is a common approach to deal with deployment constraints, but searching for optimized bit-widths can be challenging. In this work, we present Adaptive Bit-Width Quantization Aware Training (AdaQAT), a learning-based method that automatically optimizes weight and activation signal bit-widths during training for more efficient DNN inference. We use relaxed real-valued bit-widths that are updated using a gradient descent rule, but are otherwise discretized for all quantization operations. The result is a simple and flexible QAT approach for mixed-precision uniform quantization problems. Compared to other methods that are generally designed to be run on a pretrained network, AdaQAT works well in both training from scratch and fine-tuning this http URL results on the CIFAR-10 and ImageNet datasets using ResNet20 and ResNet18 models, respectively, indicate that our method is competitive with other state-of-the-art mixed-precision quantization approaches.

//---------------------- ---------------------- ---------------------- ----------------------

Our methodology builds upon recent advancements in efficient vision towers and quantization approaches, drawing insights from prior work. Below, we summarize key contributions from related works that have informed our approach.

+ *EfficientViT* @cai2024efficientvitmultiscalelinearattention carefully selects computational modules to maximize accuracy while maintaining high throughput. The architecture strategically offloads local inductive bias computation to lightweight convolution kernels, avoiding the computationally expensive full softmax attention. For example, it employs grouped depthwise convolutions to capture rich local structure at low cost, while using linear attention to obtain global receptive fields.

  This careful allocation of operations results in significant speedups while maintaining state-of-the-art accuracy across dense prediction tasks. However, EfficientViT focuses on module selection and efficient operations without incorporating compression strategies, which could further improve both throughput and accuracy while reducing model storage cost.

+ *LayerDrop *@fan2019reducingtransformerdepthdemand applies structured, layer-wise dropout in Transformer networks. During training, entire layers are randomly skipped either stochastically or through learned skipping which regularizes the network and makes it robust to layer removal. At inference, this enables extraction of arbitrary-depth sub-networks from a single model without retraining, providing scalability across compute budgets. While effective for pruning at the layer level, LayerDrop does not perform fine-grained compression such as bit-depth quantization or weight-level pruning.

+ *DiffQ *@défossez2022differentiablemodelcompressionpseudo presents a differentiable quantization framework that learns both weights and their bit depths without gradient approximations. To emulate the effect of quantization during training, it injects independent pseudo-quantization noise, enabling end-to-end optimization of precision at the level of individual weights or small groups, governed by a single accuracy–size trade-off hyperparameter. This design achieves competitive compression across vision, language, and audio tasks.

+ *Self-Compressing Networks *@cséfalvay2023selfcompressingneuralnetworks remove the need for noise emulation by directly learning bit depths as trainable parameters, which can be reduced to zero to prune weights or entire channels. Unlike DiffQ’s coalesced memory grouping, they adopt coarser mode-level (channel) grouping for greater stability and reduced forgetting. They also treat number format parameters (scale, exponent, bit depth) as optimizable network parameters, allowing complete channel elimination without specialized hardware, and rely on the Straight-Through Estimator instead of pseudo-noise for gradient flow.
