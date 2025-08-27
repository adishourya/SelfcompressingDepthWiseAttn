= Dicussion <section_discussion>
// -------------------------------
// Interpret findings and patterns
// relflect on unexpected trends.
// limitations
// future work
// Social/Scientific Impacts
// -------------------------------

Our results demonstrate that self-compressing vision towers can closely match the accuracy of state-of-the-art lightweight backbones such as EfficientViT, while often providing modest improvements on target device in throughput and computational cost. The gains are most evident at smaller scales, where pruning and quantization remove redundant tensor channels without harming accuracy, and at larger scales, where modest accuracy improvements are coupled with faster inference. These findings support our central hypothesis: not all modules contribute equally to performance (@figure_bitdepth), and selectively allocating precision and compute across layers can yield a better balance of accuracy and efficiency.


An important aspect of our approach is that pruning is applied only at inference time. This design choice avoids irreversible pruning specially with small batchsize, where permanently removing parameters during training might lead to degraded accuracy. Since we do not re-compile the computational graph during training, the pruned modules remain "active" (zero tensors) in the forward and backward passes, and as a result, training time across epoch remains constant. While this limits efficiency gains during optimization, it ensures that the model retains full representational capacity during learning and only discards redundant modules once training is complete, leading to stable accuracy at deployment.



// At the same time, our experiments also reveal limitations. In segmentation, for example, our model achieves nearly the same mIoU as EfficientViT but does not surpass it. This suggests that while pruning improves efficiency, dense prediction tasks with high spatial complexity may require more careful trade-offs to maintain top accuracy. Similarly, optical flow results show that our method matches EfficientViT but does not exceed its accuracy, even though inference is faster. These cases indicate that pruning and quantization can occasionally reduce representational flexibility, limiting accuracy gains.

// On CUB-200-2011, where classes are fine-grained and visually similar, compression has little negative effect, and our model can even slightly outperform baselines. In contrast, on Country211, which is noisier and more diverse, improvements are modest. This suggests that dataset characteristics, such as intra-class variation and data imbalance, play a role in how much efficiency can be gained without hurting accuracy. Future work could adapt pruning schedules or quantization precision dynamically based on dataset difficulty or task type.
// 

