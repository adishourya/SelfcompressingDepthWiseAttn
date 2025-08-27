= Conclusion <section_conclusion>
// --------------------------------
// Final Summary
// --------------------------------
// We introduced a self-compressing vision tower that prunes and quantizes all major modules in proportion to their computational cost and layersize. Across classification, segmentation, and optical flow benchmarks, our model matches the accuracy of EfficientViT. Compared to DiffQ, our method consistently provides both higher accuracy and better throughput, highlighting the advantage of unified compression and pruning framework.

We developed a novel, self-compressing vision tower that strategically prunes and quantizes all major modules, including convolutions, transposed convolutions, and linear attention, based on their computational cost and layer size. Our model's performance on classification, segmentation, and optical flow benchmarks consistently matches the accuracy of the state-of-the-art EfficientViT, while often providing the added benefits of reduced storage and FLOPs. Furthermore, our method consistently surpasses DiffQ in both accuracy and throughput. We conclude that this constructive approach to compression not only reduces model size but also allows for the inspection of module importance across layers, facilitating more effective model design for specific target devices.
