#import "../template.typ":my_colors
#import "../callouts.typ":*

// #counter("table").update(0)
== Evaluation <section_evaluation>
//-----------------------------
// Describe Eval Stratergy
// list metrics
// equation and implementation details
// --- lets keep results in this section too
// remember to emphasize on numbers to focus
// show deviation wherever possible
// compare results.
//-----------------------------
// table entries


#let cub200_table =figure(
table(
  columns: 6,
  [model],[size],[top1(%)],[top5(%)],[FLOPS (G)],[throughput nvidia 4070 (images/sec)],

  [efficientvit],  [0.98m],  [71.3],         [95.5],   [7],            [309],
  [diffq],         [1.12m],  [68.1],         [92.3],   [9],            [281],
  [*ours*],        [0.88m],  [73.4],         [98.0],   [7],            [338],
  [#line(stroke:   0.1pt)],  [#line(stroke:  0.1pt)],  [#line(stroke:  0.1pt)],  [#line(stroke:  0.1pt)],  [#line(stroke:  0.1pt)],  [#line(stroke:  0.1pt)],
  [efficientvit],  [2.41m],  [76.2],         [100],    [24],           [262],
  [diffq],         [2.50m],  [71.3],         [100],    [28],           [276],
  [*ours*],        [1.98m],  [75.8],         [100],    [24],           [308],
  [#line(stroke:   0.1pt)],  [#line(stroke:  0.1pt)],  [#line(stroke:  0.1pt)],  [#line(stroke:  0.1pt)],  [#line(stroke:  0.1pt)],  [#line(stroke:  0.1pt)],
  [efficientvit],  [4.81m],  [81.5],         [100],    [48],           [183],
  [diffq],         [4.92m],  [78.1],         [100],    [54],           [172],
  [*ours*],        [4.68m],  [83.5],         [100],    [51],           [230],
 
 ),
caption:[comparison of our models on the cub-200-2011 @wah_branson_welinder_perona_belongie_2022 dataset.],
scope: "parent",
placement: auto)


#let country_table =figure(
table(
  columns: 6,
  [model],[size],[top1(%)],[top5(%)],[FLOPS (G)],[throughput nvidia 4070 (images/sec)],

  [efficientvit],  [2.41m],  [44],  [65],  [23],  [291],
  [diffq],         [2.50m],  [36],  [41],  [28],  [270],
  [*ours*],        [2.13m],  [43],  [65],  [25],  [324],
  [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)],

  [efficientvit],  [4.81m],  [52],  [80],  [48],  [228],
  [diffq],         [4.92m],  [38],  [65],  [54],  [213],
  [*ours*],        [4.68m],  [55],  [81],  [51],  [294],

  
),caption:[benchmark results on the country211 dataset @country211.],scope: "parent",placement: auto)


#let ade_table=figure(
table(
  columns: 5,
[model],[size],[miou],[FLOPS (G)],[throughput  nvidia    4070     (images/sec)],
[efficientvit],                             [4.68m],  [44.1],  [47],           [255],
[diffq],                                    [4.92m],  [40.1],  [54],           [241],
[*ours*],                                   [4.28m],  [43.7],  [44],           [282],
  
),caption:[semantic segmentation results on the ade20k dataset @zhou2017ade20k. where miou is mean intersection over union.],scope: "parent",placement: auto)


#let hd1k_table=figure(
table(
  columns: 5,
  [model],[size],[epe],[FLOPS (G)],[throughput nvidia 4070 (images/sec)],
  [efficientvit],  [4.81m],  [4.8],  [48],  [191],
  [diffq],         [4.92m],  [5.4],  [54],  [155],
  [*ours*],        [4.68m],  [5.0],  [51],  [227],
),caption:[optical flow estimation results on the hd1k dataset @kondermann2016hd1k where epe is end point error],scope: "parent",placement: auto)


//------------------------------
// #figure(
//   grid(
//     rows:4,
//   [#table_color(color:my_colors.accent14.lighten(40%),  cub200_table) <cub_table>],
//   table_color(color:my_colors.accent10.lighten(40%),  country_table),
//   table_color(color:my_colors.accent4.lighten(40%),   ade_table),
//   table_color(color:my_colors.accent2.lighten(40%),   hd1k_table)
//   ),
//   scope: "parent",
//   placement: top
// )


#figure(
  grid(
    rows:4,
    [#cub200_table <cub_table>],
    [#country_table<country_table>],
    [#ade_table<ade_table>],
    [#hd1k_table<hd_table>]
  ),
  scope: "parent",
  placement: top
)

// For all of our experiments, we report the inference-time size and FLOPs after module pruning of our model, ensuring that results reflect the actual configuration used during inference.

// On the CUB-200-2011 dataset (@cub_table), our models achieve accuracy that is comparable to EfficientViT across different parameter scales. In the smallest configuration (1M parameters), our model reaches 73.4% Top-1 accuracy, slightly higher than EfficientViT (71.3%), while also offering a throughput advantage (338 vs. 309 images/sec). At the medium scale (~2M parameters), our model closely matches EfficientViT (75.8% vs. 76.2%) but processes more images per second (308 vs. 262). At the larger scale (~5M parameters), we obtain a modest improvement in accuracy (83.5% vs. 81.5%) together with higher throughput (230 vs. 183). On the Country211 dataset (@country_table), performance follows a similar pattern: our model achieves results on par with EfficientViT at ~2M parameters, and shows a modest gain at the larger scale (55% vs. 52% Top-1 accuracy), while sustaining higher throughput. These comparisons indicate that our method maintains efficiency advantages while matching or slightly improving accuracy relative to EfficientViT.


// In semantic segmentation on ADE20K (@ade_table), our model achieves 43.7% mean IoU with 4.28M parameters, which is nearly identical to EfficientViT (44.1%). Importantly, our design reduces computational cost (44 vs. 47 FLOPS (G)) and provides higher throughput (282 vs. 255 images/sec). Compared to DiffQ (40.1%), our model achieves both higher accuracy and efficiency. These results suggest that our method retains strong segmentation performance while offering a favorable efficiency profile.

// For optical flow estimation on HD1K (@hd_table), our model reaches an End-Point Error of 5.0, very close to EfficientViT (4.8) and notably better than DiffQ (5.4). The key difference lies in throughput, where our model achieves 227 images/sec compared to 191 for EfficientViT. This shows that while accuracy remains nearly the same as the state-of-the-art baseline, our design enables faster inference, which can be valuable for real-time deployment scenarios. Overall, across classification, segmentation, and optical flow, our models match the performance of EfficientViT while often delivering efficiency improvements.
//
For all experiments, we report inference-time size and FLOPs after module pruning of our and diffq @défossez2022differentiablemodelcompressionpseudo model, so results reflect the actual deployed configuration.

In *Classification* On the CUB-200-2011 dataset (@cub_table), our models achieve accuracy comparable to EfficientViT across all the scales experimented. At \<1M parameters, we reach 73.4% Top-1 (vs. 71.3%) with higher throughput (338 vs. 309 images/sec). At ~2M, accuracy is nearly identical (75.8% vs. 76.2%) but throughput improves (308 vs. 262). At ~5M, we observe a modest gain (83.5% vs. 81.5%) while maintaining faster inference (230 vs. 183). On Country211 (@country_table), performance follows the same trend: our model matches EfficientViT at ~2M parameters and improves slightly at ~5M (55% vs. 52%) with higher throughput. These results show that our approach preserves accuracy while offering better efficiency.

In *semantic segmentation* on ADE20K (@ade_table), our model achieves 43.7% mIoU, nearly identical to EfficientViT (44.1%) but with lower FLOPs (44 vs. 47) and higher throughput (282 vs. 255). Compared to DiffQ (40.1%), our model is both more accurate and efficient, indicating robust performance in dense prediction tasks.

For *optical flow* on HD1K (@hd_table), our model attains 5.0 EPE, close to EfficientViT (4.8) and better than DiffQ (5.4). Throughput is higher (227 vs. 191), showing that our design matches the accuracy of state-of-the-art baselines while enabling faster inference.
