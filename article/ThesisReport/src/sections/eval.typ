#import "../template.typ":my_colors
#import "../callouts.typ":*

= Evaluation <section_evaluation>
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


#let cifar100_table =figure(
table(
  columns: 6,
  [Model],[Size],[Top1],[Top5],[FLOPS],[Throughput H100 (Images/sec)],

  [EfficientVIT],  [0.98M],  [-],  [-],  [-],  [-],
  [LayerDrop],     [1.12M],  [-],  [-],  [-],  [-],
  [*Ours*],        [0.88M],  [-],  [-],  [-],  [-],
  [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)],

  [EfficientVIT],  [2.41M],  [-],  [-],  [-],  [-],
  [LayerDrop],     [2.50M],  [-],  [-],  [-],  [-],
  [*Ours*],        [1.98M],  [-],  [-],  [-],  [-],
  

  
),caption:[cifar100],scope: "parent",placement: auto)


#let country_table =figure(
table(
  columns: 6,
  [Model],[Size],[Top1],[Top5],[FLOPS],[Throughput H100 (Images/sec)],

  [EfficientVIT],  [0.98M],  [-],  [-],  [-],  [-],
  [LayerDrop],     [1.12M],  [-],  [-],  [-],  [-],
  [*Ours*],        [0.88M],  [-],  [-],  [-],  [-],
  [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)], [#line(stroke: 0.1pt)],

  [EfficientVIT],  [2.41M],  [-],  [-],  [-],  [-],
  [LayerDrop],     [2.50M],  [-],  [-],  [-],  [-],
  [*Ours*],        [2.13M],  [-],  [-],  [-],  [-],

  
),caption:[country211],scope: "parent",placement: auto)


#let ade_table=figure(
table(
  columns: 5,
  [Model],[Size],[MIOU],[FLOPS],[Throughput H100 (Images/sec)],

  [EfficientVIT],  [2.41M],  [-],  [-],  [-],
  [LayerDrop],     [2.50M],  [-],  [-],  [-],
  [*Ours*],        [2.13M],  [-],  [-],  [-],

  
),caption:[ade],scope: "parent",placement: auto)


#let ade_table=figure(
table(
  columns: 5,
  [Model],[Size],[MIOU],[FLOPS],[Throughput H100 (Images/sec)],

  [EfficientVIT],  [2.41M],  [-],  [-],  [-],
  [LayerDrop],     [2.50M],  [-],  [-],  [-],
  [*Ours*],        [2.13M],  [-],  [-],  [-],
  
),caption:[ade],scope: "parent",placement: auto)



#let hd1k_table=figure(
table(
  columns: 5,
  [Model],[Size],[EPE],[FLOPS],[Throughput H100 (Images/sec)],

  [EfficientVIT],  [2.41M],  [-],  [-],  [-],
  [LayerDrop],     [2.10M],  [-],  [-],  [-],
  [*Ours*],        [2.13M],  [-],  [-],  [-],
  
),caption:[hd1k],scope: "parent",placement: auto)


//------------------------------
#lorem(500)

#table_color(color:my_colors.accent14.lighten(40%),  cifar100_table)
#table_color(color:my_colors.accent10.lighten(40%),  country_table)
#table_color(color:my_colors.accent4.lighten(40%),   ade_table)
#table_color(color:my_colors.accent2.lighten(40%),   hd1k_table)
