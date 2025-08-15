== Experiments <section_experiment>
//--------------------------------
// 1.Setup:
// Hardware and more.
// 2.Design
// Dataset split decisions.. parameter tuning decisions (like cosine increase)
// 3. Ablation studies [not explicitly said]
//--------------------------------

#figure(
  grid(columns: 3,
  include "../graphs/prune_heads.typ",
  include "../graphs/prune_upscalers.typ",
  include "../graphs/prune_conv.typ",
  ),
  scope: "parent",
  placement:bottom,
  caption:[Module Pruning]
)

#figure(
  include "../graphs/pseudobitdepth_layers.typ",
  scope: "parent",
  placement: bottom,
  caption:[Pseudo Model Bit Depth Across Layers]
)

#lorem(500)

== Ablation Studies <section_ablation>
#lorem(500)
