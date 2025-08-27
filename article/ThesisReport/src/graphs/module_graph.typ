#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import fletcher.shapes: house
// #set text(font: "New Computer Modern",size:0.8em)
#set text(size:0.8em)

#let blob(pos, label, tint: white, ..args) = node(
  pos, align(center, label),
  width: 24mm,
  fill: tint.lighten(60%),
  stroke: 1pt + tint.darken(20%),
  corner-radius: 1pt,
  ..args,
)

#let node_pos = (
  "expand": (1,0),
  "upscale": (2,0),
  "depth": (3,0),
  "proj": (4,0),
  "attn": (5,0),
  "title": (3, 0.8),
)

#diagram(
  spacing: 8pt,
  cell-size: (8mm, 10mm),
  edge-stroke: 1pt,
  edge-corner-radius: 3pt,
  mark-scale: 70%,

  // pipeline nodes (left -> right)
  blob(node_pos.expand, [Expand\ Block(ConvD)],   tint: yellow, name: <exp>),
  edge("->"),
  blob(node_pos.upscale, [Upscaling\ Block(ConvT)], tint: olive, name: <up>),
  edge("->"),
  blob(node_pos.depth,  [Depthwise\ Block(ConvD)], tint: orange, name: <dw>),
  edge("->"),
  blob(node_pos.proj,   [Projection\ Block(Conv)], tint: olive, name: <prj>),
  edge("->"),
  blob(node_pos.attn,   [Linear\ Attention], tint: yellow,  name: <la>),

  // enclosing box (light brown, behind everything)
  node(
    (3,0),
    shape: rect,
    enclose: (<exp>, <up>, <dw>, <prj>, <la>),
    // fill: rgb("e0e0e0"),   // light brown shade
    fill:gray.lighten(40%),
    stroke: black,
    corner-radius: 3pt,
    // outset: 10pt,
    name: <block>
  ),

  // caption above
  node(node_pos.title, [Transformer Block], stroke: none)
)
