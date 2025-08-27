// use #comment[comment here] <- to add comments
// use #todo[todo text here]  <- to add todo
//--------------------------------

#import "src/template.typ" : *
#show: my_template
#import "src/callouts.typ": *

// Title

#let title = [
Self-Compressing Vision Tower for Efficient #linebreak() Dense Prediction Tasks
]


#show_title(title)

// authors
#grid(columns: (1fr,1fr,1fr),

  align(center)[Aditya Shourya
  #link(<affiliation_a>,super[a])\
  #text(size:8pt,
  "a.shourya@student.maastrichtuniversity.nl")],

  
  align(center)[Guangzhi Tang
  #link(<affiliation_a>,super[a])\
  #text(size:8pt,
  "guangzhi.tang@maastrichtuniversity.nl")],
  
  align(center)[Chang Sun #link(<affiliation_a>,super[a])
  #h(-0.2em)#super[,]#h(-0.2em)
  #link(<affiliation_a>,super[b])\
  #text(size:8pt,
  "chang.sun@maastrichtuniversity.nl")]
  
)


#text(0.6em,style: "italic",
[#super([b]) Institute of Data Science, Faculty of Science and Engineering, Maastricht University, Maastricht, The Netherlands])<affiliation_a>

#text(0.6em,style:"italic",
[#super([a]) Department of Advanced Computing Sciences, Faculty of Science and Engineering, Maastricht University, Maastricht, The Netherlands
])<affiliation_b>

// #text(5pt,style:"italic",
// [#super([c]) work done during master thesis... at #super[a]])<affiliation_star>



// abstract
#let abstract = [
High-resolution dense prediction, essential for tasks such as classification, segmentation, and optical flow, requires models that balance accuracy with efficiency. Most state-of-the-art architectures rely on deep sequential operations that are computationally expensive and difficult to deploy on resource-constrained devices. We introduce a novel, self-compressing vision architecture that integrates structured pruning and quantization across key modules: convolutional layers, transposed convolutions, and linear attention in proportion to their computational cost. By strategically reducing precision and pruning tensors in less critical layers, our method achieves significant model compression with minimal impact on accuracy. We evaluated our approach on fine-grained classification (CUB-200-2011, Country211), semantic segmentation (ADE20K), and optical flow (HD1K). Our models consistently match the accuracy of the current state-of-the-art, EfficientViT, while often reducing both storage requirements and FLOPs. Furthermore, our method outperforms DiffQ in both accuracy and efficiency under identical parameter constraints. We conclude that compression can be a constructive tool for not only reducing model size but also enabling the inspection of module importance across layers, which facilitates better model design for specific target devices.  
]
#show_abstract(abstract)

#columns(2)[
  #include "src/sections/intro.typ"
  #include "src/sections/related.typ"
  #include "src/sections/method.typ"
  #include "src/sections/experiment.typ"
  #include "src/sections/eval.typ"
  #include "src/sections/discussion.typ"
  #include "src/sections/conclusion.typ"
]

#bibliography("src/refs.bib",style: "institute-of-electrical-and-electronics-engineers")

// #bibliography("refs.bib",style: "harvard-cite-them-right")
// #bibliography("refs.bib",style: "mla")
// #bibliography("refs.bib",style: "american-physics-society")

#pagebreak()

#counter(heading).update(0)
#set heading(numbering:"A.")
#include "src/sections/appendix.typ"
#include "src/sections/checklist.typ"

