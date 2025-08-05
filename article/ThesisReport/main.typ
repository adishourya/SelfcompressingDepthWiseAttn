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

  align(center)[Aditya Shourya #link(<affiliation_b>,super[b])
  #h(-0.2em)#super[,]#h(-0.2em)
  #link(<affiliation_star>,super[c])\
  #text(size:8pt,
  "a.shourya@student.maastrichtuniversity.nl")],

  
  align(center)[Guangzhi Tang
  #link(<affiliation_a>,super[b])\
  #text(size:8pt,
  "guangzhi.tang@maastrichtuniversity.nl")],
  
  align(center)[Chang Sun #link(<affiliation_a>,super[a])
  #h(-0.2em)#super[,]#h(-0.2em)
  #link(<affiliation_a>,super[b])\
  #text(size:8pt,
  "chang.sun@maastrichtuniversity.nl")]
  
)


#text(5pt,style: "italic",
[#super([a]) Institute of Data Science, Faculty of Science and Engineering, Maastricht University, Maastricht, The Netherlands])<affiliation_a>

#text(5pt,style:"italic",
[#super([b]) Department of Advanced Computing Sciences, Faculty of Science and Engineering, Maastricht University, Maastricht, The Netherlands
])<affiliation_b>

#text(5pt,style:"italic",
[#super([c]) work done during master thesis... at #super[a]])<affiliation_star>

// abstract
#let abstract = [
#lorem(200)
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

