//---- Global Variables-----
#let all_off = false
#let my_global = (
  hide_section_plan :  true or all_off,
  hide_comments : true or all_off,
  hide_todo : false or all_off 
)


/// ------ Imports------
// get imports from typst universe
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()


// --------- our colors -----

// accent colour
// override accent color in main for weekly documents
#let my_colors = (
  // accent : rgb("#5e81ac"),
  accent : rgb("#3b4252"),
  // accent: rgb("#d08770"),
  accent1: rgb("#81a1c1"),
  accent2 : rgb("#a3be8c"),
  accent3 : rgb("#d08770"),
  accent4 : rgb("#8fbcbb"),
  accent5 : rgb("#2e3440"),
  accent6: rgb("#b48ead"),
  accent7 : rgb("#8fbcbb"),
  accent8: rgb("#57a773"),
  accent9: rgb("#4392f1"),
  accent10: rgb("#8d94ba"),
  accent11:rgb("#39304a").lighten(30%),
  accent12:rgb("#a9c8da"),
  accent13:rgb("#e0ccd7"),
  accent14:rgb("#c2b7cd"),
  accent15:rgb("#bf616a"),
  // alt_accent : rgb("#a3be8c"),
  alt_fg : rgb("#f1f4f7"),
  code_fg : rgb("#eceff4"),
  // code_fg : rgb("#fff9e8"),
  // code_fg : rgb("#f7d407"),
  plan_fg : rgb("#a3be8c"),
  alt_bg : rgb("#484848"),
  heading1_color : rgb("#003a6c"),
  // heading1_color : rgb("#2e3440"),
  // heading1_color : rgb("#000000"),
  link_color : rgb("#2e6798"),
  
  gradient_color:  gradient.linear(..color.map.crest),
  comment_color : gradient.linear(..color.map.flare),
  cmap_color : gradient.linear(..color.map.icefire),
  text_cmap_color:
  gradient.linear(..color.map.crest)

)

#let abstract_gradient = gradient.linear(
  my_colors.accent.lighten(30%),
  my_colors.accent.lighten(20%),
  my_colors.accent.lighten(10%),
  my_colors.accent,
  my_colors.accent.darken(10%),
  my_colors.accent.darken(20%),
  my_colors.accent.darken(30%),
)

#let icon(name) = {
	let path = "../icons/" + name + ".svg"
	box(
		height: .7em,
		inset: (x: 0.05em, y: -0.1em),
		image(path, 
		height: 1em
		)
	)
}

#let list_colors = (black,my_colors.accent)


#let my_template(doc) = [

  
  #set footnote(numbering: "*")
  // set column layout
  #let column_layout = 1
  
  // set fonts and paragraph settings
  // #set text(font: "Barlow", size: 11pt, spacing: .35em)
  // #set text(font: "Barlow", size: 12pt)
  // #set text(font: "Arial", size: 11pt, spacing: .35em)
  // #set text(font: "Arial", size: 11pt)
  #set text(font: "Barlow", size: 11pt)
  // #set text(font: "New Computer Modern",size:11pt)
  // #set text(size:11pt)
  // #set text(font: "IBM Plex Sans", size: 11pt)
  // #set text(font: "IBM Plex Mono", size: 11pt)
  // #set text(font: "Barlow", size: 11pt,spacing:0.22em)
  #set math.equation(numbering: "(1)")
  #set heading(numbering: "1.")
  // line height
  #set par(leading: 8pt, justify: true)
  #set grid(column-gutter: 8pt)
  // Configure paragraph properties.
  #set quote(block: true)
  #set columns(gutter: 12pt)
  
  // set raw text
  #show raw: set text(font:"Space Mono", size:6pt)
  #show raw : set par(leading:4pt)
  // plugin setup

  #codly(stroke:1pt+my_colors.accent)
  #codly(fill: my_colors.code_fg.lighten(10%),inset:2.5pt)
  #codly(zebra-fill: none)
  #codly(number-format:none) // turn off line numberings


  // set header and footer text

  // #let section_header = [#context {
  //   let selected = selector(heading).before(here())
  //   let level = counter(selected)

  //   let headings = query(selected)
  //   if headings.len() ==0 {
  //     return
  //   }
  //   let heading = headings.last()
    
  //   level.display("1.")
  //   h(0.2em)
  //   heading.body
  // }]  

  #let section_header = [#context {
    // no heading in the first page
    
    // Skip header on first page
    if here().page() == 1 {
      return
    }
    // Try to find a level-1 heading on the current page.
    let h = query(heading.where(level: 1).after(here()))
      .filter(h => h.location().page() == here().page())
      .at(0, default: {
        // If none on the current page, fall back to the last one before here.
        query(heading.where(level: 1).before(here())).at(-1, default: none)
      })

    // If no heading found at all, return nothing.
    if h == none {
      return
    }

    // If numbering exists, resolve and show it.
    let number = if h.numbering != none {
      numbering(h.numbering, ..counter(heading).at(h.location()))
    }

    // Return combined number and heading text.
    [#number #h.body]
  }]
  
  #let header_text1 = [#text(fill:my_colors.alt_fg)[#section_header]]
  #let header_text2 = [#text(fill: my_colors.alt_fg)[]]

  // #let footer_text1 =  [#smallcaps[#text(fill:blue)[Maastricht University #icon("maas_logo")]]]
  // #let footer_text1 =  [#icon("maas_logo")]
  #let footer_text1 = []

  #let footer_text2 = [Page #context {
    let counter = counter(page)
    counter.display("1 of 1",both:true)
  }]
  

  
  
  // style headings
  #show heading.where(level: 1): content => context{

  	v(.3em)
  	text(fill: my_colors.heading1_color, weight: "bold", 1.25em, content)

 }
 
  #show heading.where(level: 2): content => context{

  	v(.1em)
  	text(weight: "bold", 1.1em, content)
  	v(.3em)

 }
 
  #show heading.where(level: 3): content => context{
  
    v(.1em)
    text(weight: "bold", 1.05em, content)
    v(.2em)
  
  }
  
  // underline all links and refernce
  #show link: it => {text(my_colors.link_color,it)}
  // #show ref: it => {text(my_colors.link_color, underline(it))}
  #show ref: it => {text(my_colors.link_color, it)}
  
  // caption colors
  #show figure.caption: set text(.85em, fill: my_colors.alt_bg)

  // table settings

  #set table(
  stroke:none,
  gutter: 0.1em,
  fill: (x, y) =>{ 
    if x==0 and y==0 {my_colors.accent1.lighten(40%)}
    else if x == 0 and y > 0 {gray.lighten(80%)} 
    else if y==0 {gray.lighten(60%)} 
    else {my_colors.alt_fg.lighten(40%)}},
  inset: (right: 1.5em),
  )

  #show table.cell: it => {
    if it.x == 0 or it.y == 0 {
      text(size:8pt,[#it])
    }else if it.x == 0 and it.y == 0 {
      text(fill:white,[*#it*])
    }else if it.body == [] {
      text(size:8pt,[])
    } else {
      text(size:8pt,[#it])
    }
  }
  
  
  // Page setup
  #set page(
  	// paper: "presentation-4-3",
  	paper: "a4",
  	margin: (x: 41.5pt, top: 50.51pt, bottom: 70.51pt),
  	// margin: (x: 41.5pt),
  	columns: column_layout,
  	fill:rgb("#ffffff"),
  	header: align(left+horizon)[#header_text1 #h(1fr) #header_text2],
  	footer: [#footer_text1 #h(1fr) #footer_text2],
  	numbering: "1",
  	background: {
  	align(top)[#box(width: 100%, height: 30pt, fill: my_colors.accent)]
  	}
  )
  
  
  
    #doc
]

#let show_title(title) = [
  
  #let title_ls_top = (paint:my_colors.accent, thickness:2.5pt, cap:"round")
  #let tiltle_ls_bottom = (paint:my_colors.accent, thickness:1.5pt, cap:"round")
  #align(center)[#line(length: 100%,stroke: title_ls_top)]
  #align(center, text(20pt)[#title])
  #align(center)[#line(length: 100%,stroke: tiltle_ls_bottom)]
  
]



#let show_abstract(content) = [

  #rect(fill: my_colors.alt_fg, radius: 8pt,inset: 15pt, stroke:(bottom:(2pt+abstract_gradient)))[
  	#align(left)[
  			#content
  		]
  	#linebreak()
  	*Date* : #datetime.today().display("[month repr:long] [day], [year]")\
  	*URL* : #icon("github-mark") #link("https://github.com/adishourya/SelfcompressingDepthWiseAttn")
  		#align(right)[
  		#image("../icons/Maastricht_University_logo_(2017_new_version).svg", height: 15pt)
  	]
]

]
