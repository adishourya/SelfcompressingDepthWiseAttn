#import "template.typ": *

// callouts 
// convert content to string
#let to-string(content) = {
  if content.has("text") {
    content.text
  } else if content.has("children") {
    content.children.map(to-string).join("")
  } else if content.has("body") {
    to-string(content.body)
  } else if content == [ ] {
    " "
  }
}

// gray lorem
#let glorem(num:int) = {
  text(fill:white.darken(20%), [#lorem(num)])
} 



// inline callouts
#let comment(content) = {
  if not(my_global.hide_comments) {
  text(8pt,style:"italic", weight: "medium",my_colors.comment_color,[#emoji.bubble.speech.r #underline[Comment] : #content])
  }
  else {
    []
  }
}

#let todo(content) = {
  if not(my_global.hide_todo) {
  text(8pt,weight: "medium",my_colors.gradient_color,[#emoji.notepad #underline[TODO] : #content])
  }
  else {
    []
  }
}

#let draw_line = line(length:100%, stroke:my_colors.accent)

#let quote_box(content, attribution) = {
	// Remove newlines from the content
	let stripped_content = str.replace(content, "\n", " ");
	rect(fill: alt_fg, stroke: accent)[
		#quote(attribution: attribution)[
			#align(left)[#stripped_content]
		]
	]
}


// block callouts
#let block-inset = 1em
#let block-radius = 0.5em

#let icon(name) = {
	let path = "../icons/" + name + ".svg"
	box(
		height: .7em,
		inset: (x: 0.05em, y: -0.22em),
		image(path, 
		height: 1.15em
		)
	)
}

// your colored table


#let bullet(content) = {
  if my_global.show_section_plan {
    rect(fill:rgb("#a3be8c"),radius:2pt)[
    #wrap-content(image("../icons/circle.svg",height:10pt),[#h(-0.8em)#content])
  ]
}}


#let coherence(title:"Coherence?",content) = {
  if not(my_global.hide_comments) {

  block(
  fill: rgb("#FDF1E5"),
  inset: block-inset, 
  radius: block-radius,
  width: 100%,
  {
    text(rgb("#EC7500"),weight: 700)[#icon("warning") #title]
    text(size:6pt)[#content]
  }
)}}

#let tasks(title:"Tasks",content) = {
  if not(my_global.hide_comments) {

  block(
  fill: rgb("#E5F8F8"),
  inset: block-inset, 
  radius: block-radius,
  width: 100%,
  {
    text(rgb("#00bfbc"),weight: 700)[#icon("flame") #title]
    text(size:6pt)[#content]
  }
)}}



#let plan(content) = {
  if not(my_global.hide_section_plan) {
  block(
	fill: rgb("#E6F0FC"),
	inset: block-inset, 
	radius: block-radius,
	width: 100%,
	{
		text(fill: rgb("#086DDD"), weight: 700)[#icon("info") Section Plan Scratchpad]
		text(size: 6pt)[#content]
	})
}
else {
  []
}
}


#let tip(content) = block(
	fill: rgb("#E5F8F8"),
	inset: block-inset, 
	radius: block-radius,
	width: 100%,
	{
   // 00bfbc flame
		text(fill: rgb("#00bfbc"), weight: 700)[Quick Tip]
		content
	}
)

#let success(title:"Intro",content) = block(
	fill: rgb("#E6F8ED"),
	inset: block-inset, 
	radius: block-radius,
	width: 100%,
	{
		heading(level: 3, text(rgb("#08B94E"), [#icon("check") #title]))
		content
	}
)


#let plan_line = line(length:100%, stroke:(paint:my_colors.gradient_color, dash:"dashed",thickness:1pt))
// #let plan(abcd) = [_section plan:_#text(7pt,style: "italic",abcd) #plan_line]


