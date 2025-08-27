#import "@preview/cetz:0.3.1": canvas, draw
#import "@preview/cetz-plot:0.1.0": plot
#import "../callouts.typ":*

#set text(size:8pt)

#let plot_colors = color.map.viridis
#let color1 = plot_colors.at(0)
#let color2 = plot_colors.at(5)
#let style1 = (stroke:(paint: color1, thickness: 1pt, cap: "round"))
#let style2 = (stroke:(paint: color2, thickness: 1.5pt, cap: "round"))
#let legend_color = black

// -----data
#let data1 = (
   (32,6*1.5),
   (64,3*1.5),
   (128,2*1.5)
)


#let data2 = (
   (32,13*2),
   (64,11.8*2),
   (128,10.5*2)
)

#canvas({
   draw.set-style(legend:())
  import draw: *

  // Set-up a thin axis style
  set-style(axes: (stroke: 0pt, tick: (stroke: 0pt)),
            legend: (stroke: 0.1pt+legend_color, orientation: ttb, item: (spacing: 0.1), scale: 100%))

  plot.plot(size:(4,4),
    // axis-style:"school-book",
    x-tick-step:32 ,
    // x-mode:"log",
    // x-base:2,
    y-grid: true,
    x-grid: true,
    // y-tick-step: 0.15, 
    // y-min: 1.5, y-max: 5,
    y-tick-step:5,
    y-min:0,
    y-max:30,

    // y-mode:"log",
    // y-base:1,
  
    x-label:"Resolution",
    y-label:"Throughput (GB/s)",
    legend: "inner-north-east",
    // legend:"inner-north",
    // set-origin((0,0)),
    // legend:"north",
    {

      plot.add(data1,label:[#text(size:0.8em)[Softmax Attn]],style:style1)
      plot.add(data2,label:[#text(size:0.8em)[Linear Attn]],style:style2)

    })
})
