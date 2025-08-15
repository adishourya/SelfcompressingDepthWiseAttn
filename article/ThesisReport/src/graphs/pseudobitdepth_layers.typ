 #import "@preview/cetz:0.3.1"
 #import "@preview/cetz-plot:0.1.0"

 #let values = (
 1.0193889141082764*4,
 1.0656800270080566*4,
 1.0056366920471191*4,
 0.8189668297767639*4,
 0.6289276671409607*4,
 0.6241380643844604*4,
 0.5219485759735107*4,
 0.52187111854553*4
 )

 #let errors = (0.23,0.122,0.65,0.22,0.19,0.35,0.24,0.22)
 #let data = range(values.len()).map(i => (i+1, values.at(i), errors.at(i)))

 #cetz.canvas({
   import cetz-plot: *
   plot.plot(
     size: (16, 4),
     y-grid: true,
     x-grid: false,
     // x-min:0.5,x-max:8.5,
     // x-tick: 1,
     y-max: 5,
     y-tick-step:1,
     x-label: "Layer",
     y-label: "Bit Depth",
     legend: "inner-north-east",
     {
       plot.add-bar(
         bar-width: 0.58,
         x-key: 0,
         y-key: 1,
         error-key: 2,
         data,
       )
     },
   )
 })
