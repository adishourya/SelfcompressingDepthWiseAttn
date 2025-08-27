#import "@preview/cetz:0.3.1"
#import "@preview/cetz-plot:0.1.0"

// The data for the plot.
#let data = (
  (0, 4.9332, 0.12),
  (1, 4.0775556564331055, 0.23),
  (2, 4.312720108032227, 0.122),
  (3, 4.022546768188477, 0.65),
  (4, 3.2758673191070558, 0.22),
  (5, 2.515710668563843, 0.19),
  (6, 2.4965522575378416, 0.35),
  (7, 2.087794303894043, 0.24),
  (8, 2.08748447418212, 0.22),
)

// This creates the bar plot.
#cetz.canvas({
  import cetz-plot: *
  plot.plot(
  axis-style: "scientific-auto",
    size: (10, 4),
    y-grid: true,
    x-grid: false,
    x-min: -0.5,
    x-max: 8.5,
    y-max: 6.5,
    x-tick-step:none,
    // The x-tick-step parameter is removed to prevent the numerical labels from appearing.
    y-tick-step: 1,
    x-label: "Model Layers and Embedding Convolutions",
    y-label: "Average Bit Depth",
    legend: "inner-north-east",
    
    // This is the key change. We manually define a list of tuples for the ticks.
    // Each tuple contains the numerical position on the x-axis and the string label to display there.
    x-ticks: (
      (0, "Embed"),
      (1, "L0"),
      (2, "L1"),
      (3, "L2"),
      (4, "L3"),
      (5, "L4"),
      (6, "L5"),
      (7, "L6"),
      (8, "L7"),
    ),
    
    {
      plot.add-bar(
        bar-width: 0.88,
        x-key: 0,
        y-key: 1,
        error-key: 2,
        data,
      )
      // plot.add-hline(4)      

    }
  )
})
