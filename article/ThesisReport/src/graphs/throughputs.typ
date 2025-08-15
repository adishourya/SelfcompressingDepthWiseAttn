#import "@preview/cetz:0.3.1": canvas, draw
#import "@preview/cetz-plot:0.1.0": plot

#canvas({
    // Make the canvas larger
    width: 15cm,
    height: 10cm,

    plot.plot(
        x-label: `Batch Size`,
        y-label: `Throughput (GB/s)`,
        x-min: 0, x-max: 3,  // dummy scale for batch sizes 16,32,256
        y-min: 0, y-max: 20, // adjust as needed
        x-ticks: [(0, `16`), (1, `32`), (2, `256`)],
        y-tick-step: 2,
        x-grid: true,
        y-grid: true,
        {
            // Convolution line
            plot.add([(0, 8), (1, 12), (2, 15)], color: blue, marker: circle, label: `Convolution`),

            // Depthwise convolution line
            plot.add([(0, 6), (1, 10), (2, 14)], color: red, marker: square, label: `Depthwise Convolution`)
        }
    )
})
