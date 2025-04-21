#import "@preview/ilm:1.4.1": *
#import "@preview/tablem:0.2.0": *

#let three-line-table = tablem.with(
  render: (columns: auto, ..args) => {
    table(
      columns: columns,
      stroke: none,
      align: center + horizon,
      table.hline(y: 0),
      table.hline(y: 1, stroke: .5pt),
      ..args,
      table.hline(),
    )
  }
)

#set text(lang: "en")

#show: ilm.with(
  title: [CS 336: Assignment 2],
  author: "Brandon Snider",
  date: datetime(year: 2025, month: 04, day: 29),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
)

#set enum(numbering: "a)")
#set heading(numbering: none)
#show link: underline

= 1. Profiling and Benchmarking

== Problem (`benchmarking_script`): 4 points

+ See `cs336_systems/benchmark.py`

+ Benchmarking results on CUDA (5 warmup steps):
  
  #figure(tablem[
  | *Size*  | *Forward ($mu$)* | *Forward ($sigma$)* | *Backward ($mu$)* | *Backward ($sigma$)* |
  |---------|------------------|---------------------|-------------------|----------------------|
  | small   | 21.7 ms          | 0.0 ms              | 42.0 ms           | 0.1 ms               |
  | medium  | 57.8 ms          | 0.1 ms              | 115.5 ms          | 0.1 ms               |
  | large   | 127.6 ms         | 0.1 ms              | 264.9 ms          | 0.2 ms               |
  | xl      | 252.8 ms         | 0.6 ms              | 503.6 ms          | 0.3 ms               |
  | 2.7B    | 340.3 ms         | 0.1 ms              | 741.2 ms          | 0.1 ms               |
  ],
  caption: "CUDA Benchmarking Results"
  )

  There is little variation across measurements, as seen by the small standard deviations.

+ Benchmarking results on CUDA (0 warmup steps):

  #figure(tablem[
  | *Size*  | *Forward ($mu$)* | *Forward ($sigma$)* | *Backward ($mu$)* | *Backward ($sigma$)* |
  |---------|------------------|---------------------|-------------------|----------------------|
  | small   | 61.7 ms          | 121.5 ms            | 49.5 ms           | 23.2 ms              |
  | medium  | 60.4 ms          | 8.5 ms              | 115.5 ms          | 0.1 ms               |
  | large   | 131.0 ms         | 10.4 ms             | 265.0 ms          | 0.5 ms               |
  | xl      | 255.8 ms         | 9.2 ms              | 503.5 ms          | 1.0 ms               |
  | 2.7B    | 345.0 ms         | 13.9 ms             | 741.0 ms          | 0.6 ms               |
  ],
  caption: "CUDA Benchmarking Results (no warmup)"
  )

  Without warmup, the standard deviations are much larger, presumably due to the overhead of launching kernels in the first few iterations. Additionally, the backward pass is reported as being faster than the forward pass for the small model, presumably because the forward pass incurs the initial system overheads (like kernel launching, JIT compilation, and initial memory allocation) which are partially amortized by the time the backward pass begins.

  Benchmarking results on Cuda (1 warmup step):

  #figure(tablem[
  | *Size*  | *Forward ($mu$)* | *Forward ($sigma$)* | *Backward ($mu$)* | *Backward ($sigma$)* |
  |---------|------------------|---------------------|-------------------|----------------------|
  | small   | 21.6 ms          | 0.1 ms              | 41.7 ms           | 0.1 ms               |
  | medium  | 57.5 ms          | 0.0 ms              | 115.1 ms          | 0.1 ms               |
  | large   | 127.4 ms         | 0.2 ms              | 264.4 ms          | 0.2 ms               |
  | xl      | 252.2 ms         | 0.2 ms              | 503.1 ms          | 0.9 ms               |
  | 2.7B    | 340.3 ms         | 0.5 ms              | 740.8 ms          | 0.2 ms               |
  ],
  caption: "CUDA Benchmarking Results (1 warmup step)"
  )

  In my case, the results with a single warmup step were almost identical to the results with 5 warmup steps. However, one could imagine that benchmarking after insufficient warmup steps could still yield different results due to incomplete JIT compilation, cache population, or other initialization overheads that haven't yet stabilized.
