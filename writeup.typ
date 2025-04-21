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

+ See `cs336_systems/benchmark.py` and `cs336_systems/benchmark.sh`

+ Benchmarking results (CUDA, *5 warmup steps*, 10 steps, varying sequence length):
  
  #figure(tablem[
  | *Model*   | *Forward ($mu plus.minus sigma$)* | *Backward ($mu plus.minus sigma$)* | *Total ($mu plus.minus sigma$)* |
  |-----------|----------------------------------|-----------------------------------|----------------------------------|
  | small     | 15.839 ± 0.974 ms                 | 15.571 ± 0.080 ms                  | 31.411 ± 1.044 ms                 |
  | medium    | 30.328 ± 0.161 ms                 | 30.963 ± 0.067 ms                  | 61.291 ± 0.191 ms                 |
  | large     | 45.521 ± 0.777 ms                 | 46.152 ± 0.274 ms                  | 91.673 ± 1.043 ms                 |
  | xl        | 60.960 ± 0.977 ms                 | 68.650 ± 0.038 ms                  | 129.610 ± 0.997 ms                |
  | 2.7B      | 42.300 ± 0.587 ms                 | 86.600 ± 0.066 ms                  | 128.900 ± 0.607 ms                |
  ],
  caption: "Benchmarking Results (sequence length = 128)"
  )

  #figure(tablem[
  | *Model*   | *Forward ($mu plus.minus sigma$)* | *Backward ($mu plus.minus sigma$)* | *Total ($mu plus.minus sigma$)* |
  |-----------|----------------------------------|-----------------------------------|----------------------------------|
  | small     | 15.356 ± 0.062 ms                 | 16.091 ± 0.098 ms                  | 31.447 ± 0.140 ms                 |
  | medium    | 30.226 ± 0.072 ms                 | 31.871 ± 0.100 ms                  | 62.098 ± 0.156 ms                 |
  | large     | 45.633 ± 0.211 ms                 | 61.766 ± 0.146 ms                  | 107.399 ± 0.285 ms                |
  | xl        | 62.196 ± 0.671 ms                 | 107.093 ± 0.215 ms                 | 169.289 ± 0.602 ms                |
  | 2.7B      | 45.968 ± 0.122 ms                 | 132.788 ± 0.162 ms                 | 178.756 ± 0.274 ms                |
  ],
  caption: "Benchmarking Results (sequence length = 256)"
  )

  #figure(tablem[
  | *Model*   | *Forward ($mu plus.minus sigma$)* | *Backward ($mu plus.minus sigma$)* | *Total ($mu plus.minus sigma$)* |
  |-----------|----------------------------------|-----------------------------------|----------------------------------|
  | small     | 15.943 ± 0.266 ms                 | 22.940 ± 0.028 ms                  | 38.883 ± 0.277 ms                 |
  | medium    | 32.684 ± 0.923 ms                 | 57.238 ± 0.172 ms                  | 89.922 ± 1.089 ms                 |
  | large     | 49.374 ± 0.525 ms                 | 116.740 ± 0.074 ms                 | 166.114 ± 0.503 ms                |
  | xl        | 81.372 ± 0.152 ms                 | 200.596 ± 0.230 ms                 | 281.968 ± 0.327 ms                |
  | 2.7B      | 89.902 ± 0.273 ms                 | 236.174 ± 0.117 ms                 | 326.076 ± 0.304 ms                |
  ],
  caption: "Benchmarking Results (sequence length = 512)"
  )

  #figure(tablem[
  | *Model*   | *Forward ($mu plus.minus sigma$)* | *Backward ($mu plus.minus sigma$)* | *Total ($mu plus.minus sigma$)* |
  |-----------|----------------------------------|-----------------------------------|----------------------------------|
  | small     | 23.971 ± 0.019 ms                 | 52.898 ± 0.013 ms                  | 76.869 ± 0.023 ms                 |
  | medium    | 62.225 ± 0.308 ms                 | 137.851 ± 0.240 ms                 | 200.076 ± 0.409 ms                |
  | large     | 118.398 ± 0.123 ms                | 273.568 ± 0.418 ms                 | 391.965 ± 0.399 ms                |
  | xl        | OOM                               | OOM                                | OOM                               |
  | 2.7B      | OOM                               | OOM                                | OOM                               |
  ],
  caption: "Benchmarking Results (sequence length = 1024)"
  )

  There is little variation across measurements, as seen by the small standard deviations (generally well under 1 ms).

+ Benchmarking results (CUDA, *0 warmup steps*, 10 steps, varying sequence length):

  #figure(tablem[
  | *Model* | *Forward ($mu plus.minus sigma$)* | *Backward ($mu plus.minus sigma$)* | *Total ($mu plus.minus sigma$)* |
  |---------|----------------------------------|-----------------------------------|----------------------------------|
  | small   | 50.044 ± 109.166 ms              | 24.972 ± 29.328 ms                | 75.016 ± 138.493 ms              |
  | medium  | 71.781 ± 128.984 ms              | 40.520 ± 30.044 ms                | 112.301 ± 159.028 ms             |
  | large   | 84.114 ± 122.191 ms              | 59.346 ± 37.632 ms                | 143.460 ± 159.822 ms             |
  | xl      | 101.616 ± 125.648 ms             | 80.611 ± 38.144 ms                | 182.228 ± 163.791 ms             |
  | 2.7B    | 79.821 ± 122.315 ms              | 94.699 ± 25.789 ms                | 174.521 ± 148.104 ms             |
  ],
  caption: "CUDA Benchmarking Results (no warmup, sequence length = 128)"
  )

  #figure(tablem[
  | *Model* | *Forward ($mu plus.minus sigma$)* | *Backward ($mu plus.minus sigma$)* | *Total ($mu plus.minus sigma$)* |
  |---------|----------------------------------|-----------------------------------|----------------------------------|
  | small   | 54.248 ± 122.671 ms              | 27.946 ± 38.036 ms                | 82.194 ± 160.707 ms              |
  | medium  | 69.122 ± 121.881 ms              | 41.867 ± 30.631 ms                | 110.989 ± 152.511 ms             |
  | large   | 87.265 ± 129.787 ms              | 73.268 ± 34.331 ms                | 160.534 ± 164.111 ms             |
  | xl      | 108.917 ± 140.552 ms             | 116.053 ± 27.636 ms               | 224.971 ± 168.187 ms             |
  | 2.7B    | 85.747 ± 126.751 ms              | 140.738 ± 26.156 ms               | 226.485 ± 152.906 ms             |
  ],
  caption: "CUDA Benchmarking Results (no warmup, sequence length = 256)"
  )

  #figure(tablem[
  | *Model* | *Forward ($mu plus.minus sigma$)* | *Backward ($mu plus.minus sigma$)* | *Total ($mu plus.minus sigma$)* |
  |---------|----------------------------------|-----------------------------------|----------------------------------|
  | small   | 54.190 ± 119.762 ms              | 33.965 ± 35.434 ms                | 88.154 ± 155.196 ms              |
  | medium  | 71.163 ± 122.827 ms              | 65.110 ± 26.234 ms                | 136.273 ± 149.061 ms             |
  | large   | 90.134 ± 126.374 ms              | 125.395 ± 28.581 ms               | 215.528 ± 154.953 ms             |
  | xl      | 123.206 ± 130.294 ms             | 210.528 ± 28.432 ms               | 333.734 ± 158.725 ms             |
  | 2.7B    | 124.921 ± 111.795 ms             | 244.471 ± 28.501 ms               | 369.392 ± 140.296 ms             |
  ],
  caption: "CUDA Benchmarking Results (no warmup, sequence length = 512)"
  )

  #figure(tablem[
  | *Model* | *Forward ($mu plus.minus sigma$)* | *Backward ($mu plus.minus sigma$)* | *Total ($mu plus.minus sigma$)* |
  |---------|----------------------------------|-----------------------------------|----------------------------------|
  | small   | 60.625 ± 115.953 ms              | 61.123 ± 26.407 ms                | 121.748 ± 142.360 ms             |
  | medium  | 100.052 ± 119.305 ms             | 146.729 ± 27.650 ms               | 246.781 ± 146.955 ms             |
  | large   | 155.401 ± 116.441 ms             | 281.863 ± 25.416 ms               | 437.264 ± 141.853 ms             |
  | xl      | OOM                              | OOM                               | OOM                              |
  | 2.7B    | OOM                              | OOM                               | OOM                              |
  ],
  caption: "CUDA Benchmarking Results (no warmup, sequence length = 1024)"
  )

  Without warmup, the standard deviations are much larger, likely because the initial steps incur one-time overheads such as CUDA kernel loading and memory allocation for tensors like activations and gradients. Once these setup costs are paid and the stead-state throughput is reached, subsequent steps exhibit much less variability.

  Benchmarking results (CUDA, *1 warmup step*, 10 steps, varying sequence length):

  #figure(tablem[
  | *Model* | *Forward ($mu plus.minus sigma$)* | *Backward ($mu plus.minus sigma$)* | *Total ($mu plus.minus sigma$)* |
  |---------|----------------------------------|-----------------------------------|----------------------------------|
  | small   | 15.457 ± 0.097 ms                | 15.801 ± 0.130 ms                 | 31.258 ± 0.212 ms                |
  | medium  | 30.379 ± 0.127 ms                | 31.172 ± 0.182 ms                 | 61.551 ± 0.298 ms                |
  | large   | 46.200 ± 0.706 ms                | 47.573 ± 0.253 ms                 | 93.772 ± 0.809 ms                |
  | xl      | 61.188 ± 1.109 ms                | 69.504 ± 0.060 ms                 | 130.692 ± 1.120 ms               |
  | 2.7B    | 41.377 ± 0.642 ms                | 86.992 ± 0.140 ms                 | 128.370 ± 0.710 ms               |
  ],
  caption: "CUDA Benchmarking Results (1 warmup step, sequence length = 128)"
  )

  #figure(tablem[
  | *Model* | *Forward ($mu plus.minus sigma$)* | *Backward ($mu plus.minus sigma$)* | *Total ($mu plus.minus sigma$)* |
  |---------|----------------------------------|-----------------------------------|----------------------------------|
  | small   | 15.383 ± 0.151 ms                | 16.121 ± 0.145 ms                 | 31.503 ± 0.286 ms                |
  | medium  | 31.065 ± 0.880 ms                | 31.977 ± 0.267 ms                 | 63.042 ± 1.110 ms                |
  | large   | 46.086 ± 0.400 ms                | 62.285 ± 0.047 ms                 | 108.371 ± 0.404 ms               |
  | xl      | 63.262 ± 1.544 ms                | 108.228 ± 0.181 ms                | 171.490 ± 1.652 ms               |
  | 2.7B    | 45.812 ± 0.222 ms                | 132.597 ± 0.515 ms                | 178.410 ± 0.539 ms               |
  ],
  caption: "CUDA Benchmarking Results (1 warmup step, sequence length = 256)"
  )

  #figure(tablem[
  | *Model* | *Forward ($mu plus.minus sigma$)* | *Backward ($mu plus.minus sigma$)* | *Total ($mu plus.minus sigma$)* |
  |---------|----------------------------------|-----------------------------------|----------------------------------|
  | small   | 16.191 ± 0.861 ms                | 22.714 ± 0.094 ms                 | 38.905 ± 0.945 ms                |
  | medium  | 31.881 ± 1.170 ms                | 57.215 ± 0.050 ms                 | 89.096 ± 1.204 ms                |
  | large   | 49.574 ± 0.717 ms                | 116.698 ± 0.148 ms                | 166.273 ± 0.802 ms               |
  | xl      | 81.754 ± 0.239 ms                | 201.487 ± 0.385 ms                | 283.241 ± 0.429 ms               |
  | 2.7B    | 89.518 ± 0.112 ms                | 235.488 ± 0.240 ms                | 325.006 ± 0.304 ms               |
  ],
  caption: "CUDA Benchmarking Results (1 warmup step, sequence length = 512)"
  )

  #figure(tablem[
  | *Model* | *Forward ($mu plus.minus sigma$)* | *Backward ($mu plus.minus sigma$)* | *Total ($mu plus.minus sigma$)* |
  |---------|----------------------------------|-----------------------------------|----------------------------------|
  | small   | 23.848 ± 0.035 ms                | 52.626 ± 0.044 ms                 | 76.474 ± 0.054 ms                |
  | medium  | 62.023 ± 0.067 ms                | 137.619 ± 0.159 ms                | 199.642 ± 0.175 ms               |
  | large   | 118.616 ± 0.044 ms               | 273.708 ± 0.310 ms                | 392.324 ± 0.301 ms               |
  | xl      | OOM                              | OOM                               | OOM                              |
  | 2.7B    | OOM                              | OOM                               | OOM                              |
  ],
  caption: "CUDA Benchmarking Results (1 warmup step, sequence length = 1024)"
  )

  Even with a single warmup step, the variance is noticeably higher than with five warmup steps. This suggests that one iteration may not be sufficient to complete all initialization processes, such as loading all necessary CUDA kernels or stabilizing memory allocation patterns. Subsequent steps might still encounter some initial overheads until a true steady state is reached, which appears to take a few iterations.
  
== Problem  (`nsys_profile`): 5 points

+ Mean total forward pass time; all sizes, all sequence lengths:

  #figure(tablem[
  | *Model* | *128*      | *256*      | *512*       | *1024*     |
  |---------|------------|------------|-------------|------------|
  | small   | 19.805 ms  | 20.595 ms  | 19.575 ms   | 44.577 ms  |
  | medium  | 39.319 ms  | 38.625 ms  | 56.818 ms   | 126.23 ms  |
  | large   | 58.511 ms  | 58.284 ms  | 127.522 ms  | OOM        |
  | xlarge  | 79.637 ms  | 127.036 ms | 251.931 ms  | OOM        |
  | 2.7b    | 87.422 ms  | 173.87 ms  | 339.198 ms  | OOM        |
  ],
  caption: "Forward Pass Total Time (ms) by Model Size and Sequence Length"
  )

  The `nsys` timings are quite significantly different from the timing using `timeit`, particularly for larger models and longer sequence lengths. In general, the `nsys` timings are significantly larger, sometimes by as much as 4-5x.