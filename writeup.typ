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
  | small   | 19.018 ms  | 19.058 ms  | 19.738 ms   | 26.163 ms  |
  | medium  | 38.074 ms  | 38.077 ms  | 39.196 ms   | 69.533 ms  |
  | large   | 61.813 ms  | 58.349 ms  | 61.301 ms   | 135.094 ms |
  | xlarge  | 79.546 ms  | 76.364 ms  | 90.199 ms   | OOM        |
  | 2.7b    | 53.105 ms  | 62.066 ms  | 106.359 ms  | OOM        |
  ],
  caption: "Forward Pass Total Time (ms) by Model Size and Sequence Length"
  )

  The timings are quite similar to what was observed with `timeit` (generally within 10%).

+ Kernel that takes the most cumulative time during the forward pass (large model, sequence length = 512):

  `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize128x128x32_warpgroupsize1x1x1_execute_segment
  _k_off_kernel__5x_cublas`

  This is a general matrix-matrix multiplication kernel where the inputs, accumulator, and outputs are all `float32`. The particular kernel is different for different model sizes (different tile sizes, etc.), but it's always a general matrix-matrix multiplication kernel.

  Number of instances: 109

  This is the same kernel as the one that takes the most cumulative time during the backward pass.

+ In general, the non‑matmul kernels that contribute significantly to the forward pass are element‑wise tensor operators—pointwise arithmetic, vectorized element‑wise computations, reductions, and simple data‑movement copies.

  A few specific examples (listed in decreasing order of contribution):

  `void at::native::elementwise_kernel<(int)128, (int)2,
    void at::native::gpu_kernel_impl_nocast<
        at::native::BinaryFunctor<float, float, float,
            at::native::binary_internal::MulFunctor<float>>>(
        at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>
    (int, T3)`

  `void at::native::vectorized_elementwise_kernel<(int)4,
    at::native::BinaryFunctor<float, float, float,
        at::native::binary_internal::MulFunctor<float>>,
    std::array<char *, (unsigned long)3>>
    (int, T2, T3)`

  `void at::native::elementwise_kernel<(int)128, (int)2,
    void at::native::gpu_kernel_impl_nocast<
        at::native::BinaryFunctor<float, float, float,
            at::native::binary_internal::DivFunctor<float>>>(
        at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>
    (int, T3)`

  `void at::native::elementwise_kernel<(int)128, (int)2,
    void at::native::gpu_kernel_impl_nocast<
        at::native::CUDAFunctor_add<float>>(
        at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>
    (int, T3)`


+ With forward‑pass inference only, the four cublas GEMM kernels (all the `sm90_xmma_gemm_*` kernels) add up to \~36 % of the work.

  During a full training step (forward + backward + AdamW update), those kernels take roughly the same amount of time, but the overall kernel increases significantly because of the many vectorised element‑wise AdamW and reduction kernels (the “vectorized_elementwise_kernel” calls and “reduce_kernel” calls). Consequently, GEMMs now represent only \~19 % of the total. In other words, matrix multiplication’s share of runtime is roughly halved, while inexpensive but numerous element‑wise update kernels (mul/add/div/sqrt/fill) and a few extra reductions become the dominant cost.

+ In many cases, the softmax operation takes as long as computing the attention scores and taking the inner products with the value vectors combined (the softmax:matmul ratio within the attention operation varies from \~0.6x to \~1.2x in my experiments).

  This is despite a vastly lower FLOP count (on the order of a 10x difference) for the softmax operation, compared to the matmuls.

  This suggests that the softmax operation consumes significantly more wall time per FLOP, yielding poor utilization due to its memory-bound, control-flow-heavy nature.

== Problem (`mixed_precision_accumulation`): 1 point

We get the most accurate result (10.0001) with both the accumulator and the summands in `float32` (the first loop). With the accumulator in `float32` and the summands in `float16` (the third and fourth loops), we get close (10.0021). With the accumulator in `float16`, though, we a much less accurate result (9.9531). This is because, as the spacing between representable values in `float16` increases, many of the of the late additions round away and the sum stalls.

== Problem (`benchmarking_mixed_precision`): 2 points

+ Model parameters: `float32`
  Output of `fc1`: `float16`
  Output of `ln`: `float32`
  Predicted logits: `float16`
  Loss: `float32`
  Gradients: `float32`

+ The sensitive parts are the mean and variance reductions to compute the layer normalization statistics, and the reciprocal square root computation. The sensitivity is due to the possibility of overflow when the intermediate values are held in the $plus.minus$ 65k range of FP16. BF16 matches the dynamic range of FP32. That removes the overflow risk, and makes it possible to run LayerNorm in BF16.

+ Forward pass timings (sequence length = 512):

  #figure(tablem[
  | *Model*   | *BF16 (ms)* | *FP32 (ms)* |
  |-----------|-------------|-------------|
  | small     | 22.008      | 20.015      |
  | medium    | 43.673      | 56.011      |
  | large     | 67.021      | 130.569     |
  | xl        | 90.293      | 249.101     |
  | 2.7B      | 84.217      | 362.488     |
  ],
  caption: "Forward Pass Timings (sequence length = 512)"
  )

  Backward pass timings (sequence length = 512):

  #figure(tablem[
  | *Model*   | *BF16 (ms)* | *FP32 (ms)* |
  |-----------|-------------|-------------|
  | small     | 27.576      | 41.988      |
  | medium    | 54.394      | 114.903     |
  | large     | 79.397      | 259.485     |
  | xl        | 137.134     | 503.508     |
  | 2.7B      | 149.459     | 716.287     |
  ],
  caption: "Backward Pass Timings (sequence length = 512)"
  )
  
  BF16 is fatster than FP16 for all model sizes. At smaller sizes, the difference in forward pass timings is small, though the difference in backward pass timings is still significant. As the size increases, the difference grows dramatically. This makes intuitive sense, given that matmuls come to dominate the wall clock time when running a forward pass in FP32, and those are the operations for which we get the most benefit from running using mixed precision.


== Problem (`memory_profiling`): 4 points

+ 
  #figure(image("images/mem_2.7b_f_512.png"), caption: "Memory Profile (FP32, 2.7B, forward pass only, 512 sequence length)")

  #figure(image("images/mem_2.7b_fbs_512.png"), caption: "Memory Profile (FP32, 2.7B, full train step, 512 sequence length)")

+ \@TODO

+ \@TODO

+ \@TODO

+ \@TODO