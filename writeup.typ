#import "@preview/ilm:1.4.1": *
#import "@preview/tablem:0.2.0": *

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

+ On CUDA:

  #tablem[
  | Size | Forward ($mu$) | Forward ($sigma$) | Backward ($mu$) | Backward ($sigma$) |
  |------|---------|----------|-------|-------|
  | small | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
  | medium | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
  | large | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
  | xl | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
  | 2.7B | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
  ]
