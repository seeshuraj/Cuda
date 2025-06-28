
# CUDA Exponential Integral Solver (CPU vs GPU)

## Overview
This project computes the exponential integral Ei(n, x) using both CPU and GPU implementations. It strictly follows the CPU algorithm as provided and ports it directly to GPU using CUDA.

## Files Included
- `main.cpp`: Handles command-line parsing and orchestrates CPU/GPU calls.
- `cpu_integral.cpp`, `cpu_integral.h`: CPU implementation of the exponential integral.
- `gpu_integral.cu`, `gpu_integral.h`: GPU kernel for exponential integral using the same method as the CPU.
- `gpu_launcher.cu`: Launches CUDA kernels for float and double precision.
- `Makefile`: Builds the executable `ei_exec`.

## Compilation
To build the project, run:
```bash
make clean
make
```

## Execution
To run on CPU:
```bash
./ei_exec -c -n 100 -m 100
```

To run on GPU:
```bash
./ei_exec -g -n 100 -m 100
```

## Key Note
- The GPU implementation now **faithfully replicates** the CPU logic.
- This resolves previous feedback which flagged the use of trapezoidal or alternative methods.

## Result Sample
```
CPU float time: 6.64e-05s
GPU float time: 0.328178s
```

