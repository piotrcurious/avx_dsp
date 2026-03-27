# avx_dsp

A high-performance AVX/FMA-optimized DSP library for Linux, providing essential signal processing primitives and real-time visualization.

## Features

- **Optimized DSP Kernels**:
  - Dot Product (Single vector & large arrays)
  - Convolution (Linear & Cyclic)
  - FFT (Vectorized 4-point & Radix-2 $O(N \log N)$ for arrays)
  - Windowing Functions (Hann, Hamming)
  - FIR Filtering (Vectorized)
  - Complex Vector Math (Multiplication, Magnitude, Phase)
- **Memory Management**:
  - Aligned memory allocation (`avx_malloc`/`avx_free`) for optimal SIMD performance.
- **Real-time Visualization**:
  - OpenGL-based FFT Waterfall display.
  - Phase-sensitive Waterfall display (Phase mapped to Hue).
- **Comprehensive Testing**:
  - Python-based test signal generator for large datasets.
  - C test runner for accuracy verification.
  - Benchmarking tools to compare AVX vs. scalar performance.

## Prerequisites

To build and run the demos, you will need the following libraries:

```bash
sudo apt-get install -y libfltk1.3-dev libgl1-mesa-dev libglu1-mesa-dev libasound2-dev
```

## Building

The project uses a standard Makefile. Available targets include:

- `make lib`: Builds the shared library `libavx_dsp.so`.
- `make demo`: Builds the live FFT waterfall GUI.
- `make phase_demo`: Builds the phase-sensitive waterfall GUI.
- `make tests_large`: Generates test data and runs correctness verification.
- `make benchmark`: Runs performance benchmarks.
- `make clean`: Removes build artifacts and temporary files.

## Usage

To run any of the binaries, ensure the current directory is in your `LD_LIBRARY_PATH`:

```bash
LD_LIBRARY_PATH=. ./gui_demo
```

For headless verification of the GUI demos:
```bash
HEADLESS=1 LD_LIBRARY_PATH=. ./gui_demo
```

## Project Structure

- `avx_dsp.c/h`: Core library implementation and header.
- `audio_engine.c`: ALSA-based real-time audio capture.
- `gui_demo.cpp`: Standard waterfall visualization.
- `gui_phase_demo.cpp`: Phase-sensitive waterfall visualization.
- `generate_data.py`: Python script for golden reference signal generation.
- `tests_large.c`: Integration test suite.
- `benchmark.c`: Performance measurement tool.

---
Original interface by Bing AI. Extensively improved and expanded for functionality, performance, and visualization.
