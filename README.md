# Milestone 1 — Canny Sequential Baseline

Files:
- `milestone1_report.tex` — IEEE-style LaTeX report
- `canny_baseline.c` — single-threaded C reference implementation
- `README.md` — build/run instructions

## Build

macOS / clang:

```bash
clang -std=c11 -O0 -Wall -Wextra -o canny_baseline canny_baseline.c -lm
```

Optimized CPU comparison:

```bash
clang -std=c11 -O2 -Wall -Wextra -o canny_baseline_O2 canny_baseline.c -lm
```

## Run

```bash
./canny_baseline input.ppm output.pgm 50 100
```

The final two arguments are optional:
- low threshold: default 50
- high threshold: default 100

Input must be binary PPM (P6), 8-bit (`maxval=255`).
Output is binary PGM (P5).

## Benchmark

Suggested resolutions, matching the reference paper:

- 40x30
- 640x480
- 1280x720
- 1920x1080
- 3840x2160

Run each configuration multiple times and average the reported stage
and total timings. Keep `-O0` and `-O2` results separate.

## Important implementation choices

- Gaussian kernel normalization is 1/150, matching Equation I of the paper.
- Gaussian and Sobel convolutions use zero padding.
- NMS uses standard four-direction quantization (0/45/90/135 degrees).
- Hysteresis uses 8-connected stack-based propagation from strong pixels.
- The paper does not specify numerical low/high thresholds in the methodology;
  50/100 are configurable baseline values.
- I/O is excluded from the algorithm timing.

## LaTeX

Compile with:

```bash
pdflatex milestone1_report.tex
pdflatex milestone1_report.tex
```

Fill in your name/university/email and replace the `--` benchmark cells with
your measured results.
