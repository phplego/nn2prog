# NN2Prog

NN2Prog compiles supported quantized TFLite models into standalone,
deterministic C++17. The output contains the model arithmetic and constants but
does not link TensorFlow, TFLite Runtime or TFLite Micro.

## Scope

NN2Prog operates at the tensor-model boundary. Its generated API accepts the
same logical input tensor as the source TFLite model and returns the model
result. This keeps the generated code independent of data sources and
application frameworks:

```text
platform input -> domain preprocessing -> generated Model -> application policy
```

For a wake-word application, microphone capture and PCM-to-feature encoding
happen before the generated model; score aggregation and the detection policy
happen after it. The same API boundary works for audio, image, text and other
models.

## Results at a glance

MLPerf Tiny KWS, identical int8 inputs and top-1 decision semantics:

| Target | Reference | Reference latency | NN2Prog latency | Speedup | Working memory | Program size |
|---|---|---:|---:|---:|---:|---:|
| ESP32-D0WD-V3, 240 MHz | TFLite Micro + ESP-NN | 161.72 ms | **135.37 ms** | **1.195x** | 22,780 → **16,064 B** | flash 287,039 → **187,799 B** |
| ESP32-S3, 240 MHz | TFLite Micro + ESP-NN | 17.902 ms | **15.678 ms** | **1.142x** | 35,500 → **16,704 B** | flash 322,919 → **206,835 B** |
| Intel i7-11390H, GCC 13.3 `-O3` | TFLite Micro reference kernels | 7.432 ms | **2.383 ms** | **3.12x** | 24,000 → **16,064 B** | stripped executable 125,840 → **43,136 B** |

The classic ESP32 result shows 34.6% less flash, 29.5%
less engine working memory and 1.195x higher model-only throughput than TFLite
Micro with ESP-NN. The larger x86 speedup is included as a portability result,
but its baseline uses TFLite Micro's portable reference kernels and should not
be interpreted as a comparison with an optimized x86 inference engine.

All comparisons are model-only benchmarks, excluding audio capture and feature
extraction. NN2Prog produced zero top-1 mismatches against TFLite Micro on 1,024
deterministic full-range int8 inputs; the ESP32 run additionally passed the same
golden checksum in all three 64-inference hardware trials.
The S3 comparison uses ESP-IDF 5.5.3 and six trials across two boots per engine.
Working memory means tensor-arena use versus generated working buffers, not total
RAM. For KWS, TFLite Micro computes softmax while NN2Prog returns the same top-1 decision.

The project is intentionally small: Python's standard library is enough to
generate code, and a normal C++17 compiler is enough to test it.

## Quick check

Requirements: Python 3.8+, Bash and `g++` (or set `CXX`).

```bash
./test.sh
```

This regenerates and tests MLPerf Tiny KWS and two Hey Jarvis models without
network access. A successful run reports `PASS` for MLPerf KWS and zero
mismatches for both streaming models.

## Ready-generated example

Hey Jarvis v1 is also committed as ordinary portable tensor-level C++ so it can
be inspected and built without first running the generator:

```bash
./examples/hey-jarvis-v1/build.sh
./build/hey-jarvis-v1-example
```

The published snapshot is tested against the same 4,000-frame golden stream as
freshly generated code. Its byte-for-byte provenance can be checked separately:

```bash
./examples/hey-jarvis-v1/verify-generated.sh
```

This example demonstrates the model API: it feeds prepared feature tensors,
prints the score and exits. Audio capture and feature extraction are supplied
by the application that embeds the generated model.

## Visible compilation pipeline

Each build or verification script calls the NN2Prog tools directly, so one file
shows the complete process and every argument. For example, `build-esp32.sh`
performs:

```bash
python3 nn2prog/import_tflite.py models/mlperf-kws/model.tflite \
  generated/mlperf-kws/esp32/model.ir.json \
  generated/mlperf-kws/esp32/model.constants.h

python3 nn2prog/compiler.py \
  --model-dir=generated/mlperf-kws/esp32 \
  --target=esp32

pio run --project-dir examples/esp32-kws
```

The compiler targets are `portable`, `x86-avx2`, `esp32`, and `esp32s3`. Outputs include:

```text
model.ir.json
model.constants.h
model.h
model.cpp
```

The `.tflite` file is always the sole source of graph structure, tensor shapes,
quantization and weights. A standard `model.sha256` file locks source identity.

Every generated model exposes the same small API:

```cpp
nn2prog::generated::Model model;
auto result = model.invoke(input);
```

The compiler automatically derives tensor lifetimes, reuses typed buffers and
selects applicable target kernels. A scalar model returns its output byte; a
classifier ending in softmax returns the top-1 class. No class name or output
mode has to be configured by hand.

## ESP32

Install PlatformIO, then run:

```bash
./build-esp32.sh
./flash-esp32.sh /dev/ttyUSB0
```

For ESP32-S3, use `./build-esp32.sh esp32s3` and
`./flash-esp32.sh /dev/ttyACM0 esp32s3`.

PlatformIO downloads the ESP-IDF toolchain on its first build. The firmware
itself uses only generated C++; it does not download or link a TFLite engine.
The serial benchmark validates a deterministic checksum and prints cycle and
memory measurements.

## Repository map

```text
nn2prog/       executable importer, analyzer and C++ compiler
  _internal/   shared library code used by those commands
models/        immutable source models plus provenance
tests/         dependency-free golden protocols and checks
examples/      ready-generated C++ and thin target integrations
generated/     disposable build output (ignored)
```

See [METHODOLOGY.md](METHODOLOGY.md) for design, reproducibility and validation
details.

## Related work

[CustomDLCoder](https://doi.org/10.1145/3650212.3652119) demonstrated compiling
TFLite models into specialized C++ programs by extracting and configuring
TFLite backend computing units. NN2Prog instead focuses on exact,
dependency-light code generation for microcontrollers and provides a foundation
for future synthesis of cheaper representations from weights and statically
derived activation ranges.

The compiler and examples are Apache-2.0 licensed. Bundled model provenance and
upstream license copies are kept in `models/`.
