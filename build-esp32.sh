#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

model="models/mlperf-kws"
output="generated/mlperf-kws/esp32"

# Complete model.tflite -> IR -> target C++ pipeline.
(cd "$model" && sha256sum --check model.sha256)
mkdir -p "$output"
python3 nn2prog/import_tflite.py "$model/model.tflite" \
  "$output/model.ir.json" "$output/model.constants.h"
python3 nn2prog/compiler.py --model-dir="$output" --target=esp32

# The firmware links only the generated C++, not TFLite Micro.
pio run --project-dir examples/esp32-kws
