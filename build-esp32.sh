#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

model="models/mlperf-kws"
target="${1:-esp32}"
case "$target" in esp32|esp32s3) ;; *) echo "usage: $0 [esp32|esp32s3]" >&2; exit 2 ;; esac
output="generated/mlperf-kws/$target"

# Complete model.tflite -> IR -> target C++ pipeline.
(cd "$model" && sha256sum --check model.sha256)
mkdir -p "$output"
python3 nn2prog/import_tflite.py "$model/model.tflite" \
  "$output/model.ir.json" "$output/model.constants.h"
python3 nn2prog/compiler.py --model-dir="$output" --target="$target"

# The firmware links only the generated C++, not TFLite Micro.
pio run --project-dir examples/esp32-kws -e "$target"
