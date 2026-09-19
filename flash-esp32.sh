#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
port="${1:-/dev/ttyUSB0}"
target="${2:-esp32}"
case "$target" in esp32|esp32s3) ;; *) echo "usage: $0 [PORT] [esp32|esp32s3]" >&2; exit 2 ;; esac

model="models/mlperf-kws"
output="generated/mlperf-kws/$target"

# Regenerate independently so flashing never relies on another wrapper script.
(cd "$model" && sha256sum --check model.sha256)
mkdir -p "$output"
python3 nn2prog/import_tflite.py "$model/model.tflite" \
  "$output/model.ir.json" "$output/model.constants.h"
python3 nn2prog/compiler.py --model-dir="$output" --target="$target"

pio run --project-dir examples/esp32-kws -e "$target" --target upload --upload-port "$port"
pio device monitor --port "$port" --baud 115200
