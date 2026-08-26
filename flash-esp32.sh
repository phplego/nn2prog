#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
port="${1:-/dev/ttyUSB0}"

model="models/mlperf-kws"
output="generated/mlperf-kws/esp32"

# Regenerate independently so flashing never relies on another wrapper script.
(cd "$model" && sha256sum --check model.sha256)
mkdir -p "$output"
python3 nn2prog/import_tflite.py "$model/model.tflite" \
  "$output/model.ir.json" "$output/model.constants.h"
python3 nn2prog/compiler.py --model-dir="$output" --target=esp32

pio run --project-dir examples/esp32-kws --target upload --upload-port "$port"
pio device monitor --port "$port" --baud 115200
