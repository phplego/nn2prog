#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"

model="$root/models/hey-jarvis-v1"
generated="$root/generated/hey-jarvis-v1/portable"

(cd "$model" && sha256sum --check model.sha256)
mkdir -p "$generated"
python3 "$root/nn2prog/import_tflite.py" "$model/model.tflite" \
  "$generated/model.ir.json" "$generated/model.constants.h"
python3 "$root/nn2prog/compiler.py" \
  --model-dir=generated/hey-jarvis-v1/portable --target=portable

cmp "$here/model.h" "$generated/model.h"
cmp "$here/model.constants.h" "$generated/model.constants.h"
cmp "$here/model.cpp" "$generated/model.cpp"
echo "PASS: published Hey Jarvis v1 C++ is exactly reproducible"
