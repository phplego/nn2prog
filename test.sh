#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

generate() {
  local model_id="$1"
  local target="${2:-portable}"
  local model="models/$model_id"
  local output="generated/$model_id/$target"
  (cd "$model" && sha256sum --check model.sha256)
  mkdir -p "$output"
  python3 nn2prog/import_tflite.py "$model/model.tflite" \
    "$output/model.ir.json" "$output/model.constants.h"
  python3 nn2prog/compiler.py --model-dir="$output" --target="$target"
}

generate mlperf-kws
generate hey-jarvis-v1
generate hey-jarvis-v2
generate mlperf-kws esp32s3
mkdir -p build
cxx="${CXX:-g++}"
flags=(-std=c++17 -O2 -DNDEBUG)

cmp examples/hey-jarvis-v1/model.h generated/hey-jarvis-v1/portable/model.h
cmp examples/hey-jarvis-v1/model.constants.h generated/hey-jarvis-v1/portable/model.constants.h
cmp examples/hey-jarvis-v1/model.cpp generated/hey-jarvis-v1/portable/model.cpp

"$cxx" "${flags[@]}" -Igenerated/mlperf-kws/portable -Itests/mlperf-kws \
  tests/verify_mlperf.cpp generated/mlperf-kws/portable/model.cpp \
  -o build/test-mlperf-kws

"$cxx" "${flags[@]}" -Igenerated/hey-jarvis-v1/portable \
  -DNN2PROG_FEATURE_COUNT=40 \
  tests/verify_streaming.cpp generated/hey-jarvis-v1/portable/model.cpp \
  -o build/test-hey-jarvis-v1

"$cxx" "${flags[@]}" -Igenerated/hey-jarvis-v2/portable \
  -DNN2PROG_FEATURE_COUNT=120 \
  tests/verify_streaming.cpp generated/hey-jarvis-v2/portable/model.cpp \
  -o build/test-hey-jarvis-v2

"$cxx" "${flags[@]}" -Iexamples/hey-jarvis-v1 \
  -DNN2PROG_FEATURE_COUNT=40 \
  tests/verify_streaming.cpp examples/hey-jarvis-v1/model.cpp \
  -o build/test-published-hey-jarvis-v1

./build/test-mlperf-kws
./build/test-hey-jarvis-v1 tests/hey-jarvis-v1/features.bin tests/hey-jarvis-v1/expected.bin
./build/test-hey-jarvis-v2 tests/hey-jarvis-v2/features.bin tests/hey-jarvis-v2/expected.bin
./build/test-published-hey-jarvis-v1 tests/hey-jarvis-v1/features.bin tests/hey-jarvis-v1/expected.bin

# Host checks exercise the S3 layout and exact arithmetic, not Xtensa instructions.
"$cxx" "${flags[@]}" -Igenerated/mlperf-kws/esp32s3 -Itests/mlperf-kws \
  tests/verify_mlperf.cpp generated/mlperf-kws/esp32s3/model.cpp -o build/test-s3-kws
./build/test-s3-kws
"$cxx" "${flags[@]}" -Igenerated/mlperf-kws/esp32s3 \
  tests/verify_s3.cpp -o build/test-s3-arithmetic
./build/test-s3-arithmetic
