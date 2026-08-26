#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
mkdir -p "$root/build"
"${CXX:-g++}" -std=c++17 -O2 -DNDEBUG -I"$here" \
  "$here/example.cpp" "$here/model.cpp" -o "$root/build/hey-jarvis-v1-example"
echo "ready: $root/build/hey-jarvis-v1-example"
