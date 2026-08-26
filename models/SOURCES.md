# Bundled model sources

- `hey-jarvis-v1`: `models/hey_jarvis.tflite` from
  `esphome/micro-wake-word-models` revision
  `05b65922cc433c9df13e98e32a7fe520758c837e`.
- `hey-jarvis-v2`: `models/v2/hey_jarvis.tflite` from the same revision.
- `mlperf-kws`: provenance and license are recorded inside its model directory.

Each `model.sha256` locks the original file identity. Graph structure, tensor
dimensions, quantization, weights and generated API behavior are derived from
the original TFLite file during every generation.
