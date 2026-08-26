# Ready-generated Hey Jarvis v1

This directory is the portable standalone C++ snapshot generated from
`models/hey-jarvis-v1/model.tflite`. It is committed so the result can be read
and compiled without running the NN2Prog pipeline first.

```bash
./examples/hey-jarvis-v1/build.sh
./build/hey-jarvis-v1-example
```

The example feeds ten zero-valued feature tensors and prints the raw streaming
score byte. It demonstrates the generated API; it is not an audio frontend or
recognition-quality test.

To prove that the files were generated rather than maintained by hand:

```bash
./examples/hey-jarvis-v1/verify-generated.sh
```

The script regenerates the model from the bundled TFLite file and requires
byte-for-byte equality for the header, constants and implementation.
