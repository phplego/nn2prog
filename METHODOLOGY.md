# Design and validation

## Compilation model

NN2Prog translates a supported quantized TFLite graph into standalone C++17.
Compilation has two inputs:

- the source `.tflite` model;
- the target backend: `portable`, `x86-avx2`, or `esp32`.

Together, the graph, constant tensors, quantization parameters and selected
target fully determine the generated program.

## API boundary

The generated `Model` accepts the same logical input tensor as the source model.
Acquisition and domain-specific preprocessing precede this call; application
policy follows it:

```text
platform input -> domain preprocessing -> generated Model -> application policy
```

For example, an ESP32 wake-word application obtains PCM through its chosen I²S
driver, converts PCM to the feature tensor expected by the model, calls
`Model::invoke()`, and applies its detection policy to the returned score. These
components can vary without changing NN2Prog or the generated model.

## Pipeline

1. The standard-library-only FlatBuffer importer reads the `.tflite` file.
2. It emits a JSON intermediate representation and a C++ constant table.
3. Static analysis determines tensor lifetimes, safe aliases and applicable
   target kernels.
4. These decisions form an immutable lowered program, independent of C++ text
   generation.
5. The emitter translates that program into a model header and standalone C++
   implementation.
6. Golden tests compile and execute the result.

The generated implementation links no TensorFlow, TFLite Runtime, TFLite Micro
or FlatBuffers runtime. Intermediate files are regenerated from the original
model and are not maintained as a second source of model parameters.

## Transformations

The current compiler applies:

- typed buffer reuse derived from tensor lifetimes;
- zero-copy views where graph semantics permit aliasing;
- exact integer kernels selected for the requested target;
- removal of softmax when the public result is only the top-1 class.

Target kernels are selected from graph shapes, quantization and statically
checked integer ranges. All generated arithmetic preserves the model result at
the API boundary used by the corresponding example.

## Research roadmap: certified decision execution

The published compiler currently preserves the model result at its documented
API boundary. A separate research direction is investigating `decision_exact`
execution for applications that require only a fixed threshold decision or
`argmax`: compute lower and upper bounds, stop once the original model's
decision is formally determined, and otherwise continue through an exact
fallback while reusing completed work.

This is not a currently implemented NN2Prog feature and is not model
distillation. Any future shortcut must be proved over its complete declared
input and state region; corpus data may measure its coverage after generation
but cannot establish correctness. Streaming models additionally require an
equivalent next state, not merely the same current decision.

## Validation

MLPerf Tiny KWS was compared with TFLite Micro on 1,024 deterministic full-range
int8 inputs and produced zero top-1 mismatches. Its hardware benchmark repeats
a 64-input golden sequence in three trials and checks the decision checksum.

Hey Jarvis v1 and v2 use frozen streaming feature sequences. The bundled tests
check 4,000 v1 invocations and 1,333 v2 invocations with zero output-byte
mismatches. The committed v1 C++ snapshot must also match fresh compiler output
byte for byte.

These tests cover the published models and input sequences. They are regression
and differential validation rather than a formal proof for every possible
input and unbounded streaming history.

## Measured ESP32 result

On an ESP32-D0WD-V3 at 240 MHz, MLPerf Tiny KWS measured 32,487,698 median cycles
(135.37 ms) per invocation, compared with 38,813,809 cycles for TFLite Micro
with ESP-NN. The generated firmware used 187,799 bytes of flash, 12,536 bytes of
static RAM and 16,064 bytes of working memory. Measurements depend on the
board and toolchain and can be reproduced with the bundled PlatformIO example.
