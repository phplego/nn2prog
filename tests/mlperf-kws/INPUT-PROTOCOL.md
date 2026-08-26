# Deterministic input protocol

The x86-64 and ESP32 benchmarks use the same 1,024 deterministic input tensors.
Each tensor contains 490 signed int8 values. A xorshift32 stream
starts at seed `0x4d4c5046`; all 490 values are consumed for every tensor. The
first four completed tensors are then replaced by the constants -128, 127, 83,
and 0 respectively.

Each top-1 decision is appended to a 64-bit wrapping checksum as
`checksum = checksum * 131 + decision`. The expected checksum is established by
the x86-64 differential test, which first requires exact agreement between
TFLite Micro and NN2Prog for every tensor. This protocol measures regression
correctness and execution time. The resulting single-pass golden checksum is
`675379874863121749`.
