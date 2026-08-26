#!/usr/bin/env python3
"""Select affine operators compatible with the exact x86 masked-SIMD kernel.

The x86 kernel has a vector dot-product path when an int8 input has zero point
-128: subtracting the zero point maps its entire raw domain exactly to
unsigned [0, 255].  This pass proves that precondition from quantization,
checks the constant weights and int32 accumulator bounds, and selects every
compatible affine operator.
"""

import argparse
import json
import pathlib
import re
import struct


def elements(tensor):
    result = 1
    for dimension in tensor["shape"]:
        result *= dimension
    return result


def parse_constants(path):
    text = path.read_text()
    result = {}
    for graph, tensor, body in re.findall(r"sg(\d+)_tensor(\d+)_bytes = \{([^}]*)\}", text):
        result[(int(graph), int(tensor))] = bytes(int(value) for value in body.split(",") if value)
    return result


def signed(value):
    return value if value < 128 else value - 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="generated/v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--target", default="x86-avx2")
    args = parser.parse_args()
    if args.block % 16:
        raise SystemExit("the x86 SIMD block must be a multiple of 16")
    if args.target != "x86-avx2":
        raise SystemExit("only the proved x86-avx2 kernel contract is implemented")

    root = pathlib.Path(__file__).resolve().parent.parent
    model_dir = root / args.model_dir
    graph = json.loads((model_dir / "model.ir.json").read_text())["subgraphs"][0]
    tensors = {tensor["index"]: tensor for tensor in graph["tensors"]}
    constants = parse_constants(model_dir / "model.constants.h")
    operators = []

    for op in graph["operators"]:
        if op["opcode"] not in ("CONV_2D", "FULLY_CONNECTED"):
            continue
        input_id, weight_id, bias_id = op["inputs"][:3]
        output_id = op["outputs"][0]
        input_tensor, weight_tensor = tensors[input_id], tensors[weight_id]
        input_count, output_count = elements(input_tensor), elements(tensors[output_id])
        reasons = []
        input_zero = input_tensor["quantization"]["zero_point"][0]
        weight_zeros = weight_tensor["quantization"]["zero_point"]
        if input_tensor["type"] != "INT8":
            reasons.append("input_not_int8")
        if input_zero != -128:
            reasons.append("centered_input_not_provably_unsigned_u8")
        if any(value != 0 for value in weight_zeros):
            reasons.append("weights_not_symmetric_int8")
        if elements(weight_tensor) != input_count * output_count:
            reasons.append("not_flat_affine_geometry")
        if input_count < args.block:
            reasons.append("no_full_simd_block")

        accumulator_min = accumulator_max = 0
        zero_weights = zero_blocks = 0
        if not any(reason in reasons for reason in ("weights_not_symmetric_int8", "not_flat_affine_geometry")):
            weight_bytes = constants[(0, weight_id)]
            biases = struct.unpack("<" + "i" * output_count, constants[(0, bias_id)])
            lows, highs = [], []
            for channel in range(output_count):
                channel_weights = [signed(value) for value in
                                   weight_bytes[channel * input_count:(channel + 1) * input_count]]
                zero_weights += sum(value == 0 for value in channel_weights)
                zero_blocks += sum(all(value == 0 for value in channel_weights[begin:begin + args.block])
                                   for begin in range(0, input_count, args.block)
                                   if begin + args.block <= input_count)
                low = biases[channel] + 255 * sum(value for value in channel_weights if value < 0)
                high = biases[channel] + 255 * sum(value for value in channel_weights if value > 0)
                lows.append(low)
                highs.append(high)
            accumulator_min, accumulator_max = min(lows), max(highs)
            if accumulator_min < -(1 << 31) or accumulator_max >= (1 << 31):
                reasons.append("int32_accumulator_not_proved_safe")

        accepted = not reasons
        operators.append({
            "op": op["index"],
            "opcode": op["opcode"],
            "input_tensor": input_id,
            "input_elements": input_count,
            "output_elements": output_count,
            "input_zero_point": input_zero,
            "proved_centered_input_range": [0, 255] if input_zero == -128 else [-128 - input_zero, 127 - input_zero],
            "weight_zero_points": sorted(set(weight_zeros)),
            "accumulator_range": [accumulator_min, accumulator_max],
            "zero_weight_count": zero_weights,
            "full_zero_weight_blocks": zero_blocks,
            "accepted": accepted,
            "rejection_reasons": reasons,
        })

    selected = [item["op"] for item in operators if item["accepted"]]
    result = {
        "format": "nn2prog-static-simd-selection-v1",
        "target": args.target,
        "block": args.block,
        "proof_rule": "int8_zero_point_minus_128_implies_exact_unsigned_u8_centered_range",
        "selected_ops": selected,
        "operators": operators,
    }
    output = pathlib.Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"selected {len(selected)} masked SIMD operators: {','.join(map(str, selected)) or 'none'}")


if __name__ == "__main__":
    main()
