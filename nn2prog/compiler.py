#!/usr/bin/env python3
import json
import math
import pathlib
import re
import struct
import sys

if __package__:
    from ._internal.cpp_emitter import CppContext, emit_cpp
    from ._internal.lowering import KernelChoice, elements, lower_program, write_memory_reports
else:
    from _internal.cpp_emitter import CppContext, emit_cpp
    from _internal.lowering import KernelChoice, elements, lower_program, write_memory_reports


def qround(x):
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def quantize_multiplier(value):
    if value == 0: return 0, 0
    q, shift = math.frexp(value)
    fixed = qround(q * (1 << 31))
    if fixed == 1 << 31: fixed //= 2; shift += 1
    if shift < -31: return 0, 0
    return fixed, shift


def activation_range(name, tensor):
    q = tensor["quantization"]
    scale, zero = q["scale"][0], q["zero_point"][0]
    def quant(v): return max(-128, min(127, qround(v / scale) + zero))
    if name == "NONE": return -128, 127
    if name == "RELU": return quant(0), 127
    if name == "RELU6": return quant(0), quant(6)
    if name == "RELU_N1_TO_1": return quant(-1), quant(1)
    raise ValueError(f"unsupported activation {name}")


def parse_constants(path):
    text = path.read_text()
    result = {}
    for graph, tensor, body in re.findall(r"sg(\d+)_tensor(\d+)_bytes = \{([^}]*)\}", text):
        result[(int(graph), int(tensor))] = bytes(int(x) for x in body.split(",") if x)
    return result


def s8(value): return value if value < 128 else value - 256


def constant_i32(const, tensor):
    data = const[(0, tensor)]
    if len(data) % 4:
        raise ValueError(f"tensor {tensor} is not an int32 constant")
    return list(struct.unpack(f"<{len(data) // 4}i", data))


def main():
    argmax_output = False
    masked_simd_ops = set()
    masked_simd_block = 32
    model_dir = None
    target = "portable"
    for argument in sys.argv[1:]:
        if argument.startswith("--model-dir="):
            model_dir = argument.split("=", 1)[1]
        elif argument.startswith("--target="):
            target = argument.split("=", 1)[1]
            if target not in ("portable", "x86-avx2", "esp32"):
                raise SystemExit("--target must be portable, x86-avx2 or esp32")
        else:
            raise SystemExit(f"unknown compiler option: {argument}")
    if model_dir is None:
        raise SystemExit("usage: compiler.py --model-dir=DIR [--target=portable|x86-avx2|esp32]")
    root = pathlib.Path(__file__).resolve().parent.parent
    out = root / model_dir
    if target == "x86-avx2":
        profile = json.loads((out / "model.static-simd-selection.json").read_text())
        masked_simd_ops = set(profile["selected_ops"])
        if profile["block"] != masked_simd_block:
            raise ValueError("masked SIMD plan must use 32-element blocks")
    ir = json.load((out / "model.ir.json").open())
    const = parse_constants(out / "model.constants.h")
    graph = ir["subgraphs"][0]
    tensors = {t["index"]: t for t in graph["tensors"]}
    ops = graph["operators"]
    if len(graph["inputs"]) != 1 or len(graph["outputs"]) != 1:
        raise ValueError("standalone generator currently requires one input and one output")
    input_tensor = graph["inputs"][0]
    output_tensor = graph["outputs"][0]
    output_producer = next((op for op in ops if output_tensor in op["outputs"]), None)
    argmax_output = elements(tensors[output_tensor]) > 1
    if argmax_output:
        if output_producer is None or output_producer["opcode"] != "SOFTMAX":
            raise ValueError("multi-value output requires SOFTMAX for the generated decision API")
        decision_tensor = output_producer["inputs"][0]
    else:
        decision_tensor = None

    globals_ = []
    target_kernels = []
    kernel_choices = [KernelChoice(index, "x86_dense_masked_simd32")
                      for index in sorted(masked_simd_ops)]
    for op in ops:
        if op["opcode"] in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
            inp, weights, _bias = op["inputs"][:3]; output = op["outputs"][0]
            iq, wq, oq = (tensors[x]["quantization"] for x in (inp, weights, output))
            scales = wq["scale"]
            channels = (tensors[output]["shape"][-1] if op["opcode"] == "DEPTHWISE_CONV_2D"
                        else tensors[weights]["shape"][0])
            multipliers, shifts = [], []
            for c in range(channels):
                m, s = quantize_multiplier(iq["scale"][0] * scales[c if len(scales) > 1 else 0] / oq["scale"][0])
                multipliers.append(m); shifts.append(s)
            amin, amax = activation_range(op["options"].get("activation", "NONE"), tensors[output])
            globals_.append(f"constexpr std::array<std::int32_t,{channels}> op{op['index']}_mult = {{{','.join(map(str,multipliers))}}};")
            globals_.append(f"constexpr std::array<int,{channels}> op{op['index']}_shift = {{{','.join(map(str,shifts))}}};")
            op["lower"] = {"amin": amin, "amax": amax}
            if target == "esp32" and op["opcode"] == "CONV_2D":
                input_shape = tensors[inp]["shape"]
                output_shape = tensors[output]["shape"]
                weight_shape = tensors[weights]["shape"]
                options = op["options"]
                symmetric_weights = wq["zero_point"][0] == 0
                one_by_one = (len(input_shape) == 4 and len(output_shape) == 4
                              and len(weight_shape) == 4 and weight_shape[1:3] == [1, 1]
                              and weight_shape[3] == input_shape[3]
                              and weight_shape[0] == output_shape[3]
                              and options["dilation_h"] == 1 and options["dilation_w"] == 1)
                no_padding = (options["padding"] == "VALID"
                              or (options["padding"] == "SAME"
                                  and output_shape[1] == (input_shape[1] + options["stride_h"] - 1) // options["stride_h"]
                                  and output_shape[2] == (input_shape[2] + options["stride_w"] - 1) // options["stride_w"]))
                if one_by_one and no_padding and symmetric_weights:
                    weight_values = [s8(value) for value in const[(0, weights)]]
                    biases = constant_i32(const, _bias)
                    input_zero = iq["zero_point"][0]
                    adjusted_biases = []
                    safe = True
                    input_channels = input_shape[3]
                    for channel in range(channels):
                        channel_weights = weight_values[channel * input_channels:(channel + 1) * input_channels]
                        adjusted = biases[channel] - input_zero * sum(channel_weights)
                        bound = abs(adjusted) + sum(max(abs(-128 * weight), abs(127 * weight))
                                                    for weight in channel_weights)
                        safe &= bound < 2**31
                        adjusted_biases.append(adjusted)
                    if safe:
                        globals_.append(
                            f"constexpr std::array<std::int32_t,{channels}> op{op['index']}_esp32_bias = "
                            f"{{{','.join(map(str, adjusted_biases))}}};")
                        kernel_name = "esp32_conv_1x1_unrolled8_bias_fold"
                        kernel_choices.append(KernelChoice(op["index"], kernel_name))
                        target_kernels.append({
                            "op": op["index"],
                            "opcode": op["opcode"],
                            "kernel": kernel_name,
                            "input_channels": input_channels,
                            "output_channels": channels,
                            "weight_zero_point": 0,
                            "proof": "int32 interval bound and algebraic input-zero folding",
                        })
                elif (len(input_shape) == 4 and len(output_shape) == 4 and len(weight_shape) == 4
                      and weight_shape[3] == input_shape[3]
                      and weight_shape[0] == output_shape[3]
                      and options["dilation_h"] == 1 and options["dilation_w"] == 1
                      and symmetric_weights):
                    kernel_name = "esp32_conv_nhwc_unrolled8"
                    kernel_choices.append(KernelChoice(op["index"], kernel_name))
                    target_kernels.append({
                        "op": op["index"],
                        "opcode": op["opcode"],
                        "kernel": kernel_name,
                        "input_channels": input_shape[3],
                        "output_channels": channels,
                        "weight_zero_point": 0,
                        "proof": "loop-order-preserving pointer specialization",
                    })
            elif target == "esp32" and op["opcode"] == "DEPTHWISE_CONV_2D":
                input_shape = tensors[inp]["shape"]
                output_shape = tensors[output]["shape"]
                weight_shape = tensors[weights]["shape"]
                options = op["options"]
                if (len(input_shape) == 4 and len(output_shape) == 4 and len(weight_shape) == 4
                        and weight_shape[0] == 1 and weight_shape[3] == output_shape[3]
                        and output_shape[3] == input_shape[3]
                        and input_shape[3] % 4 == 0
                        and options["depth_multiplier"] == 1
                        and options["dilation_h"] == 1 and options["dilation_w"] == 1
                        and wq["zero_point"][0] == 0):
                    kernel_name = "esp32_depthwise_channels4"
                    kernel_choices.append(KernelChoice(op["index"], kernel_name))
                    target_kernels.append({
                        "op": op["index"],
                        "opcode": op["opcode"],
                        "kernel": kernel_name,
                        "channels": input_shape[3],
                        "weight_zero_point": 0,
                        "proof": "per-channel accumulation order preserved",
                    })
        elif op["opcode"] == "MUL":
            a,b=op["inputs"]; o=op["outputs"][0]
            aq,bq,oq=(tensors[x]["quantization"] for x in (a,b,o))
            m,s=quantize_multiplier(aq["scale"][0]*bq["scale"][0]/oq["scale"][0])
            amin,amax=activation_range(op["options"].get("activation","NONE"),tensors[o])
            op["lower"]={"m":m,"s":s,"amin":amin,"amax":amax}
        elif op["opcode"] == "ADD":
            a,b=op["inputs"]; o=op["outputs"][0]
            aq,bq,oq=(tensors[x]["quantization"] for x in (a,b,o)); left=20
            twice=2*max(aq["scale"][0],bq["scale"][0])
            m1,s1=quantize_multiplier(aq["scale"][0]/twice)
            m2,s2=quantize_multiplier(bq["scale"][0]/twice)
            mo,so=quantize_multiplier(twice/((1<<left)*oq["scale"][0]))
            amin,amax=activation_range(op["options"].get("activation","NONE"),tensors[o])
            op["lower"]={"left":left,"m1":m1,"s1":s1,"m2":m2,"s2":s2,"mo":mo,"so":so,"amin":amin,"amax":amax}
        elif op["opcode"] == "AVERAGE_POOL_2D":
            output = op["outputs"][0]
            amin, amax = activation_range(op["options"].get("activation", "NONE"), tensors[output])
            op["lower"] = {"amin": amin, "amax": amax}
        elif op["opcode"] == "SPLIT_V":
            value, sizes_tensor, axis_tensor = op["inputs"][:3]
            shape = tensors[value]["shape"]
            axis_values = constant_i32(const, axis_tensor)
            split_sizes = constant_i32(const, sizes_tensor)
            if len(axis_values) != 1 or len(split_sizes) != len(op["outputs"]):
                raise ValueError(f"invalid SPLIT_V constants at op {op['index']}")
            axis = axis_values[0]
            if axis < 0: axis += len(shape)
            if axis < 0 or axis >= len(shape):
                raise ValueError(f"invalid SPLIT_V axis at op {op['index']}")
            unknown = [index for index, size in enumerate(split_sizes) if size == -1]
            if len(unknown) > 1 or any(size < -1 for size in split_sizes):
                raise ValueError(f"invalid SPLIT_V sizes at op {op['index']}")
            if unknown:
                split_sizes[unknown[0]] = shape[axis] - sum(size for size in split_sizes if size >= 0)
            if sum(split_sizes) != shape[axis]:
                raise ValueError(f"SPLIT_V sizes do not cover axis at op {op['index']}")
            for output, size in zip(op["outputs"], split_sizes):
                expected_shape = list(shape); expected_shape[axis] = size
                if tensors[output]["shape"] != expected_shape or tensors[output]["type"] != tensors[value]["type"]:
                    raise ValueError(f"SPLIT_V output contract mismatch at op {op['index']}")
            outer = math.prod(shape[:axis]); inner = math.prod(shape[axis + 1:])
            op["split_v"] = {"axis_size": shape[axis], "sizes": split_sizes,
                             "outer": outer, "inner": inner}

    # A quantized logistic input has only 256 possible values, so a lookup table is exact.
    logistic = next((op for op in ops if op["opcode"] == "LOGISTIC"), None)
    if logistic is not None:
        li, lo = (tensors[x] for x in (logistic["inputs"][0], logistic["outputs"][0]))
        si, zi = li["quantization"]["scale"][0], li["quantization"]["zero_point"][0]
        so, zo = lo["quantization"]["scale"][0], lo["quantization"]["zero_point"][0]
        lut=[]
        for raw in range(-128,128):
            real=(raw-zi)*si
            sigmoid=1.0/(1.0+math.exp(-real)) if real < 700 else 1.0
            lut.append(max(-128,min(127,qround(sigmoid/so)+zo)))
        globals_.append(f"constexpr std::array<std::int8_t,256> logistic_lut = {{{','.join(map(str,lut))}}};")

    omitted_outputs = (output_tensor,) if argmax_output else ()
    program = lower_program(graph, tensors, target, kernel_choices, omitted_outputs)
    write_memory_reports(program, out, tensors)

    target_plan_path = out / "model.target-plan.json"
    if target != "portable":
        target_plan_path.write_text(json.dumps({
            "format": "nn2prog-target-plan-v1",
            "target": target,
            "selection": "static-ir-shapes-quantization-and-int32-range-proof",
            "program": {
                "scratch_initialization": "producer-defined-no-zero-fill",
                "esp32_code_placement": "iram1",
            },
            "operators": target_kernels,
        }, indent=2) + "\n")
    else:
        target_plan_path.unlink(missing_ok=True)

    emit_cpp(CppContext(
        output_dir=out,
        graph=graph,
        tensors=tensors,
        program=program,
        globals=tuple(globals_),
        argmax_output=argmax_output,
        decision_tensor=decision_tensor,
        masked_simd_block=masked_simd_block,
    ))


if __name__ == "__main__": main()
