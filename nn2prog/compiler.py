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
            if target not in ("portable", "x86-avx2", "esp32", "esp32s3"):
                raise SystemExit("--target must be portable, x86-avx2, esp32 or esp32s3")
        else:
            raise SystemExit(f"unknown compiler option: {argument}")
    if model_dir is None:
        raise SystemExit("usage: compiler.py --model-dir=DIR [--target=portable|x86-avx2|esp32|esp32s3]")
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
    argmax_output = (elements(tensors[output_tensor]) > 1 and output_producer is not None
                     and output_producer["opcode"] == "SOFTMAX")
    if argmax_output:
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
            if target in ("esp32", "esp32s3") and op["opcode"] == "CONV_2D":
                input_shape = tensors[inp]["shape"]
                output_shape = tensors[output]["shape"]
                weight_shape = tensors[weights]["shape"]
                options = op["options"]
                symmetric_weights = all(zero == 0 for zero in wq["zero_point"])
                one_by_one = (len(input_shape) == 4 and len(output_shape) == 4
                              and len(weight_shape) == 4 and weight_shape[1:3] == [1, 1]
                              and weight_shape[3] == input_shape[3]
                              and weight_shape[0] == output_shape[3]
                              and options["dilation_h"] == 1 and options["dilation_w"] == 1)
                no_padding = (options["padding"] == "VALID"
                              or (options["padding"] == "SAME"
                                  and output_shape[1] == (input_shape[1] + options["stride_h"] - 1) // options["stride_h"]
                                  and output_shape[2] == (input_shape[2] + options["stride_w"] - 1) // options["stride_w"]))
                window = math.prod(weight_shape[1:])
                im2col = (target == "esp32s3" and len(input_shape) == 4
                          and len(output_shape) == 4 and len(weight_shape) == 4
                          and weight_shape[3] == input_shape[3]
                          and weight_shape[0] == output_shape[3]
                          and options["dilation_h"] == 1 and options["dilation_w"] == 1
                          and weight_shape[2] * input_shape[3] < 16
                          and 16 <= window <= 256)
                if ((one_by_one and no_padding) or im2col) and symmetric_weights:
                    weight_values = [s8(value) for value in const[(0, weights)]]
                    biases = constant_i32(const, _bias)
                    input_zero = iq["zero_point"][0]
                    adjusted_biases = []
                    safe = True
                    input_channels = input_shape[3]
                    for channel in range(channels):
                        channel_weights = weight_values[channel * window:(channel + 1) * window]
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
                        scratch_bytes = 0
                        if target == "esp32s3":
                            padded_channels = (window + 15) // 16 * 16
                            scratch_bytes = padded_channels
                            qacc_bound = max(128 * sum(abs(v) for v in weight_values[c*window:(c+1)*window])
                                             for c in range(channels))
                            qacc = (one_by_one and no_padding and channels % 16 == 0
                                    and qacc_bound < 2**19)
                            packed = []
                            if qacc:
                                for base in range(0, channels, 16):
                                    for i in range(padded_channels):
                                        packed.extend(weight_values[(base+c)*window+i] if i < window else 0
                                                      for c in range(16))
                                scratch_bytes += 64
                            else:
                                for channel in range(channels):
                                    packed.extend(weight_values[channel * window:(channel + 1) * window])
                                    packed.extend([0] * (padded_channels - window))
                            globals_.append(
                                f"alignas(16) constexpr std::array<std::int8_t,{len(packed)}> op{op['index']}_packed = "
                                f"{{{','.join(map(str, packed))}}};")
                            kernel_name = "esp32s3_conv_im2col_dot16" if im2col else "esp32s3_conv_1x1_dot16"
                            if qacc:
                                kernel_name = "esp32s3_conv_1x1_qacc16"
                        kernel_choices.append(KernelChoice(op["index"], kernel_name, scratch_bytes))
                        target_kernels.append({
                            "op": op["index"],
                            "opcode": op["opcode"],
                            "kernel": kernel_name,
                            "input_channels": input_channels,
                            "output_channels": channels,
                            "weight_zero_point": 0,
                            "proof": ("int32 folded-bias bound and signed20 raw-prefix bound"
                                      if kernel_name == "esp32s3_conv_1x1_qacc16"
                                      else "int32 interval bound and algebraic input-zero folding"),
                            **({"packed_layout": ("output-block16/input-padded16/output-lane" if qacc
                                                  else "output-channel/filter-window-padded16"),
                                "raw_dot_abs_bound": qacc_bound,
                                "packed_weight_bytes": len(packed),
                                "kernel_scratch_bytes": scratch_bytes} if target == "esp32s3" else {}),
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
            elif target in ("esp32", "esp32s3") and op["opcode"] == "DEPTHWISE_CONV_2D":
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
                    scratch_bytes = 0
                    if (target == "esp32s3" and input_shape[3] % 16 == 0
                            and weight_shape[1:3] == [3, 3]
                            and all(zero == 0 for zero in wq["zero_point"])):
                        values = [s8(value) for value in const[(0, weights)]]
                        biases = constant_i32(const, _bias)
                        adjusted = []
                        safe = True
                        for channel in range(channels):
                            row = values[channel::channels]
                            bias = biases[channel] - iq["zero_point"][0] * sum(row)
                            bound = sum(max(abs(-128*w), abs(127*w)) for w in row)
                            safe &= bound < 2**19 and abs(bias) + bound < 2**31
                            adjusted.append(bias)
                        if safe:
                            globals_.append(f"alignas(16) constexpr std::array<std::int8_t,{len(values)}> op{op['index']}_packed = {{{','.join(map(str,values))}}};")
                            globals_.append(f"constexpr std::array<std::int32_t,{channels}> op{op['index']}_esp32_bias = {{{','.join(map(str,adjusted))}}};")
                            kernel_name = "esp32s3_depthwise_3x3_qacc16"
                            scratch_bytes = 9*channels+64
                    kernel_choices.append(KernelChoice(op["index"], kernel_name, scratch_bytes))
                    target_kernels.append({
                        "op": op["index"],
                        "opcode": op["opcode"],
                        "kernel": kernel_name,
                        "channels": input_shape[3],
                        "weight_zero_point": 0,
                        "proof": ("signed20 dot bound, int32 folded-bias bound" if scratch_bytes
                                  else "per-channel accumulation order preserved"),
                        **({"kernel_scratch_bytes": scratch_bytes} if scratch_bytes else {}),
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
        elif op["opcode"] == "PAD":
            value, paddings_tensor = op["inputs"][:2]
            output = op["outputs"][0]
            shape = tensors[value]["shape"]
            paddings = constant_i32(const, paddings_tensor)
            if len(paddings) != 2 * len(shape) or any(amount < 0 for amount in paddings):
                raise ValueError(f"invalid PAD constants at op {op['index']}")
            before = paddings[::2]
            after = paddings[1::2]
            expected = [size + left + right for size, left, right in zip(shape, before, after)]
            if (tensors[output]["shape"] != expected
                    or tensors[output]["type"] != tensors[value]["type"]
                    or tensors[output]["quantization"] != tensors[value]["quantization"]):
                raise ValueError(f"PAD output contract mismatch at op {op['index']}")
            if tensors[value]["type"] != "INT8":
                raise ValueError(f"PAD op {op['index']} currently requires INT8")
            op["pad"] = {"before": before}
        elif op["opcode"] == "MEAN":
            value, axes_tensor = op["inputs"][:2]
            output = op["outputs"][0]
            input_shape = tensors[value]["shape"]
            axes = constant_i32(const, axes_tensor)
            normalized = []
            for axis in axes:
                axis += len(input_shape) if axis < 0 else 0
                if axis < 0 or axis >= len(input_shape):
                    raise ValueError(f"invalid MEAN axis at op {op['index']}")
                if axis not in normalized:
                    normalized.append(axis)
            keep_dims = op["options"].get("keep_dims", False)
            expected = [1 if index in normalized else size for index, size in enumerate(input_shape)]
            if not keep_dims:
                expected = [size for index, size in enumerate(input_shape) if index not in normalized]
            if tensors[output]["shape"] != expected:
                raise ValueError(f"MEAN output shape mismatch at op {op['index']}")
            if tensors[value]["type"] != "INT8" or tensors[output]["type"] != "INT8":
                raise ValueError(f"MEAN op {op['index']} currently requires INT8")
            iq, oq = tensors[value]["quantization"], tensors[output]["quantization"]
            multiplier, shift = quantize_multiplier(iq["scale"][0] / oq["scale"][0])
            count = math.prod(input_shape[axis] for axis in normalized)
            adjustment = min(count.bit_length() - 1, 32, 31 + shift)
            multiplier = (multiplier << adjustment) // count
            shift -= adjustment
            op["mean"] = {"axes": normalized, "count": count,
                          "multiplier": multiplier, "shift": shift}

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
