#!/usr/bin/env python3
"""Translate a TFLite FlatBuffer into NN2Prog IR using only Python stdlib."""

import argparse
import json
import pathlib
import struct


TENSOR_TYPES = {
    0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8", 4: "INT64",
    5: "STRING", 6: "BOOL", 7: "INT16", 8: "COMPLEX64", 9: "INT8",
    10: "FLOAT64", 11: "COMPLEX128", 12: "UINT64", 13: "RESOURCE",
    14: "VARIANT", 15: "UINT32", 16: "UINT16", 17: "INT4",
    18: "BFLOAT16",
}

# TFLite schema enum values supported by the generated runtime.
OPERATORS = {
    0: "ADD", 1: "AVERAGE_POOL_2D", 2: "CONCATENATION", 3: "CONV_2D", 4: "DEPTHWISE_CONV_2D",
    6: "DEQUANTIZE",
    9: "FULLY_CONNECTED", 14: "LOGISTIC", 18: "MUL", 22: "RESHAPE",
    25: "SOFTMAX", 45: "STRIDED_SLICE", 102: "SPLIT_V", 114: "QUANTIZE", 129: "CALL_ONCE",
    142: "VAR_HANDLE", 143: "READ_VARIABLE", 144: "ASSIGN_VARIABLE",
}

OPTION_TYPES = {
    0: "NONE", 1: "Conv2DOptions", 2: "DepthwiseConv2DOptions",
    5: "Pool2DOptions", 8: "FullyConnectedOptions", 9: "SoftmaxOptions",
    10: "ConcatenationOptions", 17: "ReshapeOptions",
    11: "AddOptions", 21: "MulOptions", 32: "StridedSliceOptions",
    79: "SplitVOptions",
    103: "CallOnceOptions", 111: "VarHandleOptions",
}

PADDING = {0: "SAME", 1: "VALID"}
ACTIVATION = {0: "NONE", 1: "RELU", 2: "RELU_N1_TO_1", 3: "RELU6", 4: "TANH", 5: "SIGN_BIT"}


class FlatBuffer:
    """Small bounds-checked FlatBuffer table/vector reader."""

    def __init__(self, data):
        self.data = data

    def unpack(self, fmt, offset):
        size = struct.calcsize("<" + fmt)
        if offset < 0 or offset + size > len(self.data):
            raise ValueError("offset outside FlatBuffer")
        return struct.unpack_from("<" + fmt, self.data, offset)[0]

    def root(self):
        if len(self.data) < 8 or self.data[4:8] != b"TFL3":
            raise ValueError("not a TFLite FlatBuffer (missing TFL3 identifier)")
        return self.unpack("I", 0)

    def field(self, table, field_id):
        vtable = table - self.unpack("i", table)
        vtable_size = self.unpack("H", vtable)
        entry = vtable + 4 + field_id * 2
        if entry + 2 > vtable + vtable_size:
            return None
        relative = self.unpack("H", entry)
        return table + relative if relative else None

    def scalar(self, table, field_id, fmt, default=0):
        address = self.field(table, field_id)
        return default if address is None else self.unpack(fmt, address)

    def indirect(self, address):
        return None if address is None else address + self.unpack("I", address)

    def table_field(self, table, field_id):
        return self.indirect(self.field(table, field_id))

    def string(self, table, field_id, default=""):
        value = self.table_field(table, field_id)
        if value is None:
            return default
        length = self.unpack("I", value)
        end = value + 4 + length
        if end > len(self.data):
            raise ValueError("string outside FlatBuffer")
        return self.data[value + 4:end].decode("utf-8", errors="replace")

    def vector(self, table, field_id):
        value = self.table_field(table, field_id)
        if value is None:
            return None
        return value + 4, self.unpack("I", value)

    def scalar_vector(self, table, field_id, fmt):
        vector = self.vector(table, field_id)
        if vector is None:
            return []
        start, length = vector
        size = struct.calcsize("<" + fmt)
        return [self.unpack(fmt, start + size * index) for index in range(length)]

    def table_vector(self, table, field_id):
        vector = self.vector(table, field_id)
        if vector is None:
            return []
        start, length = vector
        return [self.indirect(start + 4 * index) for index in range(length)]

    def bytes_vector(self, table, field_id):
        vector = self.vector(table, field_id)
        if vector is None:
            return b""
        start, length = vector
        if start + length > len(self.data):
            raise ValueError("byte vector outside FlatBuffer")
        return self.data[start:start + length]


def enum_name(mapping, value, kind):
    try:
        return mapping[value]
    except KeyError as error:
        raise ValueError(f"unsupported TFLite {kind} value {value}") from error


def f32_values(reader, table, field_id):
    # Keep JSON serialization stable while retaining enough precision for float32.
    return [float(format(value, ".10g")) for value in reader.scalar_vector(table, field_id, "f")]


def operator_options(reader, opcode, table):
    if table is None:
        return {}
    activation = lambda field: enum_name(ACTIVATION, reader.scalar(table, field, "b"), "activation")
    if opcode == "CONV_2D":
        return {"padding": enum_name(PADDING, reader.scalar(table, 0, "b"), "padding"),
                "stride_w": reader.scalar(table, 1, "i"), "stride_h": reader.scalar(table, 2, "i"),
                "dilation_w": reader.scalar(table, 4, "i", 1), "dilation_h": reader.scalar(table, 5, "i", 1),
                "activation": activation(3)}
    if opcode == "DEPTHWISE_CONV_2D":
        return {"padding": enum_name(PADDING, reader.scalar(table, 0, "b"), "padding"),
                "stride_w": reader.scalar(table, 1, "i"), "stride_h": reader.scalar(table, 2, "i"),
                "dilation_w": reader.scalar(table, 5, "i", 1), "dilation_h": reader.scalar(table, 6, "i", 1),
                "depth_multiplier": reader.scalar(table, 3, "i"), "activation": activation(4)}
    if opcode == "AVERAGE_POOL_2D":
        return {"padding": enum_name(PADDING, reader.scalar(table, 0, "b"), "padding"),
                "stride_w": reader.scalar(table, 1, "i"), "stride_h": reader.scalar(table, 2, "i"),
                "filter_w": reader.scalar(table, 3, "i"), "filter_h": reader.scalar(table, 4, "i"),
                "activation": activation(5)}
    if opcode == "FULLY_CONNECTED":
        return {"activation": activation(0), "keep_num_dims": bool(reader.scalar(table, 2, "B"))}
    if opcode == "CONCATENATION":
        return {"axis": reader.scalar(table, 0, "i"), "activation": activation(1)}
    if opcode in ("ADD", "MUL"):
        return {"activation": activation(0)}
    if opcode == "RESHAPE":
        return {"new_shape": reader.scalar_vector(table, 0, "i")}
    if opcode == "SOFTMAX":
        return {"beta": reader.scalar(table, 0, "f", 1.0)}
    if opcode == "CALL_ONCE":
        return {"init_subgraph_index": reader.scalar(table, 0, "i")}
    if opcode == "VAR_HANDLE":
        return {"container": reader.string(table, 0), "shared_name": reader.string(table, 1)}
    return {}


def import_model(data):
    reader = FlatBuffer(data)
    model = reader.root()
    buffers = [reader.bytes_vector(buffer, 0) for buffer in reader.table_vector(model, 4)]
    opcodes = []
    for code in reader.table_vector(model, 1):
        # Older producers populated only deprecated_builtin_code (field 0).
        # Newer producers use builtin_code (field 3) and retain 127 in field 0
        # for opcodes that no longer fit in a byte.  Taking the maximum matches
        # the compatibility rule used by TFLite itself without model-specific
        # format assumptions.
        builtin = max(reader.scalar(code, 0, "b"), reader.scalar(code, 3, "i"))
        opcodes.append((enum_name(OPERATORS, builtin, "operator"), reader.scalar(code, 2, "i", 1)))

    result = {"format": "nn2prog-ir-v1", "source_schema_version": reader.scalar(model, 0, "I"),
              "description": reader.string(model, 3), "subgraphs": []}
    constants = []
    for graph_index, graph in enumerate(reader.table_vector(model, 2)):
        graph_result = {"index": graph_index, "name": reader.string(graph, 4),
                        "inputs": reader.scalar_vector(graph, 1, "i"),
                        "outputs": reader.scalar_vector(graph, 2, "i"), "tensors": [], "operators": []}
        for tensor_index, tensor in enumerate(reader.table_vector(graph, 0)):
            buffer_index = reader.scalar(tensor, 2, "I")
            if buffer_index >= len(buffers):
                raise ValueError(f"tensor references missing buffer {buffer_index}")
            quant = reader.table_field(tensor, 4)
            graph_result["tensors"].append({
                "index": tensor_index, "name": reader.string(tensor, 3),
                "type": enum_name(TENSOR_TYPES, reader.scalar(tensor, 1, "b"), "tensor type"),
                "shape": reader.scalar_vector(tensor, 0, "i"),
                "variable": bool(reader.scalar(tensor, 5, "B")), "buffer": buffer_index,
                "buffer_bytes": len(buffers[buffer_index]),
                "quantization": {"scale": [] if quant is None else f32_values(reader, quant, 2),
                                 "zero_point": [] if quant is None else reader.scalar_vector(quant, 3, "q"),
                                 "axis": 0 if quant is None else reader.scalar(quant, 6, "i")},
            })
            if buffers[buffer_index]:
                constants.append((graph_index, tensor_index, buffers[buffer_index]))
        for operator_index, operator in enumerate(reader.table_vector(graph, 3)):
            opcode_index = reader.scalar(operator, 0, "I")
            if opcode_index >= len(opcodes):
                raise ValueError(f"operator references missing opcode {opcode_index}")
            opcode, version = opcodes[opcode_index]
            option_type_value = reader.scalar(operator, 3, "B")
            option_type = enum_name(OPTION_TYPES, option_type_value, "options type")
            options_table = reader.table_field(operator, 4)
            graph_result["operators"].append({
                "index": operator_index, "opcode_index": opcode_index, "opcode": opcode, "version": version,
                "inputs": reader.scalar_vector(operator, 1, "i"),
                "outputs": reader.scalar_vector(operator, 2, "i"),
                "options_type": option_type, "options": operator_options(reader, opcode, options_table),
            })
        result["subgraphs"].append(graph_result)
    return result, constants


def write_constants(path, constants):
    with path.open("w") as output:
        output.write("#pragma once\n#include <array>\n#include <cstdint>\n\nnamespace nn2prog::generated {\n")
        for graph, tensor, data in constants:
            values = ",".join(str(value) for value in data)
            output.write(f"inline constexpr std::array<std::uint8_t,{len(data)}> sg{graph}_tensor{tensor}_bytes = {{{values}}};\n")
        output.write("}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("ir", type=pathlib.Path)
    parser.add_argument("constants", type=pathlib.Path)
    args = parser.parse_args()
    try:
        model, constants = import_model(args.model.read_bytes())
        args.ir.parent.mkdir(parents=True, exist_ok=True)
        args.constants.parent.mkdir(parents=True, exist_ok=True)
        args.ir.write_text(json.dumps(model, indent=2, separators=(",", ": ")) + "\n")
        write_constants(args.constants, constants)
    except (OSError, ValueError, struct.error) as error:
        raise SystemExit(f"import failed: {error}") from error
    print(f"imported {args.model} -> {args.ir}, {args.constants}")


if __name__ == "__main__":
    main()
