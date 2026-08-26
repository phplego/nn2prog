"""Lower a tensor graph to explicit storage, alias, and kernel decisions."""

from dataclasses import dataclass
import json


TYPE_BYTES = {"INT8": 1, "UINT8": 1, "INT16": 2, "INT32": 4, "INT64": 8}


def elements(tensor):
    result = 1
    for dimension in tensor["shape"]:
        result *= dimension
    return result


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Alias:
    tensor: int
    source: int
    opcode: str


@dataclass(frozen=True)
class Allocation:
    tensor: int
    start: int
    end: int
    bytes: int
    alignment: int
    offset: int


@dataclass(frozen=True)
class KernelChoice:
    op: int
    name: str


@dataclass(frozen=True)
class LoweredProgram:
    target: str
    aliases: tuple
    allocations: tuple
    scratch_bytes: int
    kernels: tuple

    def allocation(self, tensor):
        return next((item for item in self.allocations if item.tensor == tensor), None)

    def kernel(self, op):
        return next((item.name for item in self.kernels if item.op == op), None)


def _find_aliases(graph, tensors):
    aliases = []
    for op in graph["operators"]:
        code = op["opcode"]
        if code not in ("RESHAPE", "READ_VARIABLE", "STRIDED_SLICE"):
            continue
        output = op["outputs"][0]
        source = op["inputs"][0]
        if code != "READ_VARIABLE" and tensors[source]["type"] != tensors[output]["type"]:
            raise ValueError(f"{code} op {op['index']} changes tensor type")
        if code == "RESHAPE" and elements(tensors[source]) != elements(tensors[output]):
            raise ValueError(f"RESHAPE op {op['index']} changes element count")
        if code == "STRIDED_SLICE" and elements(tensors[output]) > elements(tensors[source]):
            raise ValueError(f"STRIDED_SLICE op {op['index']} output exceeds input")
        aliases.append(Alias(output, source, code))
    return tuple(aliases)


def _allocate_scratch(graph, tensors, runtime_outputs, aliases):
    producers = {}
    consumers = {}
    for op in graph["operators"]:
        for index in op["outputs"]:
            if index >= 0:
                producers[index] = op["index"]
        for index in op["inputs"]:
            if index >= 0:
                consumers.setdefault(index, []).append(op["index"])

    alias_sources = {item.tensor: item.source for item in aliases}
    changed = True
    while changed:
        changed = False
        for alias, source in alias_sources.items():
            inherited = consumers.get(alias, [])
            target = consumers.setdefault(source, [])
            before = len(target)
            target.extend(step for step in inherited if step not in target)
            changed |= len(target) != before

    final_step = max(op["index"] for op in graph["operators"]) + 1
    graph_outputs = set(graph["outputs"])
    for output in list(graph_outputs):
        source = alias_sources.get(output)
        while source is not None:
            graph_outputs.add(source)
            source = alias_sources.get(source)

    intervals = []
    for index in runtime_outputs:
        tensor = tensors[index]
        width = TYPE_BYTES[tensor["type"]]
        start = producers[index]
        end = max(consumers.get(index, [start]))
        if index in graph_outputs:
            end = final_step
        intervals.append({
            "tensor": index,
            "start": start,
            "end": end,
            "bytes": elements(tensor) * width,
            "alignment": width,
        })

    active = []
    arena_size = 0
    for interval in sorted(intervals, key=lambda item: (item["start"], -item["bytes"], item["tensor"])):
        active = [item for item in active if item["end"] >= interval["start"]]
        occupied = sorted((item["offset"], item["offset"] + item["bytes"]) for item in active)
        offset = 0
        for begin, end in occupied:
            offset = align_up(offset, interval["alignment"])
            if offset + interval["bytes"] <= begin:
                break
            offset = max(offset, end)
        offset = align_up(offset, interval["alignment"])
        interval["offset"] = offset
        arena_size = max(arena_size, offset + interval["bytes"])
        active.append(interval)

    return arena_size, tuple(Allocation(**item) for item in intervals)


def lower_program(graph, tensors, target, kernels=(), omitted_outputs=()):
    input_tensor = graph["inputs"][0]
    aliases = _find_aliases(graph, tensors)
    alias_tensors = {item.tensor for item in aliases}
    omitted_outputs = set(omitted_outputs)
    runtime_outputs = sorted({
        index
        for op in graph["operators"]
        for index in op["outputs"]
        if (tensors[index]["type"] != "RESOURCE"
            and not tensors[index]["buffer_bytes"]
            and index != input_tensor
            and index not in alias_tensors
            and index not in omitted_outputs)
    })
    scratch_bytes, allocations = _allocate_scratch(graph, tensors, runtime_outputs, aliases)
    return LoweredProgram(target, aliases, allocations, scratch_bytes, tuple(kernels))


def write_memory_reports(program, output_dir, tensors):
    by_opcode = {}
    for alias in program.aliases:
        entry = by_opcode.setdefault(alias.opcode, {"tensors": 0, "materialized_bytes_avoided": 0})
        entry["tensors"] += 1
        entry["materialized_bytes_avoided"] += elements(tensors[alias.tensor]) * TYPE_BYTES[tensors[alias.tensor]["type"]]
    (output_dir / "model.views-plan.json").write_text(json.dumps({
        "format": "nn2prog-views-plan-v1",
        "transformations": by_opcode,
        "aliased_tensors": [
            {"tensor": item.tensor, "opcode": item.opcode,
             "bytes": elements(tensors[item.tensor]) * TYPE_BYTES[tensors[item.tensor]["type"]]}
            for item in program.aliases
        ],
    }, indent=2) + "\n")
    (output_dir / "model.scratch-plan.json").write_text(json.dumps({
        "format": "nn2prog-scratch-plan-v1",
        "arena_bytes": program.scratch_bytes,
        "initialization": "producer-defined-no-zero-fill",
        "allocations": [item.__dict__ for item in program.allocations],
    }, indent=2) + "\n")
