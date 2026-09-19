"""Lower a tensor graph to explicit storage, alias, and kernel decisions."""

from dataclasses import dataclass
import json


TYPE_BYTES = {"INT8": 1, "UINT8": 1, "INT16": 2, "INT32": 4, "INT64": 8}


def elements(tensor):
    result = 1
    for dimension in tensor["shape"]:
        result *= dimension
    return result


@dataclass(frozen=True)
class Alias:
    tensor: int
    source: int
    opcode: str


@dataclass(frozen=True)
class StorageAllocation:
    tensor: int
    slot: int
    temporary: bool


@dataclass(frozen=True)
class StorageSlot:
    index: int
    type: str
    elements: int
    bytes: int


@dataclass(frozen=True)
class KernelChoice:
    op: int
    name: str
    scratch_bytes: int = 0


@dataclass(frozen=True)
class LoweredProgram:
    target: str
    aliases: tuple
    storage_allocations: tuple
    storage_slots: tuple
    working_memory_bytes: int
    kernels: tuple

    def kernel(self, op):
        return next((item.name for item in self.kernels if item.op == op), None)

    def storage_slot(self, tensor):
        allocation = self.storage_allocation(tensor)
        return allocation.slot if allocation is not None else None

    def storage_allocation(self, tensor):
        return next((item for item in self.storage_allocations if item.tensor == tensor), None)


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


def _live_intervals(graph, tensors, runtime_outputs, aliases):
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
        })

    return intervals


def _allocate_storage(graph, intervals, tensors, aliases):
    intervals_by_tensor = {item["tensor"]: item for item in intervals}
    alias_sources = {item.tensor: item.source for item in aliases}
    fresh_outputs = set()
    for op in graph["operators"]:
        for output in op["outputs"]:
            if output not in intervals_by_tensor:
                continue
            output_tensor = tensors[output]
            for input_index in op["inputs"]:
                source = input_index
                while source in alias_sources:
                    source = alias_sources[source]
                source_interval = intervals_by_tensor.get(source)
                if (source_interval is not None
                        and source_interval["end"] == op["index"]
                        and tensors[input_index]["type"] == output_tensor["type"]
                        and elements(tensors[input_index]) == elements(output_tensor)):
                    fresh_outputs.add(output)
                    break

    slots = []
    allocations = []
    for interval in sorted(intervals, key=lambda item: (item["start"], -item["bytes"], item["tensor"])):
        tensor = tensors[interval["tensor"]]
        temporary = interval["tensor"] in fresh_outputs
        available = [item for item in slots
                     if (item["type"] == tensor["type"]
                         and (item["end"] <= interval["start"] if temporary
                              else item["end"] < interval["start"]))]
        slot = min(available, key=lambda item: max(item["bytes"], interval["bytes"]), default=None)
        if slot is None:
            slot = {"index": len(slots), "type": tensor["type"],
                    "elements": elements(tensor),
                    "bytes": interval["bytes"], "end": interval["end"]}
            slots.append(slot)
        else:
            slot["end"] = interval["end"]
            if interval["bytes"] > slot["bytes"]:
                slot["bytes"] = interval["bytes"]
                slot["elements"] = elements(tensor)
        allocations.append(StorageAllocation(interval["tensor"], slot["index"], temporary))

    persistent_bytes = sum(item["bytes"] for item in slots)
    temporary_bytes = 0
    for start in {item["start"] for item in intervals}:
        temporary_bytes = max(temporary_bytes,
                              sum(item["bytes"] for item in intervals
                                  if item["start"] == start and item["tensor"] in fresh_outputs))
    storage_slots = tuple(StorageSlot(item["index"], item["type"],
                                     item["elements"], item["bytes"])
                          for item in slots)
    return persistent_bytes + temporary_bytes, tuple(allocations), storage_slots


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
    intervals = _live_intervals(graph, tensors, runtime_outputs, aliases)
    working_bytes, storage_allocations, storage_slots = _allocate_storage(
        graph, intervals, tensors, aliases)
    working_bytes += max((item.scratch_bytes for item in kernels), default=0)
    return LoweredProgram(target, aliases, storage_allocations, storage_slots,
                          working_bytes, tuple(kernels))


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
    (output_dir / "model.storage-plan.json").write_text(json.dumps({
        "format": "nn2prog-storage-plan-v1",
        "working_memory_bytes": program.working_memory_bytes,
        "slots": [item.__dict__ for item in program.storage_slots],
        "allocations": [item.__dict__ for item in program.storage_allocations],
        "initialization": "producer-defined-no-zero-fill",
    }, indent=2) + "\n")
