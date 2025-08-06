import torch
from max import mlir
from max.graph import KernelLibrary
from max.torch.torch import CustomOpLibrary

from .mappings import MAPPING_TORCH_TO_MOJO_FUNCTIONS
from .ops import CompiledFunctionMaxOp


class TensorsBook:
    def __init__(self):
        self.tensors = {}

    def __setitem__(self, name: str, tensor):
        self.tensors[name] = tensor

    def convert_to_max(self, something):
        if isinstance(something, torch.fx.Node):
            return self.tensors[something.name]
        elif isinstance(something, int):
            return something
        raise ValueError(f"Unsupported type: {type(something)}")


def modular_max_compiler(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    gm.graph.print_tabular()

    # Use meta tensors (no memory allocation, no computation)
    # Meta tensors only track shape/dtype/device metadata
    meta_inputs = [torch.empty_like(inp, device="meta") for inp in example_inputs]
    with torch.no_grad():
        meta_outputs = gm(*meta_inputs)
        if isinstance(meta_outputs, torch.Tensor):
            meta_outputs = [meta_outputs]

    def create_max_graph(*args):
        tensor_book = TensorsBook()

        args_index = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                tensor_book[node.name] = args[args_index]
                args_index += 1
            elif node.op == "call_function":
                func_args = [tensor_book.convert_to_max(x) for x in node.args]
                func_kwags = {
                    k: tensor_book.convert_to_max(v) for k, v in node.kwargs.items()
                }
                if node.target not in MAPPING_TORCH_TO_MOJO_FUNCTIONS:
                    raise ValueError(
                        f"Function {node.target} not supported by the Max backend yet."
                    )
                tensor = MAPPING_TORCH_TO_MOJO_FUNCTIONS[node.target](
                    *func_args, **func_kwags
                )
                tensor_book[node.name] = tensor
            elif node.op == "output":
                return tuple(tensor_book.convert_to_max(x) for x in node.args[0])

    op = CompiledFunctionMaxOp(
        create_max_graph,
        create_max_graph.__name__,
        CustomOpLibrary(KernelLibrary(mlir.Context())),
        input_types=None,
        output_types=None,
        num_outputs=len(meta_outputs),
        num_inputs=len(example_inputs),
    )
    custom_op_def = op.custom_op_def()

    def equivalent_max_function(*args) -> torch.Tensor:
        results = [torch.empty_like(x, device=args[0].device) for x in meta_outputs]
        custom_op_def(*results, *args)
        return results

    return equivalent_max_function
