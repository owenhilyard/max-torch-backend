
import torch
import max
from max.torch.torch import MaxOp, CustomOpLibrary
import inspect
from max.graph import KernelLibrary
from max import mlir
import operator
print("imports done")


class MyMaxOp(MaxOp):
    def __init__(self, *args, num_inputs: int, **kwargs):
        self.num_inputs = num_inputs
        super().__init__(*args, **kwargs)
     
    @property
    def torch_signature(self) -> inspect.Signature:
        dps_args = [
            inspect.Parameter(
                f"__out{i}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
            )
            for i in range(self.num_outputs)
        ]
        args = [
            inspect.Parameter(
                f"__input{i}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
            )
            for i in range(self.num_inputs)
        ]
        return inspect.Signature((*dps_args, *args), return_annotation=None)
    

def get_the_number_of_outputs(gm: torch.fx.GraphModule) -> int:
    last_node = next(iter(reversed(gm.graph.nodes)))
    assert last_node.op == "output"
    return len(last_node.args[0])



IDENTICAL_FUNCTIONS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    operator.mod,
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.floor_divide,
    torch.pow,
    torch.remainder
]



MAPPING_TORCH_TO_MOJO_FUNCTIONS = {
    torch.abs: max.graph.ops.abs,
    torch.cos: max.graph.ops.cos,
    torch.sin: max.graph.ops.sin,
}

for func in IDENTICAL_FUNCTIONS:
    MAPPING_TORCH_TO_MOJO_FUNCTIONS[func] = func

def my_compiler(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    gm.graph.print_tabular()
    
    # Use meta tensors (no memory allocation, no computation)
    # Meta tensors only track shape/dtype/device metadata
    meta_inputs = [torch.empty_like(inp, device='meta') for inp in example_inputs]
    with torch.no_grad():
        meta_outputs = gm(*meta_inputs)
        if isinstance(meta_outputs, torch.Tensor):
            meta_outputs = [meta_outputs]
    
    def max_add_i_want_to_use(*args):
        mapping_names_to_tensors = {}

        def get_tensor_or_value(something):
            if isinstance(something, torch.fx.Node):
                return mapping_names_to_tensors[something.name]
            elif isinstance(something, int):
                return something

        args_index = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                mapping_names_to_tensors[node.name] = args[args_index]
                args_index += 1
            elif node.op == "call_function":
                func_args = [get_tensor_or_value(x) for x in node.args]
                func_kwags = {k: get_tensor_or_value(v) for k, v in node.kwargs.items()}
                if node.target not in MAPPING_TORCH_TO_MOJO_FUNCTIONS:
                    raise ValueError(f"Function {node.target} not supported by the Max backend yet.")
                tensor = MAPPING_TORCH_TO_MOJO_FUNCTIONS[node.target](*func_args, **func_kwags)
                mapping_names_to_tensors[node.name] = tensor
            elif node.op == "output":
                return tuple(get_tensor_or_value(x) for x in node.args[0])

    op = MyMaxOp(
        max_add_i_want_to_use,
        max_add_i_want_to_use.__name__,
        CustomOpLibrary(KernelLibrary(mlir.Context())),
        input_types=None,
        output_types=None,
        num_outputs=get_the_number_of_outputs(gm),
        num_inputs=len(example_inputs)
    )
    custom_op_def = op.custom_op_def()

    def torch_add_with_max(*args) -> torch.Tensor:
        results = [torch.empty_like(x, device=args[0].device) for x in meta_outputs]
        custom_op_def(*results, *args)
        return results
    print("compiler done!:!!!!!!!!!")
    return torch_add_with_max


def fn(x, y, z):
    return x + y + z, x + torch.abs(z) - torch.cos(y) + 1


fn_compiled = torch.compile(backend=my_compiler)(fn)




a = torch.randn(3).to(device="cuda")
print(a)
b = torch.randn(3).to(device="cuda")
print(b)
c = torch.randn(3).to(device="cuda")
print(c)

outputs_no_compiled = fn(a, b, c)
outputs_compiled = fn_compiled(a, b, c)
print("out_no_compiled:", outputs_no_compiled)
print("out_   compiled:", outputs_compiled)
for out, out_compiled in zip(outputs_no_compiled, outputs_compiled):
    assert torch.allclose(out, out_compiled), "Outputs do not match!"
    assert out.device == out_compiled.device, "Devices do not match!"