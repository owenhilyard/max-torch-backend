
import torch
import max
from max.torch.torch import graph_op, MaxOp, CustomOpLibrary
import inspect
from max.graph import KernelLibrary
from max import mlir

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


def my_compiler(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    gm.graph.print_tabular()

    def max_add_i_want_to_use(*args):
        return sum(args)

    op = MyMaxOp(
        max_add_i_want_to_use,
        max_add_i_want_to_use.__name__,
        CustomOpLibrary(KernelLibrary(mlir.Context())),
        input_types=None,
        output_types=None,
        num_outputs=1,
        num_inputs=len(example_inputs)
    )
    custom_op_def = op.custom_op_def()

    def torch_add_with_max(*args) -> torch.Tensor:
        result = args[0].new_empty(args[0].shape)
        custom_op_def(result, *args)
        return [result]

    return torch_add_with_max


@torch.compile(backend=my_compiler)
def fn(x, y):
    return x + y


a = torch.randn(3)
print(a)
b = torch.randn(3)
print(b)

print(fn(a, b))
