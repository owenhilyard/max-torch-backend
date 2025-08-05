
import torch
import max
from max.torch import graph_op

print("imports done")

def max_add(a, b):
    return a + b

def max_add_i_want_to_use(*args):
    return args[0] + args[1]

max_add_graph_op = graph_op(max_add)


def torch_add_with_max(*args) -> torch.Tensor:
    result = args[0].new_empty(args[0].shape)
    max_add_graph_op(result, args[0], args[1])
    return [result]

def my_compiler(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    gm.graph.print_tabular()
    return torch_add_with_max  # return a python callable

@torch.compile(backend=my_compiler)
def fn(x, y):
    return x + y

a = torch.randn(3)
print(a)
b = torch.randn(3)
print(b)

print(fn(a, b))
