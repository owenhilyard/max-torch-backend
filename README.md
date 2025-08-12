# MAX-Torch Backend

Simple use [`torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), but with Modular's MAX backend.

## Installation

Ensure you have the required dependencies:

```bash
pip install git+https://github.com/gabrieldemarmiesse/max-torch-backend.git
```

**Requirements**: Python ≥3.12 and Modular's nightly Python index.

## Quick Start

### Basic Usage

```python
from max_torch_backend import MaxCompiler
import torch

# Compile your model with MAX backend
model = YourModel()
compiled_model = torch.compile(model, backend=MaxCompiler)

# Use normally - now accelerated by MAX
output = compiled_model(input_tensor)
```

### Simple Function Example

```python
import torch
from max_torch_backend import MaxCompiler

@torch.compile(backend=MaxCompiler)
def simple_math(x, y):
    return x + y * 2

# Usage
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(simple_math(a, b))  # Accelerated execution
```

### Device Selection

Note that currently the MAX backend does not support some older nvidia/amd gpus.
So you'll need to ask MAX first if your GPU is supported before 
using the gpu.

```python
from max_torch_backend import get_accelerators

# Check available accelerators
# The CPU is necessarily included in the list of accelerators
accelerators = get_accelerators()
device = "cuda" if len(list(accelerators)) >= 2 else "cpu"
model = model.to(device)
```

## Supported Operations

The backend currently supports every op listed in [`mappings.py`](https://github.com/gabrieldemarmiesse/max-torch-backend/blob/main/max_torch_backend/mappings.py)

## Performance Tips

### Dynamic Shapes
For variable input sizes, mark dynamic dimensions to avoid recompiling:

```python
from torch._dynamo import mark_dynamic

mark_dynamic(input_tensor, 0)  # batch dimension
mark_dynamic(input_tensor, 1)  # sequence length
```

If you don't do so, Pytorch will compile a second time when it sees a different shape, which can be costly.
You can find more information about dynamic shapes in the [PyTorch documentation](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html).

### Compilation Strategy
- Use `fullgraph=True` when possible for better optimization

## Testing

```bash
# Run all tests (the first time is slow, chaching kicks in after)
uv run pytest -v -n 5

# Lint and format
uvx pre-commit run --all-files
# Or install the pre-commit hook
uvx pre-commit install
```
