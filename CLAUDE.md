# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch backend implementation using Modular's MAX framework. The project demonstrates how to create custom PyTorch compilation backends that bridge PyTorch operations to MAX/Mojo implementations.

## Dependencies and Setup

- **Python**: >=3.12 required
- **Key Dependencies**: 
  - `modular>=25.4.0` (from Modular's nightly index)
  - `torch>=2.7.0`
  - `tabulate>=0.9.0`
- **Package Manager**: Uses `uv` for dependency management

## Common Commands

```bash
# Install dependencies
uv install

# Run the main script
python main.py

# Install in development mode (if needed)
uv pip install -e .
```

## Architecture

The project implements a custom PyTorch compiler backend (`my_compiler`) that:

1. **Graph Analysis**: Takes PyTorch FX graphs and analyzes their structure
2. **Operation Mapping**: Maps PyTorch operations to Mojo/MAX equivalents via `MAPPING_TORCH_TO_MOJO_FUNCTIONS`
3. **Custom Operations**: Creates MAX custom operations (`MyMaxOp`) that wrap the compiled functions
4. **Runtime Bridge**: Provides a bridge between PyTorch tensors and MAX execution

### Key Components

- **`MyMaxOp`**: Custom MAX operation class that defines torch signatures dynamically based on input/output counts
- **`my_compiler`**: Main compiler function that processes FX graphs and returns compiled functions
- **Function Mapping**: Dictionary mapping PyTorch operations (like `operator.add`) to Mojo implementations

### Compilation Flow

1. PyTorch function decorated with `@torch.compile(backend=my_compiler)`
2. FX graph generated and passed to `my_compiler`
3. Graph nodes processed to create runtime function (`max_add_i_want_to_use`)
4. Custom MAX operation created and wrapped for PyTorch compatibility
5. Compiled function returns multiple outputs as tuples

## Current Implementation

The example implements addition operations, demonstrating compilation of functions that return multiple outputs (e.g., `x + y + z, x + z`).