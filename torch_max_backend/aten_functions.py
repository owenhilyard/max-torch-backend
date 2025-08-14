"""How to execute Pytorch's Aten functions using Max's backend.

The only ressources I could find on the subject are:
- https://github.com/pytorch/pytorch/blob/500cbb5b9013f842ba0a15ef61bbf0b079ed99ff/aten/src/ATen/native/native_functions.yaml
- https://docs.pytorch.org/docs/stable/torch.compiler_ir.html
"""

import operator
from max.torch.torch import max_device_ref

import max.graph.ops as max_ops
from max.dtype import DType
from torch.ops import aten
from torch_max_backend.mappings import (
    MAPPING_TORCH_TO_MOJO_FUNCTIONS as MAPPING_TORCH_ATEN_TO_MAX,
)
import torch

IDENTICAL_FUNCTIONS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    operator.mod,
    operator.matmul,
    operator.neg,
    operator.gt,
    operator.ge,
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.and_,
    operator.or_,
    operator.xor,
    operator.iadd,
    operator.isub,
    operator.imul,
    operator.ifloordiv,
    operator.ipow,
    operator.imod,
    operator.getitem,
    str,
    max,
    min,
]

for func in IDENTICAL_FUNCTIONS:
    MAPPING_TORCH_ATEN_TO_MAX[func] = func


def map_to(func: callable) -> callable:
    def decorator(func_to_map: callable) -> callable:
        MAPPING_TORCH_ATEN_TO_MAX[func] = func_to_map
        return func_to_map

    return decorator


# _adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
@map_to(aten._adaptive_avg_pool2d)
def aten__adaptive_avg_pool2d(input, output_size):
    # For now, we'll implement this using global average pooling for (1, 1) output
    # and regular avg pooling for other sizes
    if output_size == (1, 1) or output_size == 1:
        # Global average pooling - take mean over spatial dimensions
        return aten_mean(input, dim=(2, 3), keepdim=True)
    else:
        # For other output sizes, we'll use avg_pool2d with calculated kernel size and stride
        # Get input spatial dimensions (assuming NCHW format)
        input_h, input_w = input.shape[2], input.shape[3]

        if isinstance(output_size, int):
            output_h = output_w = output_size
        else:
            output_h, output_w = output_size

        # Calculate kernel size and stride to achieve the desired output size
        kernel_h = input_h // output_h
        kernel_w = input_w // output_w
        stride_h = input_h // output_h
        stride_w = input_w // output_w

        # Convert input from NCHW to NHWC for MAX
        input_nhwc = input.permute([0, 2, 3, 1])

        result = max_ops.avg_pool2d(
            input_nhwc,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=(0, 0),
            ceil_mode=False,
            count_boundary=True,
        )

        # Convert result back from NHWC to NCHW
        return result.permute([0, 3, 1, 2])


# _adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
# _adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor
# _cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor
# _embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
# _fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor
# _local_scalar_dense(Tensor self) -> Scalar
# _log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
# _native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
# _native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
# _native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)
# _pdist_forward(Tensor self, float p=2) -> Tensor
# _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
# _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
# abs(Tensor self) -> Tensor
# acos(Tensor self) -> Tensor
# acosh(Tensor self) -> Tensor
# adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor
# add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
# add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
# addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor


# alias(Tensor(a) self) -> Tensor(a)
@map_to(aten.alias)
def aten_alias(input):
    return input


# amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
# amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
# any(Tensor self) -> Tensor


@map_to(aten.any)
def aten_any(input, dim=None, keepdim=False, *, out=None):
    """
    Equivalent to torch.any.
    Tests if any elements in the input are True (non-zero).
    Uses max() on boolean tensor since True > False.
    """
    # Convert input to boolean first (non-zero values become True)
    input_bool = max_ops.not_equal(input, 0)

    if dim is None:
        # Return True if any element is True (reduce all dimensions)
        dim = tuple(range(len(input.shape)))
    elif isinstance(dim, int):
        dim = (dim,)

    # Handle negative dimensions
    dim = [x if x >= 0 else len(input.shape) + x for x in dim]

    result = input_bool
    # Use max() to implement any() since True > False
    for axis in sorted(dim, reverse=True):
        result = max_ops.max(result, axis=axis)

    # Handle keepdim=False
    if not keepdim:
        # Squeeze the reduced dimensions
        for axis in sorted(dim, reverse=True):
            result = max_ops.squeeze(result, axis=axis)

    return result


# any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
# any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
# arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
# argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
# as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)
# asin(Tensor self) -> Tensor
# asinh(Tensor self) -> Tensor
# atan(Tensor self) -> Tensor
# atan2(Tensor self, Tensor other) -> Tensor
# atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
# atanh(Tensor self) -> Tensor
# avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
# avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
# avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
# avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
# bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
# bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
# bitwise_not(Tensor self) -> Tensor
# bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
# bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
# bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
# bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
# bmm(Tensor self, Tensor mat2) -> Tensor
# cat(Tensor[] tensors, int dim=0) -> Tensor
# ceil(Tensor self) -> Tensor
# clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
# clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor


# clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
@map_to(aten.clone)
def aten_clone(input, *, memory_format=None):
    return input


# col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
# constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor
# convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
# convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
# copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor
# cos(Tensor self) -> Tensor
# cosh(Tensor self) -> Tensor
# cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
# diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
# div.Scalar(Tensor self, Scalar other) -> Tensor
# div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
# div.Tensor(Tensor self, Tensor other) -> Tensor
# div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
# elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
# embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
# embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) -> Tensor
# empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
# empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# eq.Scalar(Tensor self, Scalar other) -> Tensor
# eq.Tensor(Tensor self, Tensor other) -> Tensor
# erf(Tensor self) -> Tensor


# exp(Tensor self) -> Tensor
@map_to(aten.exp)
def aten_exp(input):
    return max_ops.exp(input)


# expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
@map_to(aten.expand)
def torch_aten_expand_equivalent(tensor, size: list[int]):
    target_shape = []

    # Get current tensor shape - we need this to handle -1 values
    current_shape = tensor.shape

    # Pad the current shape with 1s if target has more dimensions
    if len(size) > len(current_shape):
        padded_current_shape = [1] * (len(size) - len(current_shape)) + list(
            current_shape
        )
    else:
        padded_current_shape = list(current_shape)

    # Process each dimension in the target size
    for i, dim_size in enumerate(size):
        if dim_size == -1:
            # Keep current dimension size
            if i < len(padded_current_shape):
                target_shape.append(padded_current_shape[i])
            else:
                # This shouldn't happen in well-formed expand calls
                target_shape.append(1)
        else:
            target_shape.append(dim_size)

    return max_ops.broadcast_to(tensor, target_shape)


# expm1(Tensor self) -> Tensor
# fill.Scalar(Tensor self, Scalar value) -> Tensor
# flip(Tensor self, int[] dims) -> Tensor
# floor(Tensor self) -> Tensor
# fmod.Scalar(Tensor self, Scalar other) -> Tensor
# fmod.Tensor(Tensor self, Tensor other) -> Tensor
# full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
# gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
# ge.Scalar(Tensor self, Scalar other) -> Tensor
# ge.Tensor(Tensor self, Tensor other) -> Tensor
# gelu(Tensor self, *, str approximate=’none’) -> Tensor
# grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
# gt.Scalar(Tensor self, Scalar other) -> Tensor
# gt.Tensor(Tensor self, Tensor other) -> Tensor
# hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor


# index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
@map_to(aten.index)
def aten_index(input, indices=None):
    if not indices:
        raise NotImplementedError("We don't yet support aten.index without indices")
    if len([i for i in indices if i is not None]) != 1:
        raise NotImplementedError(
            "We only support aten.index with a single non-None index"
        )

    for i, index in enumerate(indices):
        if index is None:
            continue
        return max_ops.gather(input, index, axis=i)


# index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
# index_select(Tensor self, int dim, Tensor index) -> Tensor
# isinf(Tensor self) -> Tensor
# isnan(Tensor self) -> Tensor
# le.Scalar(Tensor self, Scalar other) -> Tensor
# le.Tensor(Tensor self, Tensor other) -> Tensor
# leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
# log(Tensor self) -> Tensor
# log10(Tensor self) -> Tensor
# log1p(Tensor self) -> Tensor
# log2(Tensor self) -> Tensor


# logical_and(Tensor self, Tensor other) -> Tensor
@map_to(aten.logical_and)
def aten_logical_and(input, other):
    """
    Computes element-wise logical AND of two tensors.
    Both inputs are converted to boolean first if they aren't already.
    """
    # Convert both inputs to boolean if they aren't already
    if input.dtype != DType.bool:
        input_bool = max_ops.not_equal(input, 0)
    else:
        input_bool = input

    if other.dtype != DType.bool:
        other_bool = max_ops.not_equal(other, 0)
    else:
        other_bool = other

    # Apply logical and
    return max_ops.logical_and(input_bool, other_bool)


# logical_not(Tensor self) -> Tensor
@map_to(aten.logical_not)
def aten_logical_not(input):
    """
    PyTorch's logical_not treats any non-zero value as True and returns the logical negation.
    MAX's logical_not requires boolean input, so we need to convert first.
    """
    # Convert input to boolean (non-zero -> True, zero -> False)
    input_bool = max_ops.not_equal(input, 0)
    # Apply logical not
    return max_ops.logical_not(input_bool)


# logical_or(Tensor self, Tensor other) -> Tensor
# logical_xor(Tensor self, Tensor other) -> Tensor
# lt.Scalar(Tensor self, Scalar other) -> Tensor
# lt.Tensor(Tensor self, Tensor other) -> Tensor
# masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
# max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)


# max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
@map_to(aten.max_pool2d_with_indices)
def aten_max_pool2d_with_indices(
    input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
) -> tuple[max_ops.TensorType, max_ops.TensorType]:
    # the first output is the values, the second output is the indices
    # most of the time people just want the values so we'll implement that
    # for now.
    if not stride:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Convert input from NCHW (PyTorch default) to NHWC (MAX requirement)
    input_nhwc = input.permute([0, 2, 3, 1])

    result = max_ops.max_pool2d(
        input_nhwc,
        kernel_size=kernel_size,
        stride=tuple(stride),
        padding=tuple(padding),
        dilation=tuple(dilation),
        ceil_mode=ceil_mode,
    )

    # Convert result back from NHWC to NCHW for PyTorch compatibility
    forward_result = result.permute([0, 3, 1, 2])
    # TODO: Add indices
    return (forward_result,)


# max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor
# max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
# maximum(Tensor self, Tensor other) -> Tensor


# mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
# mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
@map_to(aten.mean)
def aten_mean(input, dim=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        max_dtype = DType.from_torch(dtype)
        input = max_ops.cast(input, dtype=max_dtype)

    result = input

    if dim is None:
        dim = tuple(range(len(input.shape)))
    elif isinstance(dim, int):
        dim = (dim,)

    dim = [x if x >= 0 else len(input.shape) + x for x in dim]

    # Multiple dimensions reduction - reduce each dimension one by one
    # Sort dimensions in descending order to avoid index shifting issues
    for axis in dim:
        result = max_ops.mean(result, axis=axis)

    # Handle keepdim=False - MAX's mean keeps dimensions by default, so we need to squeeze
    if not keepdim:
        # Remove multiple dimensions - need to be careful about index shifting
        # Sort original dimensions and squeeze from highest to lowest
        dims_to_squeeze = sorted(dim, reverse=True)
        for axis in dims_to_squeeze:
            result = max_ops.squeeze(result, axis=axis)

    return result


# min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
# minimum(Tensor self, Tensor other) -> Tensor
# mm(Tensor self, Tensor mat2) -> Tensor
# mul.Scalar(Tensor self, Scalar other) -> Tensor
# mul.Tensor(Tensor self, Tensor other) -> Tensor
# native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)


# native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
@map_to(aten.native_group_norm)
def aten_native_group_norm(input, weight, bias, N, C, HxW, group, eps):
    """
    This is the low-level operation that F.group_norm gets compiled to.
    Returns (normalized_output, mean, rstd) tuple but we only return the first element for simplicity.
    """
    # Reshape input from [N*C, HxW] back to [N, C, H, W] format
    # First, calculate H and W from HxW
    HW = int(HxW)
    # For simplicity, assume square spatial dimensions
    H = W = int(HW**0.5)
    if H * W != HW:
        # If not square, try to factor HxW into reasonable H and W
        # For now, use 1D spatial dimension
        H, W = HW, 1

    # Reshape input to [N, C, H, W]
    input_reshaped = max_ops.reshape(input, [int(N), int(C), H, W])

    # Use the regular group_norm implementation
    result = torch_group_norm_equivalent(input_reshaped, group, weight, bias, eps)

    # Return just the normalized output (native_group_norm returns a tuple)
    return (result,)


def torch_group_norm_equivalent(input, num_groups, weight=None, bias=None, eps=1e-5):
    # input shape: [N, C, H, W]
    N, C, H, W = input.shape

    # Ensure number of channels is divisible by number of groups
    if int(C) % num_groups != 0:
        raise ValueError(
            f"Number of channels ({C}) must be divisible by number of groups ({num_groups})"
        )

    channels_per_group = int(C) // num_groups

    # Reshape input to [N, num_groups, channels_per_group, H, W]
    reshaped = max_ops.reshape(
        input, [int(N), num_groups, channels_per_group, int(H), int(W)]
    )

    # Calculate mean and variance for each group
    # Normalize over dimensions: channels_per_group, H, W (dims 2, 3, 4)
    axis_to_reduce = [2, 3, 4]

    # Calculate mean
    mean = aten_mean(reshaped, dim=axis_to_reduce, keepdim=True)

    # Calculate variance: Var(X) = E[(X - mean)^2]
    centered = reshaped - mean
    variance = aten_mean(centered * centered, dim=axis_to_reduce, keepdim=True)

    # Normalize: (x - mean) / sqrt(variance + eps)
    normalized = centered / max_ops.sqrt(variance + eps)

    # Reshape back to original shape [N, C, H, W]
    normalized = max_ops.reshape(normalized, [int(N), int(C), int(H), int(W)])

    # Apply scale and shift if provided
    if weight is not None:
        # weight shape: [C] - broadcast to [N, C, H, W]
        weight_reshaped = max_ops.reshape(weight, [1, int(C), 1, 1])
        normalized = normalized * weight_reshaped

    if bias is not None:
        # bias shape: [C] - broadcast to [N, C, H, W]
        bias_reshaped = max_ops.reshape(bias, [1, int(C), 1, 1])
        normalized = normalized + bias_reshaped

    return normalized


# native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)


# native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
@map_to(aten.native_layer_norm)
def aten_native_layer_norm(input, normalized_shape, weight, bias, eps):
    # expects a tuple or list for some reason
    # surely for the backward pass,
    # for the moment we only output the first one.
    # Layer norm normalizes over the last len(normalized_shape) dimensions
    # Calculate mean and variance over these dimensions
    axis_to_reduce = list(
        range(len(input.shape) - len(normalized_shape), len(input.shape))
    )

    # Calculate mean
    mean = aten_mean(input, dim=axis_to_reduce, keepdim=True)

    # Calculate variance: Var(X) = E[(X - mean)^2]
    centered = input - mean
    variance = aten_mean(centered * centered, dim=axis_to_reduce, keepdim=True)

    # Normalize: (x - mean) / sqrt(variance + eps)
    normalized = centered / max_ops.sqrt(variance + eps)

    # Apply scale and shift if provided
    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias

    # TODO: Add the other outputs later
    return (normalized,)


# native_layer_norm_backward(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)


# ne.Scalar(Tensor self, Scalar other) -> Tensor
# ne.Tensor(Tensor self, Tensor other) -> Tensor
# neg(Tensor self) -> Tensor
# nonzero(Tensor self) -> Tensor
# permute(Tensor(a) self, int[] dims) -> Tensor(a)
# pow.Scalar(Scalar self, Tensor exponent) -> Tensor
# pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
# pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
# prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
# prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
# rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# randperm(SymInt n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# reciprocal(Tensor self) -> Tensor
# reflection_pad1d(Tensor self, SymInt[2] padding) -> Tensor
# reflection_pad2d(Tensor self, SymInt[4] padding) -> Tensor
# reflection_pad3d(Tensor self, SymInt[6] padding) -> Tensor
# relu(Tensor self) -> Tensor
# remainder.Scalar(Tensor self, Scalar other) -> Tensor
# remainder.Tensor(Tensor self, Tensor other) -> Tensor
# repeat(Tensor self, SymInt[] repeats) -> Tensor
# replication_pad2d(Tensor self, SymInt[4] padding) -> Tensor
# replication_pad3d(Tensor self, SymInt[6] padding) -> Tensor
# resize_(Tensor(a!) self, SymInt[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
# round(Tensor self) -> Tensor
# rsqrt(Tensor self) -> Tensor


# scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
@map_to(aten.scalar_tensor)
def aten_scalar_tensor(
    value: float | int,
    dtype: torch.dtype = None,
    layout: torch.layout = None,
    device: torch.device = None,
):
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.get_default_device()

    return max_ops.constant(
        value, dtype=DType.from_torch(dtype), device=max_device_ref(device)
    )


# scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
# scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
# scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
# scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor
# select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
# select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor


# sigmoid(Tensor self) -> Tensor
@map_to(aten.sigmoid)
def aten_sigmoid(input):
    return max_ops.sigmoid(input)


# sign(Tensor self) -> Tensor
# sin(Tensor self) -> Tensor
# sinh(Tensor self) -> Tensor


# slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
@map_to(aten.slice)
def aten_slice(input, dim, start: int, end: int, step: int = 1):
    if end == 2**63 - 1:  # MAX_INT64
        end = None
    slices = [slice(None)] * len(input.shape)
    slices[dim] = slice(start, end, step)
    return input[*slices]


# slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
# sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)


# split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
@map_to(aten.split_with_sizes)
def aten_split_with_sizes(input, split_sizes, dim=0):
    result = []
    start = 0
    for size in split_sizes:
        end = start + size
        result.append(aten_slice(input, dim, start, end))
        start = end
    return result


# sqrt(Tensor self) -> Tensor
# squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
# squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)
# sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
# sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
# sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
# sym_numel(Tensor self) -> SymInt
# sym_size.int(Tensor self, int dim) -> SymInt
# sym_storage_offset(Tensor self) -> SymInt
# sym_stride.int(Tensor self, int dim) -> SymInt
# tan(Tensor self) -> Tensor
# tanh(Tensor self) -> Tensor
# topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
# trunc(Tensor self) -> Tensor
# unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
# upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
# upsample_nearest2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
# var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
# var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
# view(Tensor(a) self, SymInt[] size) -> Tensor(a)


# where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
@map_to(aten.where)
def torch_aten_where_equivalent(input, condition, other):
    return max_ops.where(input, condition, other)
