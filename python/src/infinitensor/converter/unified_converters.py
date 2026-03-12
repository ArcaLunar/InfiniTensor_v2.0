import torch
import torch.nn as nn
from pyinfinitensor import dtype_from_string, ShapeExpr
from .registry import registry

# Reference: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml

# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------

@registry.register("matmul", "default")
def convert_matmul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.gemm(a, b, None)


@registry.register("add", "Tensor")
def convert_add(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.add(a, b, None)


@registry.register("mul", "Tensor")
def convert_mul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.mul(a, b, None)


@registry.register("sub", "Tensor")
def convert_sub(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.sub(a, b, None)


# ---------------------------------------------------------------------------
# Unary activations
# ---------------------------------------------------------------------------

@registry.register("relu", "default")
def convert_relu(translator, node):
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.relu(x)


@registry.register("sigmoid", "default")
def convert_sigmoid(translator, node):
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.sigmoid(x)


@registry.register("silu", "default")
def convert_silu(translator, node):
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.silu(x)


@registry.register("gelu", "default")
def convert_gelu(translator, node):
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.gelu(x)


@registry.register("tanh", "default")
def convert_tanh(translator, node):
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.tanh(x)


@registry.register("softplus", "default")
def convert_softplus(translator, node):
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.softplus(x)


# ---------------------------------------------------------------------------
# Softmax / LogSoftmax
# aten._softmax / aten._log_softmax are the lowered forms after torch.export
# ---------------------------------------------------------------------------

@registry.register("_softmax", "default")
def convert_softmax(translator, node):
    # args: (input, dim, half_to_float)
    x = translator.tensors[node.args[0]]
    axis = int(node.args[1])
    translator.tensors[node] = translator.builder.softmax(x, axis)


@registry.register("_log_softmax", "default")
def convert_log_softmax(translator, node):
    # args: (input, dim, half_to_float)
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.log_softmax(x)


# ---------------------------------------------------------------------------
# Convolution
# aten.convolution.default: (input, weight, bias, stride, padding, dilation,
#                             transposed, output_padding, groups)
# ---------------------------------------------------------------------------

@registry.register("conv2d", "default")
def convert_conv(translator, node):
    args = node.args
    x = translator.tensors[args[0]]
    w = translator.tensors[args[1]]
    bias = translator.tensors[args[2]] if args[2] is not None else None
    strides = list(args[3])
    pads = list(args[4])
    dilations = list(args[5])
    # args[6] = transposed — skip (not supported by ConvObj)
    translator.tensors[node] = translator.builder.conv(x, w, bias, pads, strides, dilations)


# ---------------------------------------------------------------------------
# Normalization
# aten.layer_norm.default: (input, normalized_shape, weight, bias, eps, cudnn)
# aten.rms_norm.default:   (input, normalized_shape, weight, eps)
# ---------------------------------------------------------------------------

@registry.register("layer_norm", "default")
def convert_layer_norm(translator, node):
    args = node.args
    x = translator.tensors[args[0]]
    # args[1] = normalized_shape (list of ints) — determines axis
    weight = translator.tensors[args[2]] if args[2] is not None else None
    bias_t = translator.tensors[args[3]] if args[3] is not None else None
    eps = float(args[4]) if len(args) > 4 and args[4] is not None else 1e-5
    # axis = rank - len(normalized_shape)
    rank = len(x.shape())
    axis = rank - len(args[1])
    translator.tensors[node] = translator.builder.layer_norm(
        x, weight, bias_t, axis=axis, eps=eps
    )


@registry.register("rms_norm", "default")
def convert_rms_norm(translator, node):
    args = node.args
    x = translator.tensors[args[0]]
    weight = translator.tensors[args[2]] if len(args) > 2 and args[2] is not None else None
    eps = float(args[3]) if len(args) > 3 and args[3] is not None else 1e-6
    translator.tensors[node] = translator.builder.rms_norm(x, weight, epsilon=eps)


# ---------------------------------------------------------------------------
# Clamp / Clip
# aten.clamp.default: (input, min=None, max=None)
# We create scalar tensors for min/max bounds.
# ---------------------------------------------------------------------------

def _make_scalar_tensor(translator, value, ref_tensor):
    """Create a scalar (shape=[]) tensor with the same dtype as ref_tensor."""
    dtype = ref_tensor.dtype()
    t = translator.builder.tensor(ShapeExpr([]), dtype)
    buf = torch.tensor(value).to(
        dtype=getattr(torch, dtype_from_string.__module__.split('.')[0])
    )
    import ctypes
    # Create a 1-element torch tensor, set its data into the scalar tensor
    scalar_torch = torch.tensor(float(value))
    t.set_data(scalar_torch.data_ptr(), translator.runtime)
    return t


@registry.register("clamp", "default")
def convert_clamp(translator, node):
    args = node.args
    kwargs = node.kwargs
    x = translator.tensors[args[0]]
    min_val = args[1] if len(args) > 1 else kwargs.get("min", None)
    max_val = args[2] if len(args) > 2 else kwargs.get("max", None)
    min_t = _make_scalar_tensor(translator, min_val, x) if min_val is not None else None
    max_t = _make_scalar_tensor(translator, max_val, x) if max_val is not None else None
    translator.tensors[node] = translator.builder.clip(x, min_t, max_t)


@registry.register("clip", "default")
def convert_clip(translator, node):
    args = node.args
    x = translator.tensors[args[0]]
    min_t = translator.tensors[args[1]] if len(args) > 1 and args[1] is not None else None
    max_t = translator.tensors[args[2]] if len(args) > 2 and args[2] is not None else None
    translator.tensors[node] = translator.builder.clip(x, min_t, max_t)


# ---------------------------------------------------------------------------
# LP-norm
# aten.linalg_vector_norm.default: (input, ord=2, dim=None, keepdim=False, dtype=None)
# ---------------------------------------------------------------------------

@registry.register("linalg_vector_norm", "default")
def convert_lp_norm(translator, node):
    args = node.args
    kwargs = node.kwargs
    x = translator.tensors[args[0]]
    p = int(args[1]) if len(args) > 1 and args[1] is not None else 2
    dim = args[2] if len(args) > 2 and args[2] is not None else kwargs.get("dim", 0)
    if isinstance(dim, (list, tuple)):
        dim = dim[0]
    translator.tensors[node] = translator.builder.lp_norm(x, int(dim), p)