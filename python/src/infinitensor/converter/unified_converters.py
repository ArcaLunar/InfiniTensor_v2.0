import torch.nn as nn
from .registry import registry

# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml


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


@registry.register("clip", "Tensor")
def convert_clip(translator, node):
    input = translator.tensors[node.args[0]]
    min_val = translator.tensors[node.args[1]]
    max_val = translator.tensors[node.args[2]]
    translator.tensors[node] = translator.builder.clip(input, min_val, max_val, None)


@registry.register("layer_norm", "default")
def convert_layer_norm(translator, node):
    # aten.layer_norm.default usually has args:
    # (input, normalized_shape, weight, bias, eps, cudnn_enable)
    input = translator.tensors[node.args[0]]
    normalized_shape = node.args[1]  # not used since we default to normalizing the last dimension
    weight = translator.tensors[node.args[2]]
    bias = translator.tensors[node.args[3]]
    eps = node.args[4] if len(node.args) >= 5 else 1e-5

    # LayerNorm defaults to normalizing the last dimension in backend.
    translator.tensors[node] = translator.builder.layernorm(input, weight, bias, 1e-5, None)
