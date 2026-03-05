import pytest
import torch
import torch.nn as nn
import numpy as np
import infinitensor
from infinitensor import TorchFXTranslator, Runtime, DeviceType


def test_basic_matmul(runtime, torch_rng_seed):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class MatmulModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    model = MatmulModel()
    # Randomly initialize inputs, passed shapes can differ from actual values, but data types must match
    input_info = [((5, 4), "float32"), ((4, 3), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)
    # Run
    translator.run(input_tensors)

    # Get outputs
    outputs = translator.get_outputs()

    # Verify
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 5, 3)
    print("✅ Test passed!")


def test_dynamic_matmul(runtime, torch_rng_seed):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class MatmulModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    model = MatmulModel()
    # Randomly initialize inputs, passed shapes can differ from actual values, but data types must match
    input_info = [((5, 4), "float32"), ((4, 7), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    input_info_1 = [((15, 4), "float32"), ((4, 12), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (1, 15, 12)

    input_info_2 = [((3, 20), "float32"), ((20, 10), "float32")]
    input_tensors_2 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_2
    ]
    translator.run(input_tensors_2)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (1, 3, 10)
    print("✅ Test passed!")


def test_basic_elementwise(runtime, torch_rng_seed):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class AddModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = AddModel()
    # Randomly initialize inputs, passed shapes can differ from actual values, but data types must match
    input_info = [((5, 4), "float32"), ((3, 5, 1), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    translator.run(input_tensors)
    # Get outputs
    outputs = translator.get_outputs()

    # Verify
    assert len(outputs) == 1
    assert outputs[0].shape == (3, 5, 4)
    print("✅ Test passed!")


# ---------------------------------------------------------------------------
# Unary activation tests
# ---------------------------------------------------------------------------

class _UnaryModel(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, x):
        return self._fn(x)


UNARY_CASES = [
    ("relu",     lambda x: torch.relu(x)),
    ("sigmoid",  lambda x: torch.sigmoid(x)),
    ("tanh",     lambda x: torch.tanh(x)),
    ("silu",     lambda x: torch.nn.functional.silu(x)),
    ("gelu",     lambda x: torch.nn.functional.gelu(x)),
    ("softplus", lambda x: torch.nn.functional.softplus(x)),
]


@pytest.mark.parametrize("name,fn", UNARY_CASES, ids=[c[0] for c in UNARY_CASES])
def test_unary_ops(runtime, torch_rng_seed, name, fn):
    """Verify TorchFX translation of all 6 unary activations."""
    model = _UnaryModel(fn)
    inputs = [torch.randn(3, 4)]
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inputs)
    translator.run(inputs)
    outputs = translator.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].shape == (3, 4), f"{name}: unexpected shape {outputs[0].shape}"


# ---------------------------------------------------------------------------
# Softmax / LogSoftmax
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=False,
    reason="torch.export may emit softmax.int instead of _softmax.default "
           "depending on PyTorch version; translator limitation",
)
def test_softmax(runtime, torch_rng_seed):
    class SoftmaxModel(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=-1)

    model = SoftmaxModel()
    inputs = [torch.randn(4, 8)]
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inputs)
    translator.run(inputs)
    outputs = translator.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].shape == (4, 8)


@pytest.mark.xfail(
    strict=False,
    reason="torch.export may emit log_softmax.int instead of _log_softmax.default; translator limitation",
)
def test_log_softmax(runtime, torch_rng_seed):
    class LogSoftmaxModel(torch.nn.Module):
        def forward(self, x):
            return torch.log_softmax(x, dim=1)

    model = LogSoftmaxModel()
    inputs = [torch.randn(2, 10)]
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inputs)
    translator.run(inputs)
    outputs = translator.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].shape == (2, 10)


# ---------------------------------------------------------------------------
# Sub / Mul elementwise
# ---------------------------------------------------------------------------

def test_basic_sub(runtime, torch_rng_seed):
    class SubModel(torch.nn.Module):
        def forward(self, x, y):
            return x - y

    model = SubModel()
    inputs = [torch.randn(3, 4), torch.randn(3, 4)]
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inputs)
    translator.run(inputs)
    outputs = translator.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].shape == (3, 4)


def test_basic_mul(runtime, torch_rng_seed):
    class MulModel(torch.nn.Module):
        def forward(self, x, y):
            return x * y

    model = MulModel()
    inputs = [torch.randn(3, 4), torch.randn(3, 4)]
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inputs)
    translator.run(inputs)
    outputs = translator.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].shape == (3, 4)


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=False,
    reason="clamp.default converter requires torch.pyinfinitensor attribute; known translator limitation",
)
def test_clip(runtime, torch_rng_seed):
    class ClipModel(torch.nn.Module):
        def forward(self, x):
            return torch.clamp(x, min=-1.0, max=1.0)

    model = ClipModel()
    inputs = [torch.randn(4, 4)]
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inputs)
    translator.run(inputs)
    outputs = translator.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].shape == (4, 4)


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=False,
    reason="Dynamic stride symbolic dimension assertion in _process_dynamic_shapes; translator limitation",
)
def test_layer_norm(runtime, torch_rng_seed):
    class LNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.LayerNorm(8)

        def forward(self, x):
            return self.ln(x)

    model = LNModel()
    inputs = [torch.randn(2, 4, 8)]
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inputs)
    translator.run(inputs)
    outputs = translator.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].shape == (2, 4, 8)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=False,
    reason="Dynamic stride symbolic dimension assertion in _process_dynamic_shapes; translator limitation",
)
def test_rms_norm(runtime, torch_rng_seed):
    class RMSModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rms = torch.nn.RMSNorm(8)

        def forward(self, x):
            return self.rms(x)

    model = RMSModel()
    inputs = [torch.randn(2, 4, 8)]
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inputs)
    translator.run(inputs)
    outputs = translator.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].shape == (2, 4, 8)


# ---------------------------------------------------------------------------
# LPNorm (torch.linalg.vector_norm)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=False,
    reason="LPNorm infiniop CPU backend not supported (error 5); compute limitation",
)
def test_lp_norm(runtime, torch_rng_seed):
    class LPModel(torch.nn.Module):
        def forward(self, x):
            return torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

    model = LPModel()
    inputs = [torch.randn(4, 8)]
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inputs)
    translator.run(inputs)
    outputs = translator.get_outputs()
    assert len(outputs) == 1


# ---------------------------------------------------------------------------
# Conv
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=False,
    reason="Dynamic stride symbolic dimension assertion in _process_dynamic_shapes; translator limitation",
)
def test_conv2d(runtime, torch_rng_seed):
    class ConvModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)

        def forward(self, x):
            return self.conv(x)

    model = ConvModel()
    inputs = [torch.randn(1, 1, 8, 8)]
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inputs)
    translator.run(inputs)
    outputs = translator.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 4, 8, 8)


if __name__ == "__main__":
    # Can run this file directly
    import sys

    # Run all tests using pytest
    exit_code = pytest.main(
        [
            __file__,
            "-v",  # Verbose output
            "-s",  # Show print output
            "--tb=short",  # Simplified error traceback
        ]
    )

    sys.exit(0 if exit_code == 0 else 1)
