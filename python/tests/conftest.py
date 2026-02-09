import pytest
import torch
import infinitensor
from infinitensor import Runtime, DeviceType


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=[
            "cpu",
            "cuda",
            "mlu",
            "metax",
            "moore",
            "iluvatar",
            "kunlun",
            "hygon",
            "ascend",
        ],
        help="Device to run tests on: cpu, cuda, mlu, metax, moore, iluvatar, kunlun, hygon, ascend",
    )
    parser.addoption(
        "--device-id", action="store", default="0", help="Device ID to use"
    )


@pytest.fixture(scope="session")
def device_type(request):
    """Get device type from command line arguments"""
    device_name = request.config.getoption("--device").upper()

    device_map = {
        "CPU": DeviceType.CPU,
        "CUDA": DeviceType.CUDA,
        "MLU": DeviceType.MLU,
        "ASCEND": DeviceType.ASCEND,
        "METAX": DeviceType.METAX,
        "MOORE": DeviceType.MOORE,
        "ILUVATAR": DeviceType.ILUVATAR,
        "KUNLUN": DeviceType.KUNLUN,
        "HYGON": DeviceType.HYGON,
    }

    return device_map.get(device_name, DeviceType.CPU)


@pytest.fixture(scope="session")
def device_id(request):
    """Get device ID from command line arguments"""
    return int(request.config.getoption("--device-id"))


@pytest.fixture(scope="session")
def runtime(device_type, device_id):
    """Create runtime based on command line arguments"""
    print(f"\n{'='*60}")
    print(f"Creating runtime with:")
    print(f"  Device Type: {device_type}")
    print(f"  Device ID:   {device_id}")
    print(f"{'='*60}\n")

    try:
        # If there is an API to set device ID
        rt = Runtime.setup(device_type, device_id=device_id)
        print(f"✅ Runtime created successfully")
        return rt
    except Exception as e:
        print(f"❌ Failed to create runtime: {e}")

        # Fall back to CPU
        if device_type != DeviceType.CPU:
            print("🔄 Falling back to CPU...")
            return Runtime.setup(DeviceType.CPU)
        else:
            raise pytest.skip(f"Device {device_type} not available")


@pytest.fixture
def torch_rng_seed():
    """Fix random seed to ensure test reproducibility"""
    torch.manual_seed(42)
    yield 42
