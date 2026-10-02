import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.engine.accelerator import (
    CUDABackend,
    BaseAccelerator,
    is_cuda_available,
    load_cuda_shader,
    get_best_backend,
    CUDA_KERNEL_SOURCE,
)
from src.profiler.gpu import get_cuda_telemetry


def test_cuda_shader_source_integrity():
    """Verify that CUDA shader and kernel source are loaded and contain required symbols."""
    shader_str = load_cuda_shader()
    assert isinstance(shader_str, str)
    assert len(shader_str) > 0
    assert "apply_gate_1q_cuda" in shader_str
    assert "apply_gate_2q_cuda" in shader_str
    assert "cuDoubleComplex" in shader_str
    assert "c_mul" in shader_str
    assert "__global__" in shader_str


def test_cuda_backend_availability_graceful():
    """Verify that CUDABackend.is_available() executes safely without raising exceptions."""
    backend = CUDABackend()
    avail = backend.is_available()
    assert isinstance(avail, bool)
    assert is_cuda_available() == avail


def test_cuda_telemetry_graceful():
    """Verify get_cuda_telemetry returns valid dictionary structure under any environment."""
    telemetry = get_cuda_telemetry()
    assert isinstance(telemetry, dict)
    assert "available" in telemetry
    assert "device_name" in telemetry
    assert "device_count" in telemetry
    assert "vram_total_mb" in telemetry
    assert "telemetry_source" in telemetry

    # Verify device_info method on CUDABackend
    backend = CUDABackend()
    info = backend.get_device_info()
    assert isinstance(info, dict)
    assert info.get("available") == telemetry["available"]


def test_cuda_fallback_routing_when_unavailable():
    """Verify get_best_backend('cuda') falls back cleanly to available CPU/Metal tier."""
    with patch("src.engine.accelerator.is_cuda_available", return_value=False):
        backend = get_best_backend(requested="cuda")
        assert isinstance(backend, BaseAccelerator)
        # Should gracefully degrade to Metal, C++, or NumPy
        assert backend.name in ("Metal", "C++", "NumPy")
        assert backend.name != "CUDA"


def test_cuda_fallback_routing_when_available():
    """Verify get_best_backend('cuda') selects CUDABackend when CUDA is available."""
    with patch("src.engine.accelerator.is_cuda_available", return_value=True):
        backend = get_best_backend(requested="cuda")
        assert isinstance(backend, CUDABackend)
        assert backend.name == "CUDA"


def test_cuda_mocked_statevector_allocation():
    """Verify statevector allocation and initialization in host/fallback mode."""
    backend = CUDABackend()
    # 2 qubits -> 4 complex128 states
    state = backend.allocate_statevector(2)
    assert len(state) == 4
    # Statevector must be |00>
    h_state = backend.copy_to_host(state)
    assert np.isclose(h_state[0], 1.0 + 0.0j)
    assert np.allclose(h_state[1:], 0.0)


def test_cuda_apply_gate_fused_numerical():
    """Verify fused gate unitary multiplication produces exact matrix products."""
    backend = CUDABackend()
    
    H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    
    # H @ X @ H = Z
    fused = backend.apply_gate_fused([H, X, H])
    np.testing.assert_allclose(fused, Z, atol=1e-12)


def test_cuda_mocked_cupy_gate_fused():
    """Verify apply_gate_fused executes via CuPy when CuPy is available."""
    mock_cp = MagicMock()
    # Mock cp.eye and cp.matmul
    mock_cp.complex128 = np.complex128
    
    dim = 2
    mock_fused = np.eye(dim, dtype=np.complex128)
    
    def fake_matmul(a, b):
        return a @ b
        
    mock_cp.eye.return_value = mock_fused
    mock_cp.asarray.side_effect = lambda x, dtype: np.asarray(x, dtype=dtype)
    mock_cp.matmul.side_effect = fake_matmul
    mock_cp.asnumpy.side_effect = lambda x: np.asarray(x)

    with patch.dict("sys.modules", {"cupy": mock_cp}):
        backend = CUDABackend()
        H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
        Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        
        # H @ Z @ H = X
        res = backend.apply_gate_fused([H, Z, H])
        X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        np.testing.assert_allclose(res, X, atol=1e-12)


def test_cuda_gate_1q_and_2q_simulation():
    """Verify 1-qubit and 2-qubit transformations correctly generate Bell state (|00> + |11>)/sqrt(2)."""
    backend = CUDABackend()
    num_qubits = 2
    
    # Initialize |00>
    state = backend.allocate_statevector(num_qubits)
    
    # 1. Apply Hadamard to qubit 0
    H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    state = backend.apply_gate_1q(state, H, target=0, num_qubits=num_qubits)
    
    # 2. Apply CNOT with control=0, target=1
    # Standard 4x4 CNOT matrix in computational basis |00>, |01>, |10>, |11>
    # Note: bit 0 is least significant bit (low index).
    # If bit 0 is 1: |01> -> |11> and |11> -> |01>
    CX = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.complex128)
    
    state = backend.apply_gate_2q(state, CX, q0=0, q1=1, num_qubits=num_qubits)
    final_state = backend.copy_to_host(state)
    
    # Expected Bell State: 1/sqrt(2) (|00> + |11>)
    expected_bell = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=np.complex128)
    
    fidelity = np.abs(np.vdot(expected_bell, final_state)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-10)


def test_cuda_mocked_cupy_kernel_launch():
    """Verify that CuPy RawKernel is compiled and dispatched when CuPy ndarray is provided."""
    mock_cp = MagicMock()
    mock_kernel = MagicMock()
    mock_cp.RawKernel.return_value = mock_kernel
    mock_cp.complex128 = np.complex128
    
    # Create fake ndarray
    fake_state = MagicMock(spec=["__class__"])
    fake_state.__class__ = mock_cp.ndarray
    mock_cp.ndarray = type(fake_state)
    
    with patch.dict("sys.modules", {"cupy": mock_cp}):
        backend = CUDABackend()
        H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
        
        backend.apply_gate_1q(fake_state, H, target=0, num_qubits=3)
        assert mock_cp.RawKernel.called
        assert mock_kernel.called


@pytest.mark.skipif(not is_cuda_available(), reason="NVIDIA CUDA hardware not detected on this machine")
def test_cuda_live_hardware_execution():
    """Live integration test executed only when physical NVIDIA GPU and CUDA driver are present."""
    backend = CUDABackend()
    assert backend.is_available() is True
    
    num_qubits = 2
    state = backend.allocate_statevector(num_qubits)
    
    H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    CX = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.complex128)
    
    state = backend.apply_gate_1q(state, H, target=0, num_qubits=num_qubits)
    state = backend.apply_gate_2q(state, CX, q0=0, q1=1, num_qubits=num_qubits)
    final_state = backend.copy_to_host(state)
    
    expected_bell = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=np.complex128)
    fidelity = np.abs(np.vdot(expected_bell, final_state)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-7)


def test_cuda_gate_2q_non_adjacent():
    """Verify 2-qubit gate application between non-adjacent qubits (e.g., q0=0 and q1=2 in 3-qubit system)."""
    backend = CUDABackend()
    num_qubits = 3
    state = backend.allocate_statevector(num_qubits)
    
    # Apply H to qubit 0
    H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    state = backend.apply_gate_1q(state, H, target=0, num_qubits=num_qubits)
    
    # Apply CX with control=0, target=2
    CX = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.complex128)
    
    state = backend.apply_gate_2q(state, CX, q0=0, q1=2, num_qubits=num_qubits)
    final_state = backend.copy_to_host(state)
    
    # Qubit 0 is bit 0, Qubit 2 is bit 2:
    # |000> (index 0) and |101> (index 5)
    expected = np.zeros(8, dtype=np.complex128)
    expected[0] = 1.0 / np.sqrt(2.0)
    expected[5] = 1.0 / np.sqrt(2.0)
    
    fidelity = np.abs(np.vdot(expected, final_state)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-10)


def test_cuda_apply_gate_fused_empty_raises():
    """Verify that fusing an empty matrix list raises a ValueError."""
    backend = CUDABackend()
    with pytest.raises(ValueError, match="Cannot fuse an empty list"):
        backend.apply_gate_fused([])


def test_cuda_cli_backend_argument_parsing():
    """Verify CLI parser correctly parses --backend cuda and other backend options."""
    from cli.main import build_argument_parser
    parser = build_argument_parser()
    
    args_cuda = parser.parse_args(["--qubits", "4", "--backend", "cuda"])
    assert args_cuda.backend == "cuda"
    
    args_auto = parser.parse_args(["--qubits", "4", "--backend", "auto"])
    assert args_auto.backend == "auto"
    
    args_metal = parser.parse_args(["--qubits", "4", "--backend", "metal"])
    assert args_metal.backend == "metal"

