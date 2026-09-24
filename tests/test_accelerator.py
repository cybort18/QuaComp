import pytest
import numpy as np
from qiskit.circuit.library import HGate, XGate, ZGate
from qiskit.quantum_info import Operator

from src.engine.accelerator import (
    load_metal_shader,
    CUDA_KERNEL_SOURCE,
    is_metal_available,
    is_cuda_available,
    get_available_accelerators,
    detect_primary_accelerator,
    dispatch_gate_fusion,
    get_accelerator_badge
)


def test_metal_shader_loading():
    """Verify Apple Metal shader loads correctly and contains kernel signatures."""
    shader_code = load_metal_shader()
    assert shader_code is not None
    assert "matmul_2x2_metal" in shader_code
    assert "matmul_4x4_metal" in shader_code
    assert "kernel void" in shader_code


def test_cuda_kernel_source():
    """Verify NVIDIA CUDA kernel source template syntax and signatures."""
    assert "matmul_2x2_cuda" in CUDA_KERNEL_SOURCE
    assert "matmul_4x4_cuda" in CUDA_KERNEL_SOURCE
    assert "__global__ void" in CUDA_KERNEL_SOURCE


def test_accelerator_detection_smoke():
    """Verify hardware detection functions return valid types without exceptions."""
    metal_ok = is_metal_available()
    cuda_ok = is_cuda_available()
    assert isinstance(metal_ok, bool)
    assert isinstance(cuda_ok, bool)
    
    avail = get_available_accelerators()
    assert isinstance(avail, list)
    assert "CPython Fallback Engine" in avail
    
    primary = detect_primary_accelerator()
    assert isinstance(primary, str)
    assert primary in ("Metal GPU", "CUDA GPU", "C++ Native Engine", "CPython Fallback Engine")


def test_dispatch_gate_fusion_auto():
    """Verify automatic dispatch produces mathematically accurate unitary fusion."""
    H = Operator(HGate()).data
    X = Operator(XGate()).data
    Z = Operator(ZGate()).data
    
    # H, then X, then H -> H @ X @ H = Z
    matrices = [H, X, H]
    fused, tier_used = dispatch_gate_fusion(matrices)
    
    assert tier_used in ("Metal GPU", "CUDA GPU", "C++ Native Engine", "CPython Fallback Engine")
    np.testing.assert_allclose(fused, Z, atol=1e-14)


def test_dispatch_gate_fusion_fallback_routes():
    """Verify graceful fallback across all requested tiers without raising exceptions."""
    H = Operator(HGate()).data
    X = Operator(XGate()).data
    Z = Operator(ZGate()).data
    matrices = [H, X, H]
    
    # Force Python
    fused_py, tier_py = dispatch_gate_fusion(matrices, force_tier="python")
    assert tier_py == "CPython Fallback Engine"
    np.testing.assert_allclose(fused_py, Z, atol=1e-14)
    
    # Force C++ (executes C++ or falls back safely to CPython)
    fused_cpp, tier_cpp = dispatch_gate_fusion(matrices, force_tier="cpp")
    assert tier_cpp in ("C++ Native Engine", "CPython Fallback Engine")
    np.testing.assert_allclose(fused_cpp, Z, atol=1e-14)
    
    # Force Metal (if on non-Metal machine, falls back to C++/CPython)
    fused_metal, tier_metal = dispatch_gate_fusion(matrices, force_tier="metal")
    assert tier_metal in ("Metal GPU", "C++ Native Engine", "CPython Fallback Engine")
    np.testing.assert_allclose(fused_metal, Z, atol=1e-14)
    
    # Force CUDA (if on non-CUDA machine, falls back to C++/CPython)
    fused_cuda, tier_cuda = dispatch_gate_fusion(matrices, force_tier="cuda")
    assert tier_cuda in ("CUDA GPU", "C++ Native Engine", "CPython Fallback Engine")
    np.testing.assert_allclose(fused_cuda, Z, atol=1e-14)


def test_dispatch_gate_fusion_empty_raises():
    """Verify error on empty matrix sequence."""
    with pytest.raises(ValueError):
        dispatch_gate_fusion([])


def test_accelerator_badge_formatting():
    """Verify badge formatting for report and CLI outputs."""
    assert get_accelerator_badge("Metal GPU") == "[Engine: Metal GPU]"
    assert get_accelerator_badge("CUDA GPU") == "[Engine: CUDA GPU]"
    assert get_accelerator_badge("C++ Native Engine") == "[Engine: C++ SIMD]"
    assert get_accelerator_badge("CPython Fallback Engine") == "[Engine: CPython]"
    assert get_accelerator_badge(None) == "[Engine: CPython]"
