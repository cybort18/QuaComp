import os
import shutil
import pytest
from src.comparator.registry import (
    fetch_remote_baselines,
    list_available_baselines,
    get_baseline_path,
    ENTERPRISE_BASELINES
)
from src.comparator.differ import resolve_target_profile

def test_enterprise_baselines_catalog():
    """Verify catalog has all required enterprise architectures."""
    required = ["apple_m3", "ryzen3_5300u", "ryzen7_5800h", "apple_m4_max", "nvidia_h100", "aws_graviton4"]
    for r in required:
        assert r in ENTERPRISE_BASELINES
        assert "filename" in ENTERPRISE_BASELINES[r]
        assert "device" in ENTERPRISE_BASELINES[r]

def test_fetch_remote_baselines_and_caching(tmp_path):
    """Verify synchronization caches files locally and handles offline fallback."""
    cache_dir = str(tmp_path / "test_registry")
    rep = fetch_remote_baselines(cache_dir=cache_dir)
    
    assert rep["total_available"] == len(ENTERPRISE_BASELINES)
    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "example_apple_m4_max.json"))
    assert os.path.exists(os.path.join(cache_dir, "example_nvidia_h100.json"))
    
    # Check repeated sync uses existing cache
    rep2 = fetch_remote_baselines(cache_dir=cache_dir, force_refresh=False)
    assert len(rep2["existing_cached"]) == len(ENTERPRISE_BASELINES)

def test_list_available_baselines():
    """Verify list_available_baselines output structure."""
    baselines = list_available_baselines()
    assert len(baselines) >= 6
    for b in baselines:
        assert "alias" in b
        assert "description" in b
        assert "device" in b
        assert "is_available" in b

def test_get_baseline_path():
    """Verify resolution of baseline file paths."""
    p_m3 = get_baseline_path("apple_m3")
    assert p_m3 is not None
    assert os.path.exists(p_m3)
    
    p_nonexistent = get_baseline_path("non_existent_chip_xyz")
    assert p_nonexistent is None

def test_resolve_target_profile_with_registry():
    """Verify resolve_target_profile resolves enterprise aliases."""
    path_m4 = resolve_target_profile("apple_m4_max")
    assert os.path.exists(path_m4)
    assert path_m4.endswith(".json")
    
    path_h100 = resolve_target_profile("nvidia_h100")
    assert os.path.exists(path_h100)
    assert path_h100.endswith(".json")
