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

def test_fetch_remote_baselines_rate_limit(tmp_path, monkeypatch):
    """Verify GitHub rate limit (HTTP 403/429) triggers informative diagnostics and offline fallback."""
    import urllib.request
    import urllib.error
    
    def mock_urlopen_rate_limit(req, timeout=5.0):
        raise urllib.error.HTTPError(
            url="https://api.github.com",
            code=403,
            msg="rate limit exceeded",
            hdrs={},
            fp=None
        )
        
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_rate_limit)
    cache_dir = str(tmp_path / "rate_limit_cache")
    
    rep = fetch_remote_baselines(target_alias="apple_m4_max", cache_dir=cache_dir, timeout=5.0)
    assert rep["rate_limited"] is True
    assert rep["offline_mode"] is True
    assert "apple_m4_max" in rep["offline_fallback"]
    assert "rate limit" in rep["errors"]["apple_m4_max"].lower()
    assert "rate limit" in rep["status_message"].lower()
    assert os.path.exists(os.path.join(cache_dir, "example_apple_m4_max.json"))

def test_fetch_remote_baselines_timeout_handling(tmp_path, monkeypatch):
    """Verify network timeout triggers informative timeout reporting and fallback."""
    import urllib.request
    
    def mock_urlopen_timeout(req, timeout=5.0):
        raise TimeoutError("The read operation timed out")
        
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_timeout)
    cache_dir = str(tmp_path / "timeout_cache")
    
    rep = fetch_remote_baselines(target_alias="nvidia_h100", cache_dir=cache_dir, timeout=5.0)
    assert rep["offline_mode"] is True
    assert "nvidia_h100" in rep["offline_fallback"]
    assert "timed out" in rep["errors"]["nvidia_h100"].lower()
    assert os.path.exists(os.path.join(cache_dir, "example_nvidia_h100.json"))

def test_fetch_remote_baselines_offline_network_error(tmp_path, monkeypatch):
    """Verify URLError triggers informative offline diagnostics."""
    import urllib.request
    import urllib.error
    
    def mock_urlopen_offline(req, timeout=5.0):
        raise urllib.error.URLError("Temporary failure in name resolution")
        
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_offline)
    cache_dir = str(tmp_path / "offline_cache")
    
    rep = fetch_remote_baselines(target_alias="aws_graviton4", cache_dir=cache_dir)
    assert rep["offline_mode"] is True
    assert "aws_graviton4" in rep["offline_fallback"]
    assert "offline" in rep["errors"]["aws_graviton4"].lower() or "connection" in rep["errors"]["aws_graviton4"].lower()
    assert os.path.exists(os.path.join(cache_dir, "example_aws_graviton4.json"))

def test_ttl_cache_validity_and_expiration(tmp_path):
    """Verify TTL cache validation, expiration, and invalidation."""
    import time
    from src.comparator.registry import (
        is_cache_valid,
        invalidate_cache,
        save_cache_metadata,
        load_cache_metadata
    )
    
    ttl_root = str(tmp_path / ".test_quacomp_cache")
    os.makedirs(ttl_root, exist_ok=True)
    
    # 1. Non-existent cache is invalid
    assert not is_cache_valid("apple_m3", cache_root=ttl_root)
    
    # 2. Create cached file with metadata
    test_file = os.path.join(ttl_root, "example_apple_m3.json")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write('{"test": 1}')
        
    save_cache_metadata(ttl_root, {
        "apple_m3": {
            "filename": "example_apple_m3.json",
            "cached_at": time.time(),
            "source": "unit_test"
        }
    })
    
    # 3. Active within 3600s TTL
    assert is_cache_valid("apple_m3", cache_root=ttl_root, ttl_seconds=3600.0) is True
    
    # 4. Expired when TTL is 0s
    assert is_cache_valid("apple_m3", cache_root=ttl_root, ttl_seconds=0.0) is False
    
    # 5. Invalidation clears cache
    invalidate_cache("apple_m3", cache_root=ttl_root)
    assert not os.path.exists(test_file)
    assert not is_cache_valid("apple_m3", cache_root=ttl_root)

def test_ttl_cache_prevents_repeated_network_request(tmp_path, monkeypatch):
    """Verify valid local TTL cache completely skips remote network HTTP requests."""
    import urllib.request
    import time
    from src.comparator.registry import save_cache_metadata
    
    ttl_dir = str(tmp_path / ".ttl_cache")
    target_dir = str(tmp_path / "target_reg")
    os.makedirs(ttl_dir, exist_ok=True)
    
    # Pre-populate TTL cache with valid entry
    cached_content = '{"mock_baseline": "valid_ttl_cache"}'
    with open(os.path.join(ttl_dir, "example_apple_m4_max.json"), "w", encoding="utf-8") as f:
        f.write(cached_content)
        
    save_cache_metadata(ttl_dir, {
        "apple_m4_max": {
            "filename": "example_apple_m4_max.json",
            "cached_at": time.time(),
            "source": "test_preloaded"
        }
    })
    
    # If network request is attempted, raise an error to prove it was never called
    network_called = False
    def mock_urlopen_forbidden(req, timeout=5.0):
        nonlocal network_called
        network_called = True
        raise AssertionError("Network HTTP request must NOT be performed when TTL cache is valid!")
        
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_forbidden)
    
    # Run fetch_remote_baselines with valid TTL
    rep = fetch_remote_baselines(
        target_alias="apple_m4_max",
        cache_dir=target_dir,
        ttl_cache_dir=ttl_dir,
        ttl_seconds=86400.0,
        force_refresh=False
    )
    
    assert network_called is False
    assert "apple_m4_max" in rep["existing_cached"]
    assert os.path.exists(os.path.join(target_dir, "example_apple_m4_max.json"))
    with open(os.path.join(target_dir, "example_apple_m4_max.json"), "r", encoding="utf-8") as f:
        assert "valid_ttl_cache" in f.read()

def test_ttl_cache_expiration_triggers_refetch(tmp_path, monkeypatch):
    """Verify expired TTL cache triggers fresh remote fetch."""
    import urllib.request
    from unittest.mock import MagicMock
    import time
    from src.comparator.registry import save_cache_metadata
    
    ttl_dir = str(tmp_path / ".ttl_cache_stale")
    target_dir = str(tmp_path / "target_reg_stale")
    os.makedirs(ttl_dir, exist_ok=True)
    
    # Pre-populate with stale timestamp (2 days ago)
    with open(os.path.join(ttl_dir, "example_apple_m3.json"), "w", encoding="utf-8") as f:
        f.write('{"old": true}')
        
    save_cache_metadata(ttl_dir, {
        "apple_m3": {
            "filename": "example_apple_m3.json",
            "cached_at": time.time() - 172800,  # 48 hours ago
            "source": "stale"
        }
    })
    
    network_called = False
    def mock_urlopen_success(req, timeout=5.0):
        nonlocal network_called
        network_called = True
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"refreshed": true, "timestamp": "2026-09-23T00:00:00"}'
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp
        
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_success)
    
    rep = fetch_remote_baselines(
        target_alias="apple_m3",
        cache_dir=target_dir,
        ttl_cache_dir=ttl_dir,
        ttl_seconds=86400.0,
        force_refresh=False
    )
    
    assert network_called is True
    assert "apple_m3" in rep["synced_remote"]
    with open(os.path.join(target_dir, "example_apple_m3.json"), "r", encoding="utf-8") as f:
        data = f.read()
        assert "refreshed" in data
