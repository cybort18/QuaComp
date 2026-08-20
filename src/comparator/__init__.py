"""
Comparator Module for QuaComp
Handles relative benchmark comparison, metric differencing, and reporting.
"""

from src.comparator.differ import load_benchmark_json, compare_benchmarks, resolve_target_profile
from src.comparator.reporter import (
    render_comparison_terminal,
    export_comparison_to_json,
    export_comparison_to_markdown
)

__all__ = [
    "load_benchmark_json",
    "compare_benchmarks",
    "resolve_target_profile",
    "render_comparison_terminal",
    "export_comparison_to_json",
    "export_comparison_to_markdown"
]
