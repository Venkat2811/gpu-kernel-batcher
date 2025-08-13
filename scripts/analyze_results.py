#!/usr/bin/python3
"""
Analyze benchmark results and compute speedups
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Optional
from tabulate import tabulate

def load_result(result_dir: str) -> Optional[Dict]:
    """Load consumer.json from result directory"""
    consumer_file = Path(result_dir) / "consumer.json"
    if not consumer_file.exists():
        return None
    
    with open(consumer_file) as f:
        return json.load(f)

def load_config(result_dir: str) -> Optional[Dict]:
    """Load config.json from result directory"""
    config_file = Path(result_dir) / "config.json"
    if not config_file.exists():
        return None
    
    with open(config_file) as f:
        return json.load(f)

def format_params(config: Dict, workload_name: str) -> str:
    """Formats workload-specific parameters into a readable string."""
    if not config:
        return ""
    
    params = []
    wl = workload_name.lower()
    
    if "gemv-micro" in wl or "gemv" in wl:
        # Keep M and K as they are standard matrix notation
        if 'm' in config:
            params.append(f"M={config.get('m')}")
        if 'k' in config:
            params.append(f"K={config.get('k')}")
    elif "decode" in wl:
        # Use full names for clarity
        if 'hidden' in config:
            params.append(f"Hidden={config.get('hidden')}")
        if 'layers' in config:
            params.append(f"Layers={config.get('layers')}")
        if 'batch' in config:
            params.append(f"Batch={config.get('batch')}")
    elif "prefill" in wl:
        # Use full names for clarity
        if 'hidden' in config:
            params.append(f"Hidden={config.get('hidden')}")
        if 'layers' in config:
            params.append(f"Layers={config.get('layers')}")
        if 'batch' in config:
            params.append(f"Batch={config.get('batch')}")
        if 'seq_len' in config:
            params.append(f"SeqLen={config.get('seq_len')}")
    
    return f" ({', '.join(params)})" if params else ""

def compute_speedup(baseline_time: float, optimized_time: float) -> float:
    """Compute speedup percentage"""
    return (baseline_time - optimized_time) / baseline_time * 100

def analyze_workload(workload_name: str, baseline_dir: str, optimized_dir: str, dtype: str):
    """Analyze a single workload comparison"""
    baseline = load_result(baseline_dir)
    optimized = load_result(optimized_dir)
    baseline_config = load_config(baseline_dir)
    optimized_config = load_config(optimized_dir)
    
    if not baseline or not optimized:
        print(f"❌ {workload_name} ({dtype}): Missing result files")
        return
    
    baseline_data = baseline["consumer"]
    optimized_data = optimized["consumer"]
    
    # Detect timing mode
    has_gpu_timing = baseline_data.get("gpu_time_ms", 0.0) > 0 or optimized_data.get("gpu_time_ms", 0.0) > 0
    timing_mode = "GPU-timed" if has_gpu_timing else "Wall-clock only"
    
    # Print workload header with parameters
    status = "✅" if compute_speedup(baseline_data["wall_time_ms"], optimized_data["wall_time_ms"]) > 0 else "❌"
    print(f"\n{status} {workload_name} ({dtype.upper()}) - {timing_mode}")
    print("=" * 80)
    
    # Display parameters
    wl = workload_name.lower()
    if baseline_config:
        params = []
        if "gemv" in wl:
            params = [
                ("M", baseline_config.get('m', 'N/A')),
                ("K", baseline_config.get('k', 'N/A'))
            ]
        elif "decode" in wl:
            params = [
                ("Hidden", baseline_config.get('hidden', 'N/A')),
                ("Layers", baseline_config.get('layers', 'N/A')),
                ("Batch", baseline_config.get('batch', 'N/A'))
            ]
        elif "prefill" in wl:
            params = [
                ("Hidden", baseline_config.get('hidden', 'N/A')),
                ("Layers", baseline_config.get('layers', 'N/A')),
                ("Batch", baseline_config.get('batch', 'N/A')),
                ("SeqLen", baseline_config.get('seq_len', 'N/A'))
            ]
        
        if params:
            print("Parameters: " + ", ".join([f"{k}={v}" for k, v in params]))
    
    # Create results table
    headers = ["Configuration", "Wall Time (ms)", "GPU Time (ms)", "Launches", "Mode"]
    table_data = []
    
    # Baseline row
    baseline_mode = baseline_config.get("mode", "none") if baseline_config else "none"
    baseline_gpu_str = f"{baseline_data.get('gpu_time_ms', 0.0):.1f}" if baseline_data.get('gpu_time_ms', 0.0) > 0 else "N/A"
    table_data.append([
        "Baseline",
        f"{baseline_data['wall_time_ms']:.1f}",
        baseline_gpu_str,
        f"{baseline_data['total_launches']:,}",
        baseline_mode
    ])
    
    # Optimized row
    optimized_mode = optimized_config.get("mode", "shape-batched") if optimized_config else "shape-batched"
    optimized_gpu_str = f"{optimized_data.get('gpu_time_ms', 0.0):.1f}" if optimized_data.get('gpu_time_ms', 0.0) > 0 else "N/A"
    table_data.append([
        "Optimized",
        f"{optimized_data['wall_time_ms']:.1f}",
        optimized_gpu_str,
        f"{optimized_data['total_launches']:,}",
        optimized_mode
    ])
    
    # Speedup row
    wall_speedup = compute_speedup(baseline_data["wall_time_ms"], optimized_data["wall_time_ms"])
    gpu_speedup = compute_speedup(baseline_data.get("gpu_time_ms", 0.0), optimized_data.get("gpu_time_ms", 0.0)) if baseline_data.get("gpu_time_ms", 0.0) > 0 else 0
    gpu_speedup_str = f"{gpu_speedup:+.1f}%" if baseline_data.get("gpu_time_ms", 0.0) > 0 else "N/A"
    launch_reduction = optimized_data.get("launch_reduction_percent", 0)
    
    table_data.append([
        "**Speedup**",
        f"{wall_speedup:+.1f}%",
        gpu_speedup_str,
        f"{launch_reduction:.1f}% reduction",
        ""
    ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()

def main():
    if len(sys.argv) != 2:
        print("Usage: analyze_results.py <results_base_dir>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    
    print("\n🚀 CUDA Kernel Launch Optimizer - Results Analysis")
    print("=" * 80)
    
    # Define workload comparisons
    workloads = [
        ("GEMV-micro", "gemv_baseline", "gemv_batched"),
        ("Decode", "decode_baseline", "decode_batched"), 
        ("Prefill", "prefill_baseline", "prefill_batched"),
    ]
    
    # Process both FP32 and FP16 together for each workload
    for workload_name, baseline_name, optimized_name in workloads:
        # Check FP32
        baseline_fp32 = results_dir / f"{baseline_name}_fp32"
        optimized_fp32 = results_dir / f"{optimized_name}_fp32"
        if baseline_fp32.exists() and optimized_fp32.exists():
            analyze_workload(workload_name, str(baseline_fp32), str(optimized_fp32), "fp32")
        
        # Check FP16
        baseline_fp16 = results_dir / f"{baseline_name}_fp16"
        optimized_fp16 = results_dir / f"{optimized_name}_fp16"
        if baseline_fp16.exists() and optimized_fp16.exists():
            analyze_workload(workload_name, str(baseline_fp16), str(optimized_fp16), "fp16")

if __name__ == "__main__":
    main()