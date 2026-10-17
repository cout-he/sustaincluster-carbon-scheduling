"""Inspect restored inputs without cleaning, regenerating or changing their bytes."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "research/e0/data_manifest.json").read_text(encoding="utf-8"))
    matched = 0
    for entry in manifest["files"]:
        content = (root / entry["path"]).read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Restored data differs from provenance manifest: {entry['path']}")
        matched += 1
    sim = yaml.safe_load((root / "configs/env/sim_config.yaml").read_text(encoding="utf-8"))["simulation"]
    dcs = yaml.safe_load((root / "configs/env/datacenters.yaml").read_text(encoding="utf-8"))["datacenters"]
    year = sim["year"]
    files = []
    for dc in dcs:
        location = dc["location"]
        specifications = (
            (f"data/carbon_intensity/{location}/{year}/{location}_{year}_hourly.csv", "Carbon Intensity gCO₂eq/kWh (direct)"),
            (f"data/electricity_prices/standardized/{location}/{year}/{location}_electricity_prices_{year}.csv", "Price (USD/MWh)"),
        )
        for relative_path, column in specifications:
            frame = pd.read_csv(root / relative_path)
            dates = pd.to_datetime(frame["Datetime (UTC)"])
            values = frame[column].to_numpy(dtype=float)
            files.append({
                "path": relative_path,
                "rows": len(frame),
                "column": column,
                "time_start": str(dates.min()),
                "time_end": str(dates.max()),
                "duplicate_timestamps": int(dates.duplicated().sum()),
                "monotonic_time": bool(dates.is_monotonic_increasing),
                "non_hourly_intervals": int((dates.diff().dropna() != pd.Timedelta(hours=1)).sum()),
                "nonfinite_values": int((~np.isfinite(values)).sum()),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            })
        relative_path = f"data/weather/{location}/{year}.json"
        weather = json.loads((root / relative_path).read_text(encoding="utf-8"))["hourly"]
        dates = pd.Series(pd.to_datetime(weather["time"]))
        fields = ["temperature_2m", "relative_humidity_2m", "cloudcover", "windspeed_10m"]
        files.append({
            "path": relative_path,
            "rows": len(dates),
            "time_start": str(dates.min()),
            "time_end": str(dates.max()),
            "duplicate_timestamps": int(dates.duplicated().sum()),
            "monotonic_time": bool(dates.is_monotonic_increasing),
            "non_hourly_intervals": int((dates.diff().dropna() != pd.Timedelta(hours=1)).sum()),
            "fields": {
                field: {
                    "length": len(weather[field]),
                    "nonfinite_values": int((~np.isfinite(np.asarray(weather[field], dtype=float))).sum()),
                }
                for field in fields
            },
        })
    network_path = f"data/network_cost/{sim['cloud_provider']}_transmission_cost_matrix.csv"
    network = pd.read_csv(root / network_path, index_col=0)
    network_values = network.to_numpy(dtype=float)
    network_details = {
        "path": network_path,
        "shape": list(network.shape),
        "missing_values": int(np.isnan(network_values).sum()),
        "min_finite": float(network_values[np.isfinite(network_values)].min()),
        "max_finite": float(network_values[np.isfinite(network_values)].max()),
        "original_loader_missing_value_behavior": "fillna(0.0); no recovery-time transformation",
    }
    workload = pd.read_pickle(root / sim["workload_path"])
    intervals = workload["interval_15m"]
    deltas = intervals.diff().dropna()
    total_expected = int((intervals.max() - intervals.min()) / pd.Timedelta(minutes=15)) + 1
    task_values = np.concatenate([matrix[:, 4:9].astype(float) for matrix in workload["tasks_matrix"]])
    workload_details = {
        "path": sim["workload_path"],
        "rows": len(workload),
        "columns": list(workload.columns),
        "task_records": int(sum(len(matrix) for matrix in workload["tasks_matrix"])),
        "task_matrix_width_counts": dict(Counter(str(matrix.shape[1]) for matrix in workload["tasks_matrix"])),
        "time_start": str(intervals.min()),
        "time_end": str(intervals.max()),
        "unique_intervals": int(intervals.nunique()),
        "duplicate_intervals": int(intervals.duplicated().sum()),
        "monotonic_time": bool(intervals.is_monotonic_increasing),
        "non_15minute_intervals": int((deltas != pd.Timedelta(minutes=15)).sum()),
        "interval_delta_counts": {str(key): int(value) for key, value in deltas.value_counts().items()},
        "missing_15minute_slots_within_date_range": total_expected - int(intervals.nunique()),
        "empty_task_matrices": int(sum(len(matrix) == 0 for matrix in workload["tasks_matrix"])),
        "raw_task_numeric_columns": {
            name: {
                "nonfinite_values": int((~np.isfinite(task_values[:, idx])).sum()),
                "min": float(np.nanmin(task_values[:, idx])),
                "max": float(np.nanmax(task_values[:, idx])),
                "negative_values": int((task_values[:, idx] < 0).sum()),
            }
            for idx, name in enumerate(["duration_minutes", "cpu_raw", "gpu_raw", "memory_raw", "bandwidth_gb"])
        },
        "recovery_transformation": "None; only original ZIP member extraction",
    }
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "source_commit": manifest["source_commit"],
        "manifest_files_checked": matched,
        "all_manifest_hashes_match": True,
        "configured_environment_year": year,
        "configured_datacenter_locations": [dc["location"] for dc in dcs],
        "configured_environment_files": files,
        "configured_network_cost": network_details,
        "workload": workload_details,
    }
    output = root / "research/e0/data_quality.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Keep stdout usable in Windows consoles whose default encoding is GBK.
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
