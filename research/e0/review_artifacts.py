"""Independently verify recorded E0 tables; does not run or modify the simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=root / "research/e0/artifacts/baseline")
    args = parser.parse_args()
    folder = args.artifacts.resolve()
    summaries = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    with (folder / "steps.csv").open(encoding="utf-8", newline="") as stream:
        steps = list(csv.DictReader(stream))
    with (folder / "datacenters.csv").open(encoding="utf-8", newline="") as stream:
        sites = list(csv.DictReader(stream))
    runs, site_steps = defaultdict(list), defaultdict(list)
    for row in steps:
        runs[row["run"]].append(row)
    for row in sites:
        site_steps[(row["run"], int(row["step"]))].append(row)
    assert len(sites) == len(steps) * 5
    max_site_rollup_error = 0.0

    def equal(actual, expected):
        assert math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-6)

    fields = ("site_energy_kwh", "site_emissions_kg", "site_energy_cost_usd", "tasks_started",
              "tasks_finished", "sla_met", "sla_violated", "used_cpu_cores", "used_gpus",
              "used_memory_gb", "running_tasks", "pending_tasks")
    for row in steps:
        site_rows = site_steps[(row["run"], int(row["step"]))]
        assert len(site_rows) == 5
        for field in fields:
            total = sum(float(site[field]) for site in site_rows)
            equal(row[field], total)
            max_site_rollup_error = max(max_site_rollup_error, abs(float(row[field]) - total))
        for site in site_rows:
            equal(site["site_emissions_kg"], float(site["site_energy_kwh"]) *
                  float(site["battery_ci_g_per_kwh"]) / 1000.0)
            equal(site["site_energy_cost_usd"], float(site["site_energy_kwh"]) *
                  float(site["price_usd_per_mwh"]) / 1000.0)
    for summary in summaries:
        assert summary["status"] == "passed"
        recorded = runs[summary["run"]]
        assert [int(row["step"]) for row in recorded] == list(range(summary["steps"]))
        cohort_digest = hashlib.sha256(json.dumps([row["cohort_sha256"] for row in recorded],
                                                  separators=(",", ":")).encode()).hexdigest()
        assert cohort_digest == summary["cohort_sequence_sha256"]
        for field in ("incoming_tasks", "tasks_started", "tasks_finished", "site_energy_kwh",
                      "site_emissions_kg", "site_energy_cost_usd", "transmission_energy_kwh",
                      "transmission_emissions_kg", "transmission_cost_usd", "reward"):
            equal(summary[field], sum(float(row[field]) for row in recorded))
        incoming = started = finished = 0
        for row in recorded:
            incoming += int(row["incoming_tasks"])
            started += int(row["tasks_started"])
            finished += int(row["tasks_finished"])
            assert incoming == started + int(row["pending_tasks"]) + int(row["in_transit_tasks"]) + int(row["deferred_tasks"])
            assert started == finished + int(row["running_tasks"])
    assert len(runs) == len(summaries)
    assert len({len(rows) for rows in runs.values()}) == 1
    for step in range(len(next(iter(runs.values())))):
        assert len({rows[step]["cohort_sha256"] for rows in runs.values()}) == 1
    names = subprocess.check_output(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
                                    text=True).splitlines()
    large, raw_inputs, credentials = [], [], []
    pattern = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|oe_[A-Za-z0-9]{16,}|AKIA[A-Z0-9]{16})")
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            large.append(name)
        if name.startswith("data/") and path.suffix.lower() in {".csv", ".json", ".zip", ".pkl", ".pickle"}:
            raw_inputs.append(name)
        if path.stat().st_size <= 5 * 1024 * 1024:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeError:
                continue
            for number, line in enumerate(content.splitlines(), 1):
                if pattern.search(line):
                    credentials.append({"path": name, "line": number})
    assert not large and not raw_inputs and not credentials
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent table recomputation and Git candidate boundary review; no simulation rerun",
        "baseline_step_rows": len(steps),
        "baseline_site_rows": len(sites),
        "run_count": len(runs),
        "all_site_rollups_and_summary_totals_match": True,
        "max_abs_site_rollup_error": max_site_rollup_error,
        "every_step_task_count_conservation": True,
        "every_step_cohort_hash_equality": True,
        "all_cohort_sequence_hashes_match": True,
        "site_emissions_g_to_kg_conversion_matches": True,
        "site_usd_per_mwh_to_cost_conversion_matches": True,
        "tracked_candidate_files_over_5MiB": large,
        "tracked_candidate_raw_input_files": raw_inputs,
        "credential_pattern_findings_paths_only": credentials,
        "credential_scan_limit": "Common token patterns in candidate UTF-8 text files, not a proof about all history or secret types",
    }
    output = root / "research/e0/artifact_review.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
