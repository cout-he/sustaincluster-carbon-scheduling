"""Independently reconcile saved E0 CSVs and write verification evidence."""
from pathlib import Path
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone

E0 = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    start = time.perf_counter()
    folder = E0 / "artifacts/baseline"
    execution = json.loads((folder / "execution.json").read_text(encoding="utf-8"))
    summaries = read_csv(folder / "summary.csv")
    steps = read_csv(folder / "steps.csv")
    dcs = read_csv(folder / "datacenters.csv")
    checks = json.loads((folder / "checks.json").read_text(encoding="utf-8"))
    assert execution["status"] == "passed" and all(c["passed"] for c in checks)
    assert len(summaries) == 8 and len(steps) == 768 and len(dcs) == 3840
    assert len({s["cohort_sequence_sha256"] for s in summaries}) == 1
    metrics = ("incoming_tasks", "tasks_started", "tasks_finished", "sla_met", "sla_violated",
               "site_energy_kwh", "site_emissions_kg", "site_energy_cost_usd",
               "transmission_energy_kwh", "transmission_emissions_kg", "transmission_cost_usd", "reward")
    by_dc = {}
    for dc in dcs:
        by_dc.setdefault((dc["run"], dc["step"]), []).append(dc)
    for s in summaries:
        assert s["status"] == "passed" and int(s["steps"]) == 96
        assert int(s["incoming_tasks"]) == 2506
        run_steps = [r for r in steps if r["run"] == s["run"]]
        assert len(run_steps) == 96
        for metric in metrics:
            assert math.isclose(float(s[metric]), sum(float(r[metric]) for r in run_steps),
                                rel_tol=1e-10, abs_tol=1e-7), (s["run"], metric)
        for r in run_steps:
            per_dc = by_dc[(r["run"], r["step"])]
            assert len(per_dc) == 5
            for metric in ("site_energy_kwh", "site_emissions_kg", "site_energy_cost_usd",
                           "tasks_started", "tasks_finished", "running_tasks", "pending_tasks",
                           "used_cpu_cores", "used_gpus", "used_memory_gb"):
                assert math.isclose(float(r[metric]), sum(float(d[metric]) for d in per_dc),
                                    rel_tol=1e-10, abs_tol=1e-7), (r["run"], r["step"], metric)
    for name in ("local_only_repeat", "local_only_direct"):
        normal = [{k: v for k, v in r.items() if k != "run"} for r in steps if r["run"] == "local_only"]
        other = [{k: v for k, v in r.items() if k != "run"} for r in steps if r["run"] == name]
        assert normal == other
    files = {}
    for path in sorted(folder.iterdir()):
        if path.is_file():
            payload = path.read_bytes()
            files[path.name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    result = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.perf_counter() - start,
              "runs": len(summaries), "steps": len(steps), "datacenter_rows": len(dcs),
              "checks": "all summary sums, all per-step DC sums, cohorts, exact repeat/direct equality",
              "baseline_artifacts": files}
    (E0 / "evidence/verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "baseline_artifacts"}))


if __name__ == "__main__":
    main()
