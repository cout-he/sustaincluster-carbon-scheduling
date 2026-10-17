"""Collect recorded executions; never infer missing physical measurements."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent


def collect():
    rows = []
    for path in sorted((ROOT / "evidence").glob("*.command.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "experiment": entry["id"], "kind": "command",
            "status": "passed" if entry["exit_code"] == 0 else "failed",
            "elapsed_seconds": entry["elapsed_seconds"],
            "exit_code": entry["exit_code"],
            "command": json.dumps(entry["command"], ensure_ascii=False),
            "evidence": str(path.relative_to(ROOT)).replace("\\", "/"),
        })
    for path in sorted((ROOT / "artifacts").glob("*/execution.json")):
        execution = json.loads(path.read_text(encoding="utf-8"))
        summaries = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
        for summary in summaries or [{"run": "execution", "status": execution["status"],
                                      "error": execution.get("error", "")}]:
            row = {key: value for key, value in summary.items()
                   if key not in ("traceback", "initial_physical_clocks")}
            row.update(experiment=path.parent.name + "/" + row.pop("run"),
                       kind="simulation", execution_status=execution["status"],
                       command=execution["command"],
                       evidence=str(path.with_name("summary.json").relative_to(ROOT)).replace("\\", "/"))
            if row["status"] == "passed":
                row["site_plus_transmission_energy_kwh"] = row["site_energy_kwh"] + row["transmission_energy_kwh"]
                row["site_plus_transmission_emissions_kg"] = row["site_emissions_kg"] + row["transmission_emissions_kg"]
            rows.append(row)
    fields = ["experiment", "kind", "status", "execution_status", "elapsed_seconds"]
    fields += [key for row in rows for key in row if key not in fields]
    fields = list(dict.fromkeys(fields))
    with (ROOT / "baseline_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "passed": sum(r["status"] == "passed" for r in rows),
                      "failed": sum(r["status"] == "failed" for r in rows)}))


if __name__ == "__main__":
    collect()
