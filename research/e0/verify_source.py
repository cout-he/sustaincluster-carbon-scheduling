"""Verify source-only snapshot against pinned official Git blobs, without printing source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
BASE = "749dc6c1868065c12a71bac71b3ae08a25c45f56"
OFFICIAL = "3f6ea95cb835b89ba50b0ef76d66d14b8037643e"
REDACTED = "data/electricity_prices/scripts/test_prices.py"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def normalized(payload):
    return payload.replace(b"\r\n", b"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=ROOT.parent / "upstream")
    args = parser.parse_args()
    start = time.perf_counter()
    source = args.source_repo.resolve()
    manifest = json.loads((ROOT / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    assert manifest["upstream_sha"] == OFFICIAL
    assert git(ROOT, "branch", "--show-current").decode().strip() == "research/rl-scheduling-v1"
    assert git(ROOT, "merge-base", BASE, "HEAD").decode().strip() == BASE
    assert git(ROOT, "rev-parse", "main").decode().strip() == BASE
    assert not git(ROOT, "rev-list", "--merges", BASE + "..HEAD").strip()
    rows = []
    for entry in manifest["included_original_files"]:
        official = git(source, "show", OFFICIAL + ":" + entry["upstream_path"])
        assert len(official) == entry["upstream_bytes"], entry["upstream_path"]
        assert hashlib.sha256(official).hexdigest() == entry["upstream_sha256"], entry["upstream_path"]
        path = entry["release_path"]
        row = {"upstream_path": entry["upstream_path"], "release_path": path,
               "official_manifest_match": True}
        if path is not None:
            baseline = git(ROOT, "show", BASE + ":" + path)
            assert normalized(baseline) == normalized(official), path
            row["baseline_byte_identical"] = baseline == official
            current = (ROOT / path).read_bytes()
            if path == REDACTED:
                lines = baseline.decode("utf-8").splitlines()
                for index, line in enumerate(lines):
                    if line.startswith('os.environ["OPENELECTRICITY_API_KEY"] ='):
                        lines[index:index + 1] = [
                            'if not os.environ.get("OPENELECTRICITY_API_KEY"):',
                            '    raise RuntimeError("Set OPENELECTRICITY_API_KEY externally before running this acquisition example")']
                        break
                else:
                    raise AssertionError("Expected original acquisition assignment missing")
                assert normalized(current) == ("\n".join(lines) + "\n").encode(), path
                row["workspace_status"] = "verified_credential_externalization_only"
            else:
                assert normalized(current) == normalized(official), path
                row["workspace_status"] = "official_content_match"
        rows.append(row)
    changes = git(ROOT, "diff", BASE, "--name-only").decode().splitlines()
    original_changes = [p for p in changes if not p.startswith("research/e0/")]
    assert original_changes == [REDACTED], original_changes
    result = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.perf_counter() - start, "base_sha": BASE,
              "upstream_sha": OFFICIAL, "manifest_rows": len(rows),
              "exported_files": sum(r["release_path"] is not None for r in rows),
              "baseline_byte_identical": sum(r.get("baseline_byte_identical", False) for r in rows),
              "original_file_changes": original_changes, "files": rows}
    (ROOT / "research/e0/source_integrity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))


if __name__ == "__main__":
    main()
