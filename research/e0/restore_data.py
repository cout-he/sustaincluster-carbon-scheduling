"""Restore only official SustainCluster inputs from the pinned local Git objects.

Run from the repository root; no network, third-party packages or model weights
are needed. Existing files are accepted only when their bytes match upstream.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


UPSTREAM_SHA = "3f6ea95cb835b89ba50b0ef76d66d14b8037643e"
UPSTREAM_URL = "https://github.com/HewlettPackard/sustain-cluster"
ARCHIVE = "data/workload/alibaba_2020_dataset/result_df_full_year_2020.zip"
WORKLOAD_MEMBER = "result_df_full_year_2020.pkl"
INPUT_GROUPS = {"carbon_intensity", "electricity_prices", "network_cost", "weather", "workload"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_hash(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def write_matching_file(path: Path, data: bytes) -> str:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing input differs from the pinned official data: {path}")
        return "already_present_and_verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "restored"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=root.parent / "upstream")
    args = parser.parse_args()
    source = args.source_repo.resolve()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(source), *args])

    if git("rev-parse", UPSTREAM_SHA).decode().strip() != UPSTREAM_SHA:
        raise ValueError("Pinned upstream commit is unavailable")
    records = []
    tree = git("ls-tree", "-r", "--long", UPSTREAM_SHA, "data").decode("utf-8")
    archive_data = None
    archive_record = None
    for line in tree.splitlines():
        meta, relative_path = line.split("\t", 1)
        _, kind, blob, size = meta.split()
        parts = Path(relative_path).parts
        if (kind != "blob" or parts[1] not in INPUT_GROUPS
                or Path(relative_path).suffix.lower() not in {".csv", ".json", ".zip"}):
            continue
        data = git("cat-file", "blob", blob)
        if len(data) != int(size) or git_blob_hash(data) != blob:
            raise ValueError(f"Official Git object verification failed: {relative_path}")
        previous_source_file = source / relative_path
        source_file_status = "absent"
        if previous_source_file.is_file():
            source_file_status = (
                "matches_official_blob" if git_blob_hash(previous_source_file.read_bytes()) == blob
                else "differs_from_official_blob_not_used"
            )
        status = write_matching_file(root / relative_path, data)
        record = {
            "path": relative_path,
            "source_path": relative_path,
            "source_commit": UPSTREAM_SHA,
            "git_blob": blob,
            "sha256": sha256(data),
            "bytes": len(data),
            "recovery_status": status,
            "upstream_worktree_check": source_file_status,
            "ignored_by_git": None,
        }
        records.append(record)
        if relative_path == ARCHIVE:
            archive_data, archive_record = data, record

    if archive_data is None:
        raise ValueError("Official Alibaba workload archive is missing")
    with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
        if archive.namelist() != [WORKLOAD_MEMBER]:
            raise ValueError("Unexpected official workload archive members")
        workload = archive.read(WORKLOAD_MEMBER)  # ZIP CRC is checked by zipfile.
    workload_path = str(Path(ARCHIVE).parent / WORKLOAD_MEMBER).replace("\\", "/")
    status = write_matching_file(root / workload_path, workload)
    records.append({
        "path": workload_path,
        "source_path": ARCHIVE,
        "source_commit": UPSTREAM_SHA,
        "git_blob": None,
        "source_archive_git_blob": archive_record["git_blob"],
        "source_archive_sha256": archive_record["sha256"],
        "archive_member": WORKLOAD_MEMBER,
        "sha256": sha256(workload),
        "bytes": len(workload),
        "recovery_status": status,
        "ignored_by_git": None,
    })
    paths = [record["path"] for record in records]
    ignored = subprocess.check_output(
        ["git", "-C", str(root), "check-ignore", "--stdin"],
        input=("\n".join(paths) + "\n").encode("utf-8"),
    ).decode("utf-8").splitlines()
    for record in records:
        record["ignored_by_git"] = record["path"] in ignored
    if not all(record["ignored_by_git"] for record in records):
        raise ValueError("A restored original input is not ignored by Git")
    counts = Counter(record["path"].split("/")[1] for record in records)
    manifest = {
        "schema_version": 1,
        "purpose": "E0 official experiment-input restoration; no research-branch inputs or models",
        "source_url": UPSTREAM_URL,
        "source_commit": UPSTREAM_SHA,
        "source_repository_at_execution": str(source),
        "target_repository_at_execution": str(root),
        "python_at_execution": sys.executable,
        "started_utc": started_utc,
        "duration_seconds": time.perf_counter() - started,
        "network_access_used": False,
        "binary_extraction_command": "git -C <source-repo> cat-file blob <git_blob>",
        "selection": "All tracked .csv, .json and .zip files below the five official data input groups",
        "original_blob_file_count": len(records) - 1,
        "original_blob_bytes": sum(record["bytes"] for record in records[:-1]),
        "derived_archive_member_count": 1,
        "total_local_files": len(records),
        "total_local_bytes": sum(record["bytes"] for record in records),
        "counts_by_group_including_archive_member": dict(sorted(counts.items())),
        "all_inputs_ignored_by_git": True,
        "all_original_inputs_match_pinned_git_objects": True,
        "files": records,
    }
    output = root / "research" / "e0" / "data_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
