# Reproduction contract

Run from the Git repository root. The historically tested environment was Ubuntu22.04 under WSL, Python3.10.22. [environment-tested.json](environment-tested.json) records the complete observed package inventory; [requirements-tested.txt](requirements-tested.txt) lists direct packages at their observed versions. These are inventories, not a tested clean-install lock: platform-specific Torch wheels, system dependencies and transitive resolution must be checked before claiming a new environment reproduces historical results. No installation/network test was performed during Git preparation. Original [requirements.txt](../../../requirements.txt) is retained.

## Source tests versus real-data experiments

```bash
python -m pytest tests/e1 -q
python -m pytest tests/e1 tests/test_model.py -q
```

The first command covers synthetic task/CI/weather fixtures and the actual physical adapter. The optional historical `smoke-007` regression explicitly skips if its local archive is unavailable. The second also runs the upstream actor/critic forward test. Neither command collects the network data script, multiregional environment test or training-wrapper test. Do not assert108passes on a fresh clone merely from the recorded historical count.

Real-data experiments require trusted external workload, CI and weather files.
No original upstream data/weight objects are inherited into this new Git history.
The original private repository retains them; the source-only publication excludes
them in every commit. See [publication README](../../../README.md) for acquisition,
source hashes and external absolute-path configuration. Software's MIT license does
not establish separate third-party dataset distribution rights. This release is
not self-contained for real-data runs. Its source tests use synthetic inputs.


The [evidence index](results/evidence-index.json) records data/source/output hashes and aggregate tables, not the full frozen cohort. Public summary CSVs alone cannot replay the historical experiment. For exact archived replay retain privately the recorded frozen inputs, physical configuration, all scientific files, source snapshot and matching dependencies. Alternatively rerun the documented driver using authorized matching source data into a new archive and validate the resulting hashes.

```bash
python -m experiments.e1.run --config "$E1_LOCAL_CONFIG" --output /outside-git/new-fifo-001
python -m experiments.e1.run --replay /outside-git/new-fifo-001 --output /outside-git/new-fifo-replay-001
python -m experiments.e1.temporal --config "$E1_LOCAL_CONFIG" --output /outside-git/new-temporal-001
python -m experiments.e1.temporal --replay /outside-git/new-temporal-001 --output /outside-git/new-temporal-replay-001
python -m experiments.e1.plot_temporal --experiment /outside-git/new-temporal-001 --output /outside-git/new-temporal-figures-001
```

Older E1-1B comparisons use `experiments.e1.compare --frozen-from <complete-archive> --threshold-g-per-kwh 200 --output <new-directory>`. Do not reconstruct old policy results using E1-2A's different CI preprocessing. Historical frozen import and strict replay have distinct source-version contracts.

Source checks use **file bytes**, not just Git HEAD. Git tracking membership and line-ending normalization can change source inventories/hashes after a commit or fresh checkout even when numerical code appears unchanged. This independent publication's `.gitattributes` enforces LF for text; it never touches private original production files or historical archives. Never rewrite old archive hashes or silently bypass strict checks; use the recorded source/environment for historical replay, or create a new labeled run and record fresh provenance. This preparation does not run new experiments or alter archived evidence.

Per-step energy is kW×0.25h; carbon is kWh×gCO2e/kWh÷1000 in kgCO2e. The boundary is upstream IT plus its HVAC field, not complete measured facility energy. All policies share the same integration and deadline/censoring rules.

Publication note: do not run the example default config without supplying external inputs. Use the local outside-Git YAML described in ../../../README.md. No historical provenance is rewritten.
