# E1-6B development-Validation campaign driver

`run_validation_campaign.py` is a local archival orchestration script. It changes no executor, policy, physical model, training source, test under `tests/e1`, or registered configuration. It does not train or select a checkpoint. The three development-Validation workloads were already frozen in E1-5B/E1-5C; Test has no entry point.

Before any real rollout, the driver requires the completed formal Train receipt: 12 registered models, all 20,480 steps, 245,760 total steps, no Validation training, no Test, and an unchanged source hash. It binds all 12 adaptive and three original PPO checkpoint hashes, checks the registered bundle manifest hashes, loads the three allowed Validation bundles, and writes `campaign.json`, `candidates.json`, the source snapshot and registered configuration. Each exact child command is in the campaign manifest.

Each pressure factor has the same 19 methods: EDF, EDF+Carbon, RiskOnly, frozen Task Full, four adaptive PPO groups times three seeds, and three historical three-action PPO models. Adaptive names are `plain_55101`, etc.; historical models use `legacy_3action_55101`, etc. Children call the shared `evaluate_adaptive_group` implementation. Each factor independently uses its slowest method's observation end and the existing physical idle-tail integration.

The default is two independent factor processes; `--workers 3` is permitted after the formal training processes have finished and memory is available. It does not alter any training budget or model. The driver stops scheduling on a child/artifact failure, retains all outputs, and rejects an existing output path. Sources, configuration and driver bytes stay frozen during execution. A two-hour per-factor watchdog is an execution safety cutoff, not a scientific observation cutoff; an interrupted run cannot pass.

The final gate checks all 57 completed results: unique method matrix, complete Drain, no remaining or deadline-censored tasks, identical input identity and three resource hours, common endpoint, IT+HVAC energy sum, deadline count/rate denominator, and stored artifact hashes. It rechecks inputs and checkpoint identities afterward. An incomplete result cannot produce a passing campaign receipt.

Complete action-tape replay is a separate fresh campaign. It checks the original campaign passed, uses the exact current source and frozen inputs, replays all 19 methods for each factor, and requires exact ledger/resource/energy/carbon scientific hashes. Replay is recorded-action reproducibility evidence, not an online algorithm or a new training run. It does not require reloading a model to generate decisions.

Commands below run inside the configured WSL Python environment, with the existing E1-5B runtime dependency directory on `PYTHONPATH`:

```bash
python research_notes/E1/E1-6B/run_validation_campaign.py --workers 3
python research_notes/E1/E1-6B/run_validation_campaign.py --mode replay \
  --archive research_notes/E1/E1-6B/validation-campaign-001 --workers 3
```

The script and these commands are ready; this driver review does not claim the real Validation experiments have run.

Synthetic contract tests ran in the configured WSL SustainCluster Python runtime:

```text
python -m pytest research_notes/E1/E1-6B/test_validation_driver.py -q \
  -p no:cacheprovider --basetemp research_notes/E1/E1-6B/validation-driver-tests-tmp-001
.............                                                            [100%]
13 passed in 2.71s
Exit code: 0
```

These tests check 19-method completeness, role/idle/completion/energy/deadline/resource-hour rejection, artifact tampering, formal gate refusal, exact replay refusal, and non-overwrite behavior. They are archival-driver tests, independent of the frozen production E1 test inventory. `--help` also completed with exit code 0 in WSL. The Windows Anaconda runtime lacks Gymnasium and is not the intended runtime for this script.

## Recorded preflight correction

The first real Validation preflight stopped before creating `validation-campaign-001`: the driver author had transcribed the Validation 1× manifest constant with an extra `b`. The real frozen bundle was unchanged and must not be rewritten to fit a driver constant. The original supervisor failure log is retained.

Two additional tests were added before correcting the notes-only driver: all typed hashes must contain exactly 64 lowercase hexadecimal characters, and all three registered bundle manifests plus three original checkpoint hashes must match the actual existing files and recorded checkpoint summaries. Both tests failed for the erroneous 1× constant (`validation-driver-hash-red-001.txt`). After correcting that constant to the file's actual SHA256, the full archival driver tests passed: `15 passed in 4.09s`, exit code 0 (`validation-driver-green-002.txt`). The other manifest/checkpoint constants passed their real-file checks. No original data, provenance, core implementation, registered configuration, or test under `tests/e1` was altered by the correction.

## Final measured campaigns

After the recorded SHA transcription correction, evaluate and replay both completed with exit0: three factors, 19 methods each, 57 full scientific results. Every finite cohort completed under the same factor-specific inputs and common 2023-07-10 12:45 UTC endpoint, with physical idle integration and equal resource hours. All 57 replay scientific hashes match exactly. The first preflight failure log remains local; no bundle/provenance was rewritten.
