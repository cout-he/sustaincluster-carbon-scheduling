# SustainCluster Carbon Scheduling — E1 research

This research branch builds on [HewlettPackard/sustain-cluster](https://github.com/HewlettPackard/sustain-cluster), baseline `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`. The original [MIT license](LICENSE), copyright and [upstream README](UPSTREAM_README.md) are retained. Research code is in [experiments/e1](experiments/e1/README.md); production scheduling and physical equations are unchanged.

Completed stages: [E1-0 audit](docs/research/e1/E1-0.md), [E1-1A execution foundation](docs/research/e1/E1-1A.md), [E1-1B baselines](docs/research/e1/E1-1B.md), and [E1-2A temporal shifting validation](docs/research/e1/E1-2A.md). Those historical stages included no RL training; later stages are documented below.

The three policies share a single executor, task specifications, resources, backfilling and IT/HVAC model. In the E1-2A finite cohort all policies completed 9943 tasks with zero synthetic-deadline violations. Over a common 56.75h accounting interval including idle tails, Carbon-Threshold emitted 97.013 kgCO2e (0.742%) less than FIFO/EDF, with 21.799min more mean waiting and a 3h longer drain. This one-window model result does not establish general superiority or steady-state benefits.

See [reproduction and data boundaries](docs/research/e1/REPRODUCIBILITY.md), [tested environment](docs/research/e1/environment-tested.json) and [version workflow](docs/research/e1/VERSION_CONTROL.md). Full local experiment archives, task rows, raw datasets and weights are excluded from new research commits. Summary CSVs and hashes are published separately from local archival evidence. This publication has a new curated-source history and excludes the original data/weight objects. The original upstream SHA is attribution, not this repository main SHA. See README.md for external data setup.

From the repository root, in the compatible Python environment:

```bash
python -m pytest tests/e1 -q
python -m experiments.e1.run --config "$E1_LOCAL_CONFIG" --output /outside-git/new-fifo-001
python -m experiments.e1.temporal --config "$E1_LOCAL_CONFIG" --output /outside-git/new-temporal-001
python -m experiments.e1.temporal --replay /outside-git/new-temporal-001 --output /outside-git/new-temporal-replay-001
python -m experiments.e1.plot_temporal --experiment /outside-git/new-temporal-001 --output /outside-git/new-temporal-figures-001
```

Real-data runs require the trusted workload pickle, CI and weather specified by the configuration. Unit tests generate synthetic inputs; the optional local historical-artifact regression skips when that archive is absent. Never overwrite existing experiment directories. Old archived runs require their recorded source bytes and dependencies; publishing Git commits does not retroactively change their provenance.

Before real-data commands use an outside-Git YAML with external absolute input paths as described in [publication README](README.md). Historical108testpasses refer to the private original checkout, not a new clean-install claim.

## E1-4A: CARS feasibility and cap fallback diagnosis

The [E1-4A report](docs/research/e1/E1-4A/E1-4A-report.md) isolates `risk_cap_only` and `risk_shadow_only` without changing the original executor, CARS policies or power equations. Both 1.5× and 2.0× frozen E1-3 cohorts completed under five policies with a common drain/idle accounting endpoint. Shadow-only changed deadline misses by +7/−10 relative to RiskOnly; cap-only by 0/−9. Full remained worse than cap-only by 35/25 misses. These already-seen-window diagnostics do not establish generalization or that Shadow is always ineffective.

See [common metrics](docs/research/e1/E1-4A/common_metrics.csv), [guard definitions and counts](docs/research/e1/E1-4A/guard_counts.csv), [reproduction commands and archive boundaries](docs/research/e1/E1-4A/E1-4A-reproducibility.md), and [validation](docs/research/e1/E1-4A/validation.md). Original research tests: 225 passed; publication tests: 223 passed, 2 private-archive-dependent tests skipped. Strict replay of 231 scientific outputs and byte regression of 72 old-policy outputs were verified in the original research environment. Historical input archives and original source bytes remain external requirements; this curated clone is not a self-contained historical replay bundle.


## E1-5A: verified stepwise Gymnasium environment

[E1-5A summary](docs/research/e1/E1-5A/E1-5A-summary.md) documents one persistent execution kernel and `Discrete(3)` modes: existing EDF, EDF+Carbon, RiskOnly. No policy training or learned task-level delay budget is included. See [API/40-feature observation/reward](docs/research/e1/E1-5A/E1-5A-rl-interface.md), [fixed-mode equivalence](docs/research/e1/E1-5A/E1-5A-fixed-action-equivalence.md), [tests](docs/research/e1/E1-5A/E1-5A-tests.md), and [proposed data split](docs/research/e1/E1-5A/E1-5A-data-split.md).

Run `python -m pytest -q tests/e1 -p no:cacheprovider` in the compatible environment. Historical `python -m experiments.e1.rl_validation --output <new-output-directory>` requires external sealed E1-3/E1-4A archives and compatible original non-kernel source bytes. This source-only publication intentionally excludes those raw inputs, full ledgers and archives; synthetic tests run without them and two private-archive tests skip. Source refactoring does not rewrite old provenance or weaken strict historical replay.


## E1-5B: ordinary PPO baseline

Three fixed-budget SB3 PPO candidates use the unchanged E1-5A 40-feature, three-mode environment. See [delivery](docs/research/e1/E1-5B/E1-5B-delivery.md), [all Validation results](docs/research/e1/E1-5B/E1-5B-validation-report.md), and [reproduction with external data](docs/research/e1/E1-5B/E1-5B-reproduce.md). Test remains unevaluated. Ordinary mode selection is a baseline, not the proposed task-level safety method. Full weights, frozen cohorts and training logs remain local.

## E1-5C: workload pressure and SLA feasibility diagnosis

The [E1-5C report](docs/research/e1/E1-5C/E1-5C-report.md) compares three unchanged baselines in six Train/Validation load groups. Validation 1×/1.5× are new exploratory diagnostics; the E1-5B model-selection result is unchanged. All finite cohorts complete with Drain and common idle-inclusive accounting. Train 1.5× and Validation 1× provide realized references below 5% DMR despite resource competition. In Validation 2×, the certified lower bound is only 177/17416 (1.0163%); whether 5% is achievable remains undetermined.

See [all baseline metrics](docs/research/e1/E1-5C/summary-001/baseline-comparison.csv), [pressure evidence](docs/research/e1/E1-5C/pressure-evidence.md), [necessary-condition method](docs/research/e1/E1-5C/feasibility-method.md), [data identities and boundaries](docs/research/e1/E1-5C/E1-5C-data-freeze.md), [reproduction](docs/research/e1/E1-5C/E1-5C-reproduce.md), and [tests](docs/research/e1/E1-5C/E1-5C-tests.md). The [task-level budget and RL interface](docs/research/e1/E1-5C/next-stage-design.md) is a proposal awaiting review; no new algorithm or formal PPO training is included. Test remains sealed. Full raw inputs, historical archives and checkpoints are external requirements and are not distributed here.

## E1-6B: adaptive four-action PPO and risk ablations

The [E1-6B report](docs/research/e1/E1-6B/E1-6B-report.md) adds an independent 83-feature / four-action controller while preserving the old 40-feature / three-action environment and existing execution/physics. Twelve models (four matched groups × three seeds) each trained for 20,480 Train-only steps. All 57 development-Validation trajectories complete, share idle-inclusive accounting endpoints, and pass exact scientific replay plus independent resource/energy/carbon and filter-denominator audits. See [all 57 metrics](docs/research/e1/E1-6B/tables/all-methods.csv), [paired effects](docs/research/e1/E1-6B/tables/factorial-effects-summary.csv), [tests](docs/research/e1/E1-6B/E1-6B-tests.md), [interface and reward](docs/research/e1/E1-6B/E1-6B-interface-and-method.md), and [reproduction](docs/research/e1/E1-6B/E1-6B-reproduce.md).

Negative results remain: new continuous risk state does not provide stable benefit; the frozen v1 heuristic filter trades fewer high-pressure misses for more carbon and does not guarantee deadlines. All new models fail the unchanged 5% target at 1.5×/2×. At 1×, Plain/Shield/Combined pass for all three seeds; State passes only two. The E1-6A Task Full result (+341 misses and +33.714kg versus RiskOnly at 2×) is preserved. The old three-action models are external baselines with multiple training/interface differences, not a strict action-space-only ablation.

Original research regression: 532 passed. Publication regression: 530 passed, two historical private-archive-dependent skips. Test remains sealed. Frozen data, checkpoints, full action/ledger/diagnostic archives and figures are external local dependencies and are not distributed; this repository is not a self-contained historical replay bundle. The [public evidence manifest](docs/research/e1/E1-6B/public-evidence-manifest.json) records curated file, input, model and scientific result identities without rewriting original provenance.


## E1-7A: bounded scheduling headroom and carbon diagnostics

See [E1-7A scientific decision](docs/research/e1/E1-7A/E1-7A-scientific-decision.md), [A/B/C evidence](docs/research/e1/E1-7A/README.md), [tests](docs/research/e1/E1-7A/E1-7A-tests.md), and [external-data reproduction](docs/research/e1/E1-7A/E1-7A-reproduce.md). Train-only, 23 bounded oracle cases and 12 single-defer pairs; no new research PPO training or Test access. One local A2 deadline improvement (11 to 7) increases real full-window carbon by 0.908 kg; all 12 pairs keep the deadline outcome set unchanged, with 4 carbon reductions, 6 increases and 2 exact-zero copy swaps. These results do not prove a full-cluster optimum or generalization. Complete original regression: 677 passed; publication: 675 passed, two private-archive-dependent skips. Original physics, schedulers, checkpoints and negative results are preserved.

Only code, tests, registered configuration, self-written reports, small summaries and hashes are published. Complete task/environment/solver archives, figures, weights and papers remain external. The next priority is to validate lawful full-execution-window total-carbon estimation; it is not implemented or claimed effective here.


## E1-7B: carbon-gain candidate V1 and independent RL interface

The [E1-7B delivery](docs/research/e1/E1-7B/E1-7B-delivery.md), [method](docs/research/e1/E1-7B/E1-7B-method.md), [all 48 fixed-policy results](docs/research/e1/E1-7B/tables/comparison.csv), and [ablation report](docs/research/e1/E1-7B/E1-7B-ablation.md) evaluate causal CI forecasting, a Train-frozen IT/HVAC surrogate, full occupied-interval incremental carbon, and the unchanged RiskOnly-style deadline guard. Two optional read-only executor hooks expose current environment scalars and completed interval measurements; production equations and old scheduling decisions are unchanged. [54-feature / three-action Gym](experiments/e1/carbon_gain_rl_env.py) reuses the sole executor and existing reward. This stage includes only a 16-step toy PPO API smoke, no new research PPO training; Test remains sealed.

All 48 finite cohorts complete with common idle-inclusive accounting. M2 total-power gain lowers carbon relative to IT-only M1 in four of six groups, increases it in two, and lengthens JCT in all six. At development Validation 1.5×/2×, RiskOnly has both fewer misses and less carbon than M2. Of ten nonzero preselected diagnostic labels, IT-only and total estimates each match only four signs. These negative results do not support claiming that HVAC estimation or the full candidate is stably superior. The 5% admission target is unchanged.

Original regression: 729 passed; publication: 727 passed, two private-archive-dependent skips. The original environment verifies 144 historical scientific files and 375 final replay files byte for byte; fixed RL actions match the corresponding rule ledgers and energy/carbon exactly. See [tests](docs/research/e1/E1-7B/E1-7B-tests.md), [public test receipt](docs/research/e1/E1-7B/evidence/test-receipt.json), [reproduction](docs/research/e1/E1-7B/E1-7B-reproduce.md), and [public layout/data boundaries](docs/research/e1/E1-7B/PUBLICATION.md). Small CSVs and the [standalone six-figure plotter](docs/research/e1/E1-7B/plot_public_figures.py) are included; raw inputs, complete ledgers, original archives, images, checkpoints and papers remain external. Frozen surrogate coefficients are simple Train-calibrated parameters, not neural model weights. This clone is not a self-contained historical experiment bundle. Formal candidate training awaits review.
