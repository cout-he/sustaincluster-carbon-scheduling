# SustainCluster — source-only publication

This repository starts a **new Git history** from a curated source snapshot of
[HewlettPackard/sustain-cluster](https://github.com/HewlettPackard/sustain-cluster)
at upstream commit **3f6ea95cb835b89ba50b0ef76d66d14b8037643e**. Its `main` commit
has a different SHA and no parent from that upstream repository. See
[snapshot provenance](SOURCE_SNAPSHOT.json), the original [MIT LICENSE](LICENSE)
and preserved [upstream README](UPSTREAM_README.md).

Included: upstream Python modules (Task, cluster/environment, resource execution,
IT/HVAC models, utilities, rewards and data preparation code), runtime configs,
requirements and source attribution. The E1 implementation and research summaries
are added on `research/e1-carbon-deadline`; no physical equations are redesigned.

Excluded from **all commits and Git objects**: workload ZIP/pickle/CSV, CI/weather
measurements, checkpoints/model weights, notebooks, plots and generated/built
documentation, experiment archives, caches and secrets. The inherited GitHub
workflow requiring the removed ZIP is not exported. This is not the original
repository's Git history and not a complete upstream experiment distribution.

## External data and environment

The software license is retained; independently check each dataset's terms before
obtaining it. **This publication is not self-contained for real-data experiments.**
Use an existing trusted local data directory, or obtain matching data directly
from the upstream snapshot/authorized original providers. Do not commit inputs.
For the existing prepared workload, obtain the upstream snapshot's
`data/workload/alibaba_2020_dataset/result_df_full_year_2020.zip` separately and
extract its trusted pickle **outside this Git repository**. The data-preparation
source and original usage notes remain under `data/workload/`. No automatic data
download or extraction is performed by this publication step.

E1 requires these external files:

```text
<DATA_ROOT>/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl
<DATA_ROOT>/carbon_intensity/US-CAL-CISO/2023/US-CAL-CISO_2023_hourly.csv
<DATA_ROOT>/weather/US-CAL-CISO/2023.json
```

CI/weather can be obtained individually from the same pinned upstream snapshot;
its base URL is
`https://raw.githubusercontent.com/HewlettPackard/sustain-cluster/3f6ea95cb835b89ba50b0ef76d66d14b8037643e/`.
For example append `data/carbon_intensity/US-CAL-CISO/2023/US-CAL-CISO_2023_hourly.csv`.
Recorded source hashes are in the research branch's
`docs/research/e1/results/evidence-index.json`. Verify trusted input identities;
modified/reprocessed inputs constitute a different experiment.

Use a compatible Python3.10 environment and retained `requirements.txt`.
The research branch supplies an observed tested package inventory, not a verified
clean-install lock. Torch/system packages need compatible platform distributions.

On the research branch, make a **local config outside Git** from
`experiments/e1/configs/single_dc.yaml`, overriding only the three input paths:

```python
import os, pathlib, yaml
data = pathlib.Path(os.environ['E1_DATA_ROOT']).resolve()
cfg = yaml.safe_load(pathlib.Path('experiments/e1/configs/single_dc.yaml').read_text())
cfg['workload']['path'] = str(data/'workload/alibaba_2020_dataset/result_df_full_year_2020.pkl')
cfg['environment']['carbon_path'] = str(data/'carbon_intensity/US-CAL-CISO/2023/US-CAL-CISO_2023_hourly.csv')
cfg['environment']['weather_path'] = str(data/'weather/US-CAL-CISO/2023.json')
pathlib.Path(os.environ['E1_LOCAL_CONFIG']).write_text(yaml.safe_dump(cfg), encoding='utf-8')
```

Set `E1_DATA_ROOT` to your external data directory and `E1_LOCAL_CONFIG` to an
outside-repository YAML path before running that snippet. Keep SLA, resources,
time rules and model configuration unchanged for a fair comparison. Then:

```bash
python -m pytest tests/e1 tests/test_model.py -q
python -m experiments.e1.run --config "$E1_LOCAL_CONFIG" --output /outside-git/new-fifo-001
python -m experiments.e1.temporal --config "$E1_LOCAL_CONFIG" --output /outside-git/new-e1-2a-001
python -m experiments.e1.temporal --replay /outside-git/new-e1-2a-001 --output /outside-git/new-e1-2a-replay-001
python -m experiments.e1.plot_temporal --experiment /outside-git/new-e1-2a-001 --output /outside-git/new-e1-2a-figures-001
```

E1 unit tests synthesize task/CI/weather inputs and use the actual upstream power
adapter. The optional historical archive regression skips without its private
archive. Upstream `test_env_setup.py` also needs external multiregional inputs;
`test_training_loop.py` is not an E1 acceptance test and is not invoked here.

## Provenance and publication workflow

Keep full historical E1 archives private and **unchanged**: their original HEAD,
source bytes, dependencies and hashes remain valid records of that environment.
This new Git history, source inventory and line endings can differ; do not relabel
old experiments or change their hashes to match this publication. New experiments
must record their own committed source/config/input/output identities.

Only the independent publication directory may be pushed to the personal GitHub
repository after explicit approval. The original `upstream` directory and its
branches must not be pushed. No upstream fetch/clone, force-push or mirror is part
of this procedure. Local commit preparation does not authorize the first push.


## E1-5A: verified stepwise Gymnasium environment

[E1-5A summary](docs/research/e1/E1-5A/E1-5A-summary.md) documents one persistent execution kernel and `Discrete(3)` modes: existing EDF, EDF+Carbon, RiskOnly. No policy training or learned task-level delay budget is included. See [API/40-feature observation/reward](docs/research/e1/E1-5A/E1-5A-rl-interface.md), [fixed-mode equivalence](docs/research/e1/E1-5A/E1-5A-fixed-action-equivalence.md), [tests](docs/research/e1/E1-5A/E1-5A-tests.md), and [proposed data split](docs/research/e1/E1-5A/E1-5A-data-split.md).

Run `python -m pytest -q tests/e1 -p no:cacheprovider` in the compatible environment. Historical `python -m experiments.e1.rl_validation --output <new-output-directory>` requires external sealed E1-3/E1-4A archives and compatible original non-kernel source bytes. This source-only publication intentionally excludes those raw inputs, full ledgers and archives; synthetic tests run without them and two private-archive tests skip. Source refactoring does not rewrite old provenance or weaken strict historical replay.
