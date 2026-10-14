# 复现命令与E1-8接口

科研运行继续在原科研工作区；发布副本仅作版本化。Python3.10.22、numpy2.2.6、pandas2.3.3、Gymnasium1.4.0、SB3 2.9.0及原SustainDC依赖保持不变。无新增神经预测器或依赖升级。WSL使用已有sustaincluster环境，设置PYTHONDONTWRITEBYTECODE=1、GIT_OPTIONAL_LOCKS=0、OMP/OPENBLAS/MKL_NUM_THREADS=1；需要时将既有SB3 runtime-deps与OR-Tools solver-runtime加入PYTHONPATH。Notes脚本还需要仓库根目录在PYTHONPATH。

以下为实际命令主体（在科研根目录，输出必须不存在）：

```bash
python -m experiments.e1.carbon_gain_experiment calibrate --output research_notes/E1/E1-7B/calibration-003
python -m experiments.e1.carbon_gain_experiment evaluate --freeze research_notes/E1/E1-7B/calibration-003/freeze.json --split train --output research_notes/E1/E1-7B/train-002
python -m experiments.e1.carbon_gain_experiment evaluate --freeze research_notes/E1/E1-7B/calibration-003/freeze.json --split validation --output research_notes/E1/E1-7B/validation-001
python -m experiments.e1.carbon_gain_experiment replay --previous research_notes/E1/E1-7B/train-002 --output research_notes/E1/E1-7B/train-replay-001
python research_notes/E1/E1-7B/validate_cgd_interface.py --freeze research_notes/E1/E1-7B/calibration-003/freeze.json --reference research_notes/E1/E1-7B/train-002 --output research_notes/E1/E1-7B/rl-interface-003
python research_notes/E1/E1-7B/check_registered_pairs.py --freeze research_notes/E1/E1-7B/calibration-003/freeze.json --output research_notes/E1/E1-7B/paired-analysis-002
python research_notes/E1/E1-7B/build_analysis.py --output research_notes/E1/E1-7B/analysis-002
python research_notes/E1/E1-7B/audit_power_domain.py --output research_notes/E1/E1-7B/power-domain-001
python -m pytest tests/e1 tests/test_model.py -q
```

原始工作负载、CI/weather、E1-5B/E1-5C冻结包和历史完整基线归档不公开，真实实验并非完全自包含。配置逐个固定外部bundle路径和SHA；只有自行合法准备这些数据/对应历史物理回放后才能运行真实命令。原始数据来源、LICENSE及上游说明保留。已有freeze_bundle工具能生成授权Train/Validation输入，但历史身份必须一致，不能随意换数据后仍声称相同结果。Test白名单拒绝。

发布小CSV可独立重画六图，不需要私有账本：

```bash
python docs/research/e1/E1-7B/plot_public_figures.py --tables docs/research/e1/E1-7B/tables --output experiments/e1/artifacts/e1-7b-public-figures-001
```

E1-8真实可训练入口为新类与已测SB3 API，示意如下，本轮没有执行正式learn：

```python
import json
from pathlib import Path
from stable_baselines3 import PPO
from experiments.e1.config import ROOT
from experiments.e1.carbon_gain_experiment import CONFIG, load_frozen, load_registered_bundle, validate_protocol
from experiments.e1.carbon_gain_rl_env import CarbonGainSchedulingEnv
from experiments.e1.online_ci_forecast import extract_prehistory
from experiments.e1.online_power_surrogate import PowerSurrogate
protocol = validate_protocol(json.loads(CONFIG.read_text(encoding='utf8')))
bundle = load_registered_bundle(protocol, 'train', '1')
frozen = load_frozen(Path('research_notes/E1/E1-7B/calibration-003/freeze.json'))
raw_config = json.loads((ROOT / protocol['inputs']['train']['1']['path'] / 'config.json').read_text(encoding='utf8'))
legal_pre_start_history = extract_prehistory(
    ROOT / raw_config['paths']['carbon'], bundle.workload.start,
    expected_sha256=bundle.environment.carbon_source_sha256)
env = CarbonGainSchedulingEnv(bundle.workload, bundle.environment, bundle.capacity,
    bundle.power_factory, bundle.parameters, PowerSurrogate.from_dict(frozen['model']),
    prehistory=legal_pre_start_history, moderate_threshold_kg=frozen['moderate_threshold_kg'],
    reward_config=bundle.reward_config, normalization=bundle.normalization, stop_at_drain=True)
model = PPO('MlpPolicy', env, device='cpu')
# E1-8审核并冻结预算/网络/种子/奖励后才正式learn，另存54维新schema模型。
```

旧rl_train.py仍属于40维PPO-v0，不应直接覆盖或误标。训练回报只积分交互期，真实评估必须完整Drain后统一extend_idle公共终点。全实验输出使用新的目录，重复已有路径会报FileExistsError，不覆盖任何历史。
