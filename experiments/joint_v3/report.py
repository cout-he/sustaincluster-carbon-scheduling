"""Render truthful, refreshable delivery reports from actual campaign artifacts."""
import argparse
import csv
import json
from pathlib import Path
from .analyze import analyze
from .checkpoint import atomic_json


def refresh(campaign, docs):
    campaign, docs = Path(campaign).resolve(), Path(docs).resolve()
    protocol = json.loads((campaign/'protocol.json').read_text(encoding='utf8'))
    statusfile = campaign/'pipeline-status.json'
    status = json.loads(statusfile.read_text()) if statusfile.exists() else dict(stage='prepared', jobs={})
    evidence, lines = [], []
    for job in protocol['jobs']:
        training = campaign/job['training']
        resultfile, progressfile = training/'run-result.json', training/'progress.json'
        result = json.loads(resultfile.read_text()) if resultfile.exists() else {}
        progress = json.loads(progressfile.read_text()) if progressfile.exists() else {}
        steps, episodes = result.get('physical_steps', progress.get('physical_steps', 0)), result.get('completed_episodes', progress.get('completed_episodes', 0))
        phase = status.get('jobs', {}).get(job['job_id'], {}).get('stage', 'pending')
        evidence.append(dict(job=job['job_id'], seed=job['seed'], variant=job['variant'], phase=phase,
            physical_steps=steps, completed_episodes=episodes, result=result or None,
            checkpoint_pointers={n:json.loads((training/'checkpoints'/(n+'.json')).read_text())
                for n in ('latest', 'best_feasible', 'best_deadline', 'final') if (training/'checkpoints'/(n+'.json')).exists()}))
        lines.append(f"| {job['variant']} | {job['seed']} | {phase} | {steps:,}/65,536 | {episodes} |")
    analysis = analyze(campaign)
    from .monitor import plot
    for job in protocol['jobs']:
        training = campaign/job['training']
        if (training/'episodes.csv').exists():
            plot(training)
    with (campaign/'analysis/aggregate.csv').open(encoding='utf8', newline='') as stream:
        aggregates = list(csv.DictReader(stream))
    actual_table = ['| 版本/方法 | 实际种子数 | 总碳kg | 增量碳kg | DMR |',
                    '|---|---:|---:|---:|---:|']
    for row in aggregates:
        def display(key):
            value = row.get(key+'_mean')
            deviation = row.get(key+'_std')
            return '缺失' if not value else f'{float(value):.6g}'+(f' ± {float(deviation):.4g}' if deviation else '')
        actual_table.append(f"| {row['variant']}/{row['comparison_method']} | {row['n']} | "
            f"{display('carbon_kg')} | {display('incremental_carbon_kg')} | {display('dmr')} |")
    if not aggregates:
        actual_table.append('| 尚无完成的正式评价 | 0 | 缺失 | 缺失 | 缺失 |')
    conclusions = []
    full = {r['comparison_method']:r for r in aggregates if r['variant'] == 'full'}
    if 'B5' in full:
        for baseline in ('B1', 'B3'):
            if baseline in full and full[baseline]['n'] == full['B5']['n']:
                for metric, label in (('carbon_kg', '总碳'), ('incremental_carbon_kg', '增量碳')):
                    if not full[baseline].get(metric+'_mean') or not full['B5'].get(metric+'_mean'):
                        continue
                    base, proposed = float(full[baseline][metric+'_mean']), float(full['B5'][metric+'_mean'])
                    if base:
                        conclusions.append(f'B5相对{baseline}的{label}减排率（均值之比，描述性）为{(base-proposed)/base*100:.4f}%。')
    conclusion = ' '.join(conclusions) or '当前不能确认是否超过V2、Immediate+18°C或是否存在额外计算—制冷协同收益。'
    docs.mkdir(parents=True, exist_ok=True)
    training_text = f'''# JointRL-V3训练交付记录

本文件由真实运行记录生成；当前流水线状态：`{status['stage']}`。未结束的训练、未生成的对照与消融均不计为完成。

## 冻结身份与预算

- 源码身份：`{protocol['source_identity']}`。
- 场景池manifest SHA256：`{protocol['pool_manifest_sha256']}`。
- 随机种子：20261017、20261018、20261019。
- 完整方法及五个机制消融分别训练，每组3个种子，共18个有限作业。每作业目标65,536物理步、上限21,600秒；数据加载、预测器拟合、辅助初始化、训练和开发选择均计时。仅提交完整Episode；余量不足时丢弃尾部，实际步数可能小于目标。
- 只用Train拟合CI预测器；只用两条未增强June Development场景选择模型。最终对照使用这两条场景的3压力×2 SLA预设敏感性矩阵；这些增强不是独立实测。
- July未用于本轮训练/选择/最终检查；September未读取。

| 训练版本 | 种子 | 当前阶段 | 已提交物理步/目标 | 完整Episode |
|---|---:|---|---:|---:|
{chr(10).join(lines)}

## 命令与恢复

在已审计的WSL Python环境、仓库根目录执行：

```bash
export PYTHONPATH=/mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5B/runtime-deps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m experiments.joint_v3.pipeline --run --protocol {campaign}/protocol.json --jobs 3
python -m experiments.joint_v3.report --campaign {campaign} --docs docs/research/joint
```

流水线恢复拒绝变动源码/配置/数据身份，也拒绝重复启动仍存活的工作进程。单作业精确恢复命令为 `python -m experiments.joint_v3.train --pool research_notes/joint_v2/scenarios-001 --output <原training目录> --config <原冻结config文件> --resume`。不增加原累计时间/步数预算。

不可变Checkpoint保存在各作业`training/checkpoints`；指针包含权重字节、源码、配置与场景身份。完整状态包括两个Actor、四个条件价值输出及优化器、RNG、场景采样、六组约束系数、固定课程状态、教师衰减进度、预测器和完整Episode进度。当前采用固定学习率，配置与Adam状态共同保存其身份。

## 验收证据与限制

工程验收acceptance-001完成2个Train Episode、241物理步，处理7,892个任务，DMR0、无温度违规。此短程结果不属于正式训练，也不用于方法优越性或最终模型选择。验收预测器去重得到16条Train环境、1,536条CI观察；时间隔离拟合1,108条样本、校准281条完整残差向量，校准未来CI RMSE29.916 gCO₂/kWh。64个实际制冷验收状态均在18°C取得最低功率；正式初始化目标为4,096条去重状态，实际数量及不足记录在各作业pretraining-report.json。

风险/碳估计可信覆盖率可能很低；超出队列/时隙预算时保守回退会显著限制任务柔性。正式训练记录原因分布、估计耗时、教师覆盖及实际PPO更新；不能将这些回退掩饰为可靠预测。June Development已在V2多次使用，不是独立泛化证明。65,536步是有限预算目标，不代表已收敛或必然优于V2。

训练收敛、碳排/DMR、KL/梯度/熵、组系数、预测误差、制冷分布和最佳可行模型均以各作业CSV、episode-evidence.json和Checkpoint指针为准。当前未完成项见状态表，不补造结果。

## 图表与数据

`training/episodes.csv`为真实训练曲线数据，`python -m experiments.joint_v3.monitor --output <training目录>`生成PDF/SVG/PNG。最终对照图与源数据位于本地campaign/analysis。所有数值图只画真实记录；缺失种子不补值，只有3个真实种子均完成才给sample std。
'''
    (docs/'JointRL-V3-training-delivery.md').write_text(training_text, encoding='utf8')
    comparison = campaign/'analysis/aggregate.csv'
    results_text = f'''# JointRL-V3实验结果

当前完成的正式Development评价文件数：**{analysis['evaluation_files']}**；实际场景结果行数：**{analysis['scenario_rows']}**。流水线状态：`{status['stage']}`。没有正式结果时，不提供减碳率、优势判断或mean±std。

## 冻结比较

| 方法 | 任务控制 | 制冷控制 | 状态 |
|---|---|---|---|
| B0 | 资源允许即EDF启动 | 整个固定窗口22°C | 以analysis实际结果为准 |
| B1 | 资源允许即EDF启动 | 整个固定窗口18°C | 以analysis实际结果为准 |
| B2 | EDF立即启动 | V3 Cooling Actor | 以analysis实际结果为准 |
| B3 | 原V2第52轮Task Actor | 原V2第52轮Cooling Actor | 原权重只读评价 |
| B4 | V3 RAC Task Actor | 固定18°C | 以analysis实际结果为准 |
| B5 | V3 RAC Task Actor | V3 Cooling Actor | 以analysis实际结果为准 |

学习方法复用V2在排空后逐拍降至18°C的算子，并核算其电量；固定温度基线在整个运营窗口保持指定温度。所有方法共用相同任务、物理模型、资源容量、CI、天气与窗口。碳排主表包括全固定窗口总碳；增量碳统一相对于同一idle18背景，两个减排率分母分别计算。

五个消融分别训练：无队列风险、无边际碳特征、无鲁棒修正、独立局部Critic、无教师。每版本3个种子、同预算，不以测试时关闭模块替代重新训练。

## 已完成评价的实际数值

{chr(10).join(actual_table)}

## 真实结果入口

- 汇总表：`{comparison}`。
- 分种子/分场景表、来源哈希与缺失说明：campaign/analysis。
- 碳、增量碳、DMR与IT/HVAC分解图：有完整实际评价后生成；没有结果时保持缺失。
- 多压力/SLA为同源增强场景，聚合结果不把它们视为独立实测。重复基线由算法种子产生的标准差通常为0，不能误称数据不确定性。
- 风险预测误差依据实际完成/违约结果；碳收益符号误差依据执行后独立真实CI/天气校准。已知队列反事实标签与最终实际策略减排是不同指标。

{conclusion} 即便B0减排显著，也需检查B1、B2、B4与分项结果及各场景DMR。最终机制结论须等待所有预设实验完成，不能只看均值之比。

## September Test

本轮未执行Test。先完成18个冻结训练/消融与Development对照，然后生成最终评价freeze。需用户明确授权文件绑定该freeze SHA256，再运行：

```bash
python -m experiments.joint_v3.test_once --freeze <final-evaluation-freeze.json> --authorization <用户明确授权.json> --source-root /mnt/d/A-myprofile/SustainCluster/upstream --output <september-test-once目录>
```

授权文件必须含`authorize_september_test: true`及准确`freeze_sha256`。入口在授权、完整实验、源码身份校验通过前不会打开Test来源；随后先创建跨输出目录共享的一次尝试标记，再校验来源字节并执行。失败也不自动重试。原V1/V2/E1加载器仍永久拒绝Test。
'''
    (docs/'JointRL-V3-experiment-results.md').write_text(results_text, encoding='utf8')
    atomic_json(docs/'JointRL-V3-progress.json', dict(campaign=str(campaign),
        source_identity=protocol['source_identity'], status=status['stage'], jobs=evidence,
        analysis=analysis, test_access=False))
    return dict(stage=status['stage'], jobs=len(evidence), evaluations=analysis['evaluation_files'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True); p.add_argument('--docs', required=True)
    a = p.parse_args(argv); print(json.dumps(refresh(a.campaign, a.docs)))


if __name__ == '__main__': main()
