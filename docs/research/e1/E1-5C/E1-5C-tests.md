# E1-5C 实际测试与证据

| 检查 | 实际结果 | 本地记录 |
|---|---|---|
| 新增压力、SLA下界、诊断驱动行为测试 | 53 passed / 4.18s，exit0 | new-unit-002.txt |
| 原科研目录全部tests | 377 passed / 1280.20s，exit0 | full-tests-001.txt |
| 独立发布副本全部E1+上游model测试 | 373 passed，2 skipped / 732.02s，exit0 | publication-tests-001.txt |
| 独立资源工作量/见证复核 | 六档两模式；30个不同资源见证；ceil精度差异0 | witness-verification-001.json/002.json，两次字节相同 |
| 只读结果汇总核验 | 127个产物哈希；18基线/12下界/18配对集合计数正确 | result-evidence.json、final-evidence-004.txt |
| 汇总与压力证据回放 | 全部小CSV/表格/见证核验字节相同 | summary-replay-001、pressure-replay-002 |
| 历史文件保护 | 282291个旧regular文件无变化；Git身份/元数据一致 | preservation-final-001.json |

发布副本两个skip是既有私有历史归档依赖：test_cars_experiment.py和test_policy_runs.py中对应真实历史回归。新增53项没有skip。原科研完整377项包含另外两项上游测试；发布范围为tests/e1和tests/test_model.py，不能把计数差当作失败。外部历史归档在原科研目录存在，故原目录这两项实际通过。

发布实际命令（PowerShell，依赖仍来自原科研目录，不复制环境）：

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001 -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=/mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5B/runtime-deps /home/hwj/miniconda3/envs/sustaincluster/bin/python -m pytest tests/e1 tests/test_model.py -q -p no:cacheprovider --basetemp /mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5C/publication-pytest-001
```

## TDD与覆盖

先观察missing module失败，再新增独立模块。压力20项、下界25项、驱动8项，合计53。压力覆盖原任务/倍率身份、整拍占用与raw duration分离、空到达拍、固定历史分母、负slack、紧迫与资源统计、EDF时间对齐、Drain无新增到达、守恒和记录不一致拒绝。下界覆盖最大资源/区间取max、intrinsic去重、区间端点、fractional finish/资源释放、等号、必要非充分反例、occupied强化和30个固定seed四任务非抢占穷举，证明报告下界方向不超过精确最优。

驱动测试覆盖拒绝Test/旧配置漂移/历史归档漂移/越界路径/输出覆盖，以及5%等号和near-target不冒称支持，且真实微型公共执行器归档可被只读核验。第一次合并运行1 failed/52 passed：新测试错误地选用了本来会截断的fixture却断言全部完成；只修正新增fixture，未改生产执行逻辑，随后53 passed。

旧关键回归包含LocalExecutor逐拍等价、固定Action0/1/2三规则、任务/资源守恒、非抢占与释放、当前历史/不同未来分叉、Reward首次计罚/真实截断、严格冻结回放、负载复制以及SB3微型更新/保存加载测试。微型已有PPO测试只验证代码行为，不是新增正式科研训练，不产生新的科研模型选择。E1-5B完整实验/已训练checkpoint通过全文件保护核验。

## 测试并不能证明模型正确

沿用E1-5B的弱覆盖说明：上游test_training_loop.py调用不存在的train_SAC_single_action.py且不检查os.system返回值，因此passed不能证明训练机制。本轮原文件不改，不将这项作为科研训练或新功能成功证据。其他测试也只是所断言契约的验证，不证明物理模型在真实数据中心准确，或5%一定可行。

855个Linux重解析/非regular路径单独记录并保持同集合，未冒称这些路径已做逐字节哈希。完整保护清单与大日志仅本地；公开结果保留精简通过数量、清单哈希、命令和限制。实际复现命令见E1-5C-reproduce.md。
