# E1-6A 实际运行与复现说明

科研执行根目录为`D:/A-myprofile/SustainCluster/upstream`；独立发布副本不用于正式消融运行。以下环境沿用E1-5B/E1-5C既有本地依赖，不联网、不重新安装Conda或训练正式PPO。时间范围统一UTC；日志标注日期采用Asia/Shanghai。

## WSL环境与实际消融命令

PowerShell调用Ubuntu-22.04，Python为`/home/hwj/miniconda3/envs/sustaincluster/bin/python`，SB3等旧回归依赖使用已安装的`research_notes/E1/E1-5B/runtime-deps`。运行时设置PYTHONDONTWRITEBYTECODE、PYTHONUNBUFFERED和GIT_OPTIONAL_LOCKS。

```powershell
$py = '/home/hwj/miniconda3/envs/sustaincluster/bin/python'
$deps = '/mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5B/runtime-deps'
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 "PYTHONPATH=$deps" $py -m experiments.e1.task_risk_experiment --config experiments/e1/configs/e1_6a.json --split train --output research_notes/E1/E1-6A/campaign-001
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 "PYTHONPATH=$deps" $py -m experiments.e1.task_risk_experiment --config experiments/e1/configs/e1_6a.json --split validation --gate research_notes/E1/E1-6A/campaign-001/train-functional-gate.json --output research_notes/E1/E1-6A/campaign-001
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 "PYTHONPATH=$deps" $py -m experiments.e1.task_risk_experiment --replay research_notes/E1/E1-6A/campaign-001/train-1 --output research_notes/E1/E1-6A/replay-001
```

上面Train、Validation和replay参数来自实际组provenance。首次回放命令在未提升权限时返回E_ACCESSDENIED，replay-001.txt保留；按用户既有授权重新取得WSL命令权限后，replay-002.txt对应成功运行，replay-001目录的run_status=complete和replay-check.json明确科学字节一致。

campaign-001和replay-001均已存在，不能为了复现删除或覆盖。新完整运行必须换成新campaign名称，并让Validation引用新Train gate；新回放必须使用新的output目录。campaign允许继续同source/config/input的完整组或执行未运行组，绝不覆盖既有组；失败或身份改变会拒绝执行，应保留原证据并选择新运行ID。

驱动先核验六包冻结身份并保存input/source/config manifest，依次完成Train1/1.5/2后生成非性能筛选的train-functional-gate。Validation必须提供该gate，核验源码、参数和三组任务/资源守恒及完整Drain；不会依据Train/Validation碳/DMR优胜结果调系数。阈值161.075来自此前固定校准，本轮未重新调阈值。

## 测试复现

实际新增驱动完整GREEN命令如下；第一次RED使用driver-red-pytest-001，失败日志保留driver-red-001.txt。重跑须改新的basetemp，避免覆盖已有测试scratch。

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 "PYTHONPATH=$deps" $py -m pytest tests/e1/test_task_risk_experiment.py -q --basetemp research_notes/E1/E1-6A/driver-green-pytest-001
```

为从最终源码核验四个新增测试文件，可使用以下完整新增范围命令（这是复现范围，不把它伪称为已经单独执行的66项聚合批次）：

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 "PYTHONPATH=$deps" $py -m pytest tests/e1/test_task_risk.py tests/e1/test_task_risk_state.py tests/e1/test_task_risk_experiment.py tests/e1/test_task_risk_analysis.py -q --basetemp /tmp/e1-6a-new-REPLAY
```

53项批次是当时状态9项+策略41项+分析3项；后来分析扩展为8项。101项精选旧环境回归的实际文件列表未完整保存，不猜测；其日志作为补充证据。最终全部E1 suite与发布副本对应测试是可完整复跑的权威批次。原科研目录已执行完毕的实际完整E1命令为：

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 "PYTHONPATH=$deps" $py -m pytest tests/e1 -q -p no:cacheprovider --basetemp=/tmp/e1-6a-original-full-20261013-0149-001
```

该批次实际440 passed、989.24s、exit0；补充tests/test_model.py实际只有1项，本轮1 passed/1.99s/exit0；不得将未运行项计入总数。发布对应suite使用tests/e1 tests/test_model.py，结果以实际日志为准。精选通过不能代替全套。旧9项PPO接口测试中部分会在合成三任务fixture上做64步真实优化更新，只是接口回归scratch；不使用本轮真实任务集合，不改变旧研究权重，也不是新正式PPO训练。

## 独立输入核验及输出含义

只读核验命令已实际执行，标准库脚本不导入策略或物理模型：

```powershell
& 'D:/anaconda/python.exe' -X utf8 -B research_notes/E1/E1-6A/check_existing_inputs.py 'D:/A-myprofile/SustainCluster/upstream' 'D:/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-6A/existing-input-check-001.json'
```

该目标已存在，重跑必须换新名称。脚本核验Train/Validation bundle payload及旧baseline完整manifest；不读取Test或checkpoint。

每组comparison.csv包括fixed/own/common三种观察范围，common_summary.csv为八策略共同核算结果；runtime.csv单独保存挂钟/order/decide/hooks。每策略完整ledger、events、intervals、policy events、diagnostics JSONL均保留本地。科学哈希排除runtime和provenance，因此严格回放不会要求不同运行时间相同。

早完成策略维持原物理状态并使用真实天气/CI继续空闲积分。每组公共终点可能与E1-5C不同，必须在新终点重积分旧基线；baseline-regression保存旧完整任务事件和重叠逐拍资源能源队列前缀的精确核对。不能把历史不同终点总碳直接当作本轮变化。

发布副本仅包含必要源码/测试/配置/精简报告/小CSV与审查后的图。真实完整复现需要外部原数据及上述六个既有冻结包、旧基线档案；这些不上传GitHub。缺少依赖会明确失败，不能宣称公开发布内容完全自包含。不得复制或凭空生成测试CI、天气或未来任务数据来绕过缺失。


## 正式图1显示修正（不改仿真源码/输出）

原analysis图1共享上限裁切高压Validation；本轮将其标为不合格，不重写封存输出。使用新增独立脚本生成正式图1，其他4图可原字节复制：

```sh
python research_notes/E1/E1-6A/render_corrected_dmr.py --summary research_notes/E1/E1-6A/analysis-001/summary.csv --output research_notes/E1/E1-6A/corrected-figures-NEW --companions research_notes/E1/E1-6A/analysis-001
python -m unittest discover -s research_notes/E1/E1-6A -p test_corrected_figures.py
```

GitHub仅发布docs/research/e1/E1-6A中的同脚本及小汇总表：--summary可指向tables/summary.csv；没有完整图片的情况下省略--companions也能重绘正式DMR。真实大组严格原科学回放仍使用封存source，独立发布基线旧上游93文件仅CRLF/LF差异且Sphinx conf.py被早先清理排除；不能把它的总体字节manifest冒称原目录manifest。本轮新增9个研究source/config/test文件逐字节核对，旧E1源和物理源逻辑不变。


## 独立发布副本实际检查

`python -m pytest tests/e1 tests/test_model.py -q -p no:cacheprovider --basetemp=/tmp/e1-6a-publication-full-20261013-0515-001`：439 passed、2 skipped、277.83s、exit0。日志publication-tests-001.txt；PYTHONPATH仍使用本机已安装SB3依赖路径，不联网安装。两项历史档案测试需要外部私有输入；原科研目录完整suite已通过。发布目录图形unittest额外3项通过。提交前添加docs补充绘图脚本到Git索引会扩展source_hashes的tracked Python集合，因此暂存后另检查驱动冻结/回放8项；这个metadata差异不修改调度算法。结果以最终Git回执为准。
