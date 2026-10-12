# E1-5C 复现与实际命令

全部科研运行位于原科研工作区，不在发布副本进行新科研实验。旧源文件、物理公式、checkpoint、旧报告及provenance均不改。本轮没有PPO训练命令或Test评测命令。

## 实际WSL环境

Ubuntu-22.04；Python `/home/hwj/miniconda3/envs/sustaincluster/bin/python`。沿用E1-5B的独立 `research_notes/E1/E1-5B/runtime-deps`（SB3 2.9.0），没有更新Conda环境或重新联网安装。设置 `PYTHONDONTWRITEBYTECODE=1`、`PYTHONUNBUFFERED=1` 和 `GIT_OPTIONAL_LOCKS=0`。PowerShell执行如下命令；日志的WSL代理提示不参与pytest结果判定。

```powershell
$py = '/home/hwj/miniconda3/envs/sustaincluster/bin/python'
$deps = '/mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5B/runtime-deps'
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 "PYTHONPATH=$deps" $py -m experiments.e1.sla_diagnostics --config experiments/e1/configs/e1_5c_diagnostics.json --output research_notes/E1/E1-5C/diagnostics-001
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 "PYTHONPATH=$deps" $py -m pytest tests/e1/test_workload_pressure.py tests/e1/test_sla_feasibility.py tests/e1/test_sla_diagnostics.py -q --basetemp research_notes/E1/E1-5C/new-unit-002
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 "PYTHONPATH=$deps" $py -m pytest tests -q --basetemp research_notes/E1/E1-5C/full-pytest-001
```

`diagnostics-001` 已存在，命令会拒绝覆盖；重新运行必须选择全新输出目录并同步报告脚本的RUN位置。不能为了复现清理旧目录。驱动先核验旧冻结包、旧完整结果和原配置SHA，再冻结两个新Validation输入并运行三既有基线。它不会重训练或替换E1-5B模型选择结果。缺少原始数据/历史冻结包将明确失败，不凭空补充数据。

## 只读验证和汇总

独立见证复核 `verify_witnesses.py` 不导入下界实现，用原任务重新构造二维区间矩阵和k/k−1去除工作量，验证六档两模式及30个不同资源见证。`analyze_pressure_evidence.py` 只读EDF账本/逐拍结果，统一选择三类峰值，并重算等待资源mask。两脚本只读Train/Validation路径，输出仅在E1-5C。

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 $py research_notes/E1/E1-5C/verify_witnesses.py --root /mnt/d/A-myprofile/SustainCluster/upstream --diagnostics research_notes/E1/E1-5C/diagnostics-001 --output research_notes/E1/E1-5C/witness-verification-NEW.json
& 'D:/anaconda/python.exe' -X utf8 -B research_notes/E1/E1-5C/analyze_pressure_evidence.py --root 'D:/A-myprofile/SustainCluster/upstream' --diagnostics 'D:/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5C/diagnostics-001' --output 'D:/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5C/pressure-replay-NEW'
& 'D:/anaconda/python.exe' -X utf8 -B research_notes/E1/E1-5C/build_report.py --root 'D:/A-myprofile/SustainCluster/upstream' --diagnostics research_notes/E1/E1-5C/diagnostics-001 --summary research_notes/E1/E1-5C/summary-NEW --report research_notes/E1/E1-5C/report-NEW.md --evidence research_notes/E1/E1-5C/evidence-NEW.json
```

汇总脚本核验sealed输出哈希，产生18行基线、6行压力、18行三资源、6行slack、12行下界和配对违约集合统计；已有目标文件时拒绝覆盖。上例NEW必须换成未存在名称。最终报告编辑只澄清科学表述，CSV不改。实际首两次build_report因CSV字段名和Windows/WSL路径适配失败，日志保留为build-report-001/002；修正仅限新notes脚本，build-report-003成功，无科研重跑。参数化后的summary-replay-001与原CSV/表格字节相同，独立见证002与001字节相同。Windows压力回放第一次遇到该Python运行时相对Path.resolve异常，改用绝对root/diagnostics后pressure-replay-002成功且全部证据字节相同；没有删除失败记录。final-evidence-001/002/003是新汇总核验脚本的字段/哈希口径/nullable修正记录，004实际通过，未修改科研输出。

公开文档目录是docs/research/e1/E1-5C；脚本默认路径针对原科研布局。在发布副本调用这些只读脚本时须用上述显式参数指向外部原科研根目录/完整诊断档案/全新输出目录，不能误以为公开小CSV含完整冻结任务。发布审查发现新增组的相对group_path还需要root锚定；已仅在notes/build_report添加--root，从发布cwd指向原科研档案运行summary-replay-002，11个汇总文件及报告均与原结果字节相同。生产source/test及sealed科学输出未改。

## 发布副本验证

发布副本仅接收必要脚本/配置/测试/精简报告/小CSV，并逐文件比较科研源字节哈希。测试依赖使用原科研目录既有runtime-deps，绝不复制Conda环境。发布pytest实际命令与结果见E1-5C-tests.md。完整诊断重跑仍需要外部原数据与历史冻结档案；微型行为测试可在依赖具备后独立运行。

运行日志、原始输出manifest、完整preservation清单保留本地。GitHub只发布精简身份与结果，不能把发布提交SHA重写进历史实验provenance。
