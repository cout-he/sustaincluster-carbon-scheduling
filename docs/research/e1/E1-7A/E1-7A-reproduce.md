# E1-7A复现命令与外部依赖

代码基于发布研究分支80f5c84c2ae45bf86de397a4421d8cdedba7e03f；原科研目录HEAD/main仍为上游3f6ea95cb835b89ba50b0ef76d66d14b8037643e且dirty。运行结果以源码内容SHA为准，不能仅用原HEAD标识未提交的研究代码。原IT/HVAC、executor、旧PPO及checkpoint不改。

## 运行环境

Ubuntu-22.04 WSL、Python3.10.22（既有sustaincluster Conda），numpy2.2.6、pandas2.3.3及既有Gymnasium/SB3包；OR-Tools9.14.6206、protobuf6.31.1、immutabledict4.2.1、absl-py2.3.1单独--target安装。requirements-e1-diagnostics.txt只描述新增依赖，不能直接升级原Conda环境。原环境版本详情与每阶段实际sys.version/command在receipt中。

这些源码不含原数据或完整历史归档。真实复现需要研究者合法持有E1-5B Train precheck冻结bundle及25文件历史基线身份，包括environment/DC config、raw workload冻结任务、EDF/Carbon/Risk科学输出。expected manifest SHA已写死在配置，不能拿随意其他trace冒充同一实验。以既有数据加载/冻结工具生成新数据需要另行注册新的研究协议，本轮不会自动替换输入。

在发布副本或另一个合法本地研究checkout中，按config的相对路径挂接外部本地归档：research_notes/E1/E1-5B/precheck-001/load-{1,1.5,2}/input与evaluation。research_notes已忽略，不推送。Test仍封存。GitHub源码可以执行合成测试；真实历史回放**不是完全自包含**，缺少合法数据/历史身份时应失败而非编造。

## 实际主运行命令

在原科研目录下运行；设置PYTHONDONTWRITEBYTECODE=1、PYTHONUNBUFFERED=1、GIT_OPTIONAL_LOCKS=0、OMP/OPENBLAS/MKL_NUM_THREADS=1。PYTHONPATH包含独立solver-runtime、既有E1-5B/runtime-deps，独立notes脚本还须包含仓库root。以下python对应上述Conda可执行文件。

```bash
python -m experiments.e1.diagnostic_experiment freeze --config experiments/e1/configs/e1_7a.json --output research_notes/E1/E1-7A/selection-001
python -m experiments.e1.diagnostic_experiment baseline --manifest research_notes/E1/E1-7A/selection-001/E1-7A-selection-manifest.json --output research_notes/E1/E1-7A/baseline-replay-001
python -m experiments.e1.diagnostic_experiment offline --manifest research_notes/E1/E1-7A/selection-001/E1-7A-selection-manifest.json --output research_notes/E1/E1-7A/A-campaign-001
python -m experiments.e1.diagnostic_experiment carbon --manifest research_notes/E1/E1-7A/selection-001/E1-7A-selection-manifest.json --baseline-run research_notes/E1/E1-7A/baseline-replay-001 --output research_notes/E1/E1-7A/B-analysis-001
python -m experiments.e1.diagnostic_experiment counterfactual --manifest research_notes/E1/E1-7A/selection-001/E1-7A-selection-manifest.json --case-id load-1.5-gpu_tight --output research_notes/E1/E1-7A/C-smoke-001
python -m experiments.e1.diagnostic_experiment counterfactual --manifest research_notes/E1/E1-7A/selection-001/E1-7A-selection-manifest.json --output research_notes/E1/E1-7A/C-campaign-001
python -m experiments.e1.diagnostic_experiment counterfactual --manifest research_notes/E1/E1-7A/selection-001/E1-7A-selection-manifest.json --output research_notes/E1/E1-7A/C-replay-002
python -m experiments.e1.plot_diagnostics --a research_notes/E1/E1-7A/A-campaign-001 --b research_notes/E1/E1-7A/B-analysis-001 --c research_notes/E1/E1-7A/C-campaign-001 --manifest research_notes/E1/E1-7A/selection-001/E1-7A-selection-manifest.json --output research_notes/E1/E1-7A/figures-002
python -m pytest tests/e1 tests/test_model.py -q --basetemp=research_notes/E1/E1-7A/full-tests-tmp-002
```

这些目录已存在，不允许覆盖。复现应改为全新output目录，并在后续manifest/plot命令中使用新对应路径；frozen selection保留同SHA。普通full-suite不调用原test_training_loop.py的SAC训练入口；保留原64步合成PPO接口回归是软件测试，不是新科研PPO训练或模型选择。

附加archive-local脚本在精简发布的local-tools中保留原字节：analyze_zero_workload.py、plot_supplement.py、compare_proxy_to_pairs.py、verify_c_replay.py、verify_figure_exports.py。这些不是通用新数据生成器；执行时把它们放在本地research_notes/E1/E1-7A/的预期目录并提供既有manifest与receipt，使用新输出目录。验收工具assert缺失/不一致时必须停止。绘图QA需本地pdfplumber/Pillow；不导入在线策略。保护快照脚本/224528项全目录metadata不公开。

## 哈希及失败保留

selection payload SHA e62b7eddcde17ee80c5eb3300b36c3a2ff4880a8899c24adc0f6fbd8b9b6c361；config canonical SHA b7bf0bb8129e509475cc4d962f779b7ebab5ee9b3f99f2eaed9d8b19f77ff987。首次A/B/C执行来源map SHA e2187eec03707fd488cc315b8cf0241f1930a57e87d991b34123a150b7bf8308；修复新增驱动等价时区审计与完善绘图后，C-replay-002 source map SHA10901fd60c4fdcc3a1e56990fc21a41b7260639f3d9413406626c1d70e58cb09，完整科学数据和配对证据仍byteexact。

C-replay-001因执行期间新增绘图测试而sourceguard拒绝，未接受为复现证据；失败目录/日志保持原样。full-tests-001因错写test_model.py顶层路径未收集，随后用实际tests/test_model.py重跑；补充空任务脚本首次缺少root PYTHONPATH、图形QA首次缺fitz/旋转字符geometry口径错误均保留失败日志，最终使用实际可用pdfplumber、PDF Tf字号及PNG metadata验证。没有删失败测试、改deadline或放宽物理守恒容差。

完整raw case/ledger/events/CI/weather、solver及PPO中间数据、论文PDF/全文、巨大日志、完整图像与环境依赖均保持本地且不上传。公开只含必要源码、测试、固定配置、自写报告、小CSV与不可逆身份哈希，不重写任何旧provenance。
