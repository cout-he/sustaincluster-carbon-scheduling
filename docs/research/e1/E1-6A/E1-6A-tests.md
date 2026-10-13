# E1-6A 测试与回归证据

本报告依据实际日志和输出记录，日期为2026-10-13（Asia/Shanghai）。测试环境沿用本地Ubuntu-22.04与既有`sustaincluster` Conda环境，以及E1-5B隔离的runtime-deps；没有联网安装或改变旧环境。WSL日志前部的代理提示/编码NUL不作为pytest结论，以下结果取实际pytest末尾及退出状态。

## 已完成的行为验证

四个新增测试文件合计66项。政策文件41项、状态/静态缓存文件9项、分析文件8项、独立实验驱动8项；下列批次有重叠，不能把各行passed数量相加当作独立测试数。

| 范围 | 实际结果 | 本地日志 |
|---|---|---|
| 新任务级策略与公式41项 | 41 passed，2.60s | test-task-risk-green-002.log |
| 状态/静态缓存9项、策略41项、当时分析3项 | 53 passed，2.05s | cache-green-001.txt |
| 新分析/图源及冻结campaign要求 | 8 passed，48.69s | test-analysis-green-002.log |
| 八策略公平组、历史回归、回放、Train gate | 8 passed，97.57s | driver-green-001.txt |
| 新模块及旧执行/固定动作环境精选回归 | 101 passed，7.77s | old-env-regression-001.txt |
| 原科研目录全部tests/e1 | 440 passed，989.24s，exit 0 | full-tests-001.txt |

新测试检查原duration=16分钟时的最新网格启动和资源释放差异；阈值等号、safe_slack=15分钟等号；unknown/cold、120分钟左开历史边界及到达去重；CPU/GPU/MEM任务需求匹配、多资源权重及加权平均可能低估极稀缺小需求的反例；零资源；过期任务继续执行；同拍start/defer/capacity waiting后的状态；EDF全队列与稳定tie/backfill；任务和资源守恒；相同当前历史但不同未来的本拍决定与能源一致；旧registry、Discrete(3)动作与新registry隔离。

静态缓存优化的逐候选诊断与正确参考版要求完全一致，五个消融逐个验证；完整LocalExecutor结果也要求精确一致。512个合成候选的开发profiling由3.151792s降至0.065773s，仅说明实现开销，不是研究性能或创新证明。缓存只保存不变的任务占用工作量，没有缓存未来任务、CI或改变pending/running更新语义。

驱动微型测试使用真实任务执行器和KnownPower隔离物理依赖，确认八策略同一公共终点、早排空后仍使用原模型积分、模型在尾段结束后才关闭、完整任务账本、截断删失与不得作等工作量碳比较；同时验证旧档案不改、旧基线轨迹及能源队列前缀精确回归、科学回放及篡改拒绝、参数/源码变动拒绝和Train correctness gate。

## TDD及开发失败记录

RED日志保留：任务策略初始缺少模块5 failed；状态模块3 failed；驱动8 failed；分析模块3 failed及后来新增analyze_campaign接口1 failed；静态缓存新参数尚未实现时6 failed、3 passed。对应test-tdd-red-policy-001.log、state-red-001.txt、driver-red-001.txt、analysis-red-001.txt、test-analysis-red-001.log、cache-red-001.txt。各行为在新实现后的GREEN批次实际通过，RED不删除或伪装为GREEN。

test-task-risk-green-001.log和combined-green-001.txt分别记录1 failed/40 passed及1 failed/54 passed。失败的“过期等待任务释放后启动”fixture让两个任务同时到达，却预期短deadline任务在长任务之后执行；EDF实际正确地优先启动短任务。修正仅限本轮新增fixture：先到长任务，15分钟后到达短任务，后者在45分钟资源释放时启动并违约。没有改变EDF、执行器或物理模型。后续策略41项和精选101项实际通过。

## 完整suite及发布测试状态

原科研目录全部E1测试`pytest tests/e1`已结束：440 passed，989.24s，exit 0，完整日志full-tests-001.txt。此前文档把上游test_model.py误记为3项；实际源文件仅一项forward测试，本轮补跑1 passed/1.99s/exit0（original-model-tests-002.txt）。历史377项与当前完整E1的范围不能靠差值猜测，本轮精确报告原E1 440项及上游model 1项，不调用会启动shell训练的tests/test_training_loop.py。发布副本对应完整suite（tests/e1 tests/test_model.py）实际439 passed、2 skipped、277.83s、exit0；两项skip为私有E1-2B及E1-1A档案外部依赖，原科研目录已通过，不假称public自包含。精选101项仅是补充证据，没有替代完整E1 suite。

旧suite包括`tests/e1/test_rl_training.py`的9项合成fixture PPO接口回归测试。部分用例确实调用SB3 PPO进行64步微型优化更新，其他用例在覆盖/未对齐budget/加速器错误等边界拒绝前退出；这些不是9个新科研策略训练。它们只使用三个合成任务、KnownPower与pytest临时目录，并检查optimizer、保存载入、seed、在线observation、奖励截断和边界契约。不会使用真实Train/Validation/Test任务、不会修改E1-5B已训练checkpoint，也不会产生本轮PPO性能结论。本轮没有启动正式PPO研究训练。

## 冻结输入和真实实验回归

只读`existing-input-check-001.json`确认六个既有Train/Validation冻结包payload及旧实验manifest完整匹配。旧实验核验文件数分别为Train三组各65、Validation1/1.5各32、Validation2为65；三个旧基线所需ledger/events/intervals/队列字段均存在。

campaign-001的参数规范化SHA256为`ec11ecebe81e9e1091d46937e1b847a55cea8afe786937aaac214eeba10d4668`，冻结175个源码/测试文件的规范化SHA为`c3b45a58bb2493d47200f1b517539001e6e88065d6a8b976a736859b45f6ee24`，campaign配置规范化SHA为`a5c9e5a33287141d20bf21ab3f04e5489caac15d36f56c9c3c55bf8fa6f6ec5b`。真正旧文件保护终检仍应独立于这一新增源码freeze。

三档Train八策略均已完整排空，公共终点均为2023-05-03 14:00 UTC；三组baseline-regression均checked=true、matched=true，Train functional gate已通过，明确not_performance_selected=true。gate文件字节SHA为`965539168402a0d4ff144f5a2e264428aeff29f82d1915e75149edd0ff6236b9`。随后Validation三个负载组也全部完成并退出码0；六组共48个真实策略运行全部完整排空，每组旧三个基线均checked=true、matched=true。科研性能分析另见消融报告，不以完成与守恒代替性能优势。

实际Train1×八策略完整冻结回放已完成，执行退出码0；replay-001/replay-check.json记录`scientific_outputs_exact=true`、`runtime_excluded=true`。回放比较完整科学输出字节，包括八策略账本、事件、逐拍能源资源、诊断、公共窗口和公平报告；挂钟时间与provenance分离。首次未提升权限的replay-001.txt遇到WSL E_ACCESSDENIED，无研究运行；按既有授权取得WSL权限后成功运行，实际命令日志为replay-002.txt。两个日志均保留。只完成Train1×真实组回放，不声称六个大组都重跑。

原科研保护终检已完成，退出码0；preservation-final-001.json确认225228个真正旧普通文件内容/尺寸未变、unexpected_additions为空、原Git元数据与Git状态均一致。capture时漏入的一份本轮新测试显式重新分类，原始preservation-before.json保持原字节SHA`28e6b9d3a7382b59b2a688a7d48e29cb304c0a0fca7d4a5b1db9bb3a8319b7c1`。非普通文件及已声明scratch/runtime/cache排除目录仍是保护范围边界；不宣称225229条都属于旧成果。

## 审核边界

单元测试证明被覆盖机制的实现契约，不能证明物理模型真实、启发式预算提供严格SLA保证、未知未来资源竞争可行，或方法性能/新颖性成立。科研性能必须另据完整六组同工作量同公共终点结果分析；保留失败方法与高压负面结果，Test继续封存。


## 图形显示边界补充

原48行CSV正确，但图形QA发现DMR共享纵轴裁切Validation高压点。独立notes renderer在新目录修正，原科学源/输出不改；新增unittest 3项通过0.373s，包括Train低/Validation高时共享上界、零违约仍显示5%线、无效率拒绝。RED因模块尚未存在；首轮GREEN的max(empty)是微型fixture未包含全部策略产生空曲线，仅修正测试跳过空曲线，renderer不改；日志figure-fix-red-001、figure-fix-green-001/002全部保留。


发布目录独立图形边界3项也通过，源码复制和暂存核验记录见publication-verification.json与最终Git回执。

暂存后驱动/冻结回放复核：8 passed、2 skipped、15.13s、exit0，publication-staged-tests-001.txt明确两项skip原因。新增源码/配置/测试暂存blob逐字节匹配原科研版本，git diff --cached --check通过；配置CRLF与封存测试末尾空行使用仅限定新路径的attributes保留，未改源文件。
