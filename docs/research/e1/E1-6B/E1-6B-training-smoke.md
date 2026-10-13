# E1-6B：1024 步训练 smoke 与功能/成本门槛

四组 smoke 已实际执行并退出 0，仅使用冻结 Train 1.5×/2×，seed 55100，每组精确 1,024 步。本报告是正式训练之前的功能和成本验收，不是模型性能比较。Smoke 未读取开发 Validation 或 Test；没有按这些 reward/outcome 修改风险公式、Reward、PPO 超参数或预算。后续正式版本加强 checkpoint/replay/分析审计时，必须另外冻结源码；原 smoke 源码快照、模型和记录保留原身份。

## 实际结果

| 组 | 环境步 | 墙钟秒 | 独立进程峰值 GiB | Action 0/1/2/3 | 完整 episode | 尾段步 |
|---|---:|---:|---:|---|---:|---:|
| Plain | 1,024 | 30.662 | 0.91671 | 291/268/246/219 | 4 | 150 |
| State | 1,024 | 41.464 | 0.91540 | 270/262/259/233 | 4 | 150 |
| Shield | 1,024 | 28.800 | 0.91693 | 293/262/244/225 | 4 | 150 |
| Combined | 1,024 | 40.643 | 0.91684 | 265/257/262/240 | 4 | 150 |

各组都满足预注册门槛：墙钟 ≤720 秒、独立进程峰值 ≤4 GiB、真实参数更新、有限输出和 checkpoint 保存/加载精确往返。四个进程独立运行，最多两个 smoke 同时运行；其他本地测试可能并行，所以这些墙钟数用于运行成本可行性判断，不作为严格隔离的速度基准。

四组网络都恰有 **19,397** 个参数，状态维数83、动作数4，初始参数 SHA 完全相同，最终参数均发生变化，checkpoint 均保存12个有状态优化器条目。每组4个 rollout、40次 SB3 epoch 更新；256个探针观测在保存/加载后动作与参数 SHA 一致。

同 seed 的初始参数 SHA：`96438b3c73e9e422ed90e4d0bd644fa06c71f5a56e579d3c8267349e3292e77a`。

各组实际采样 Train 1.5× 为590拍、2×为434拍；前四个完整 episode 按1.5×/2×交替，最后150拍属于未结束的1.5× episode。50%/50%是场景 episode 选择规则，不是环境步比例；此处四组恰好长度一致不保证正式训练一直相同。尾段使用既有 critic bootstrap 语义，未伪造 `truncated`，不能将其作为完整等工作量结果。四组所有真实环境 episode 均无安全截止截断，terminal cost 为零。

## 记录与独立只读核验

`verify_smoke.py` 实际核对每组13个根层训练产物哈希和185个冻结源码文件哈希，逐行检查全部4,096拍：观测/奖励/诊断数值有限；83维长度；动作范围；场景身份与交替规则；Plain/Shield 的43维风险块全零；首次违约事件没有重复；真实启动/容量等待/主动延迟集合不混淆；policy diagnostics 与 risk_counts 对齐；奖励来自实际分项；三个核心公平哈希和参数量一致。

产物：`smoke-gate-001.json`，实际执行结果 `passed=true`。该门槛只判定功能和成本，不以任何 reward/违约/减碳成绩筛选组或调参。

| 组 | can-fit候选查询 | solo档位允许defer | risk veto | 实际主动defer |
|---|---:|---:|---:|---:|
| Plain | 55,335 | 10,071 | 0 | 10,071 |
| State | 55,917 | 10,653 | 0 | 10,653 |
| Shield | 51,796 | 8,790 | 2,258 | 6,532 |
| Combined | 52,496 | 9,908 | 2,676 | 7,232 |

Shield/Combined 在真实满足 solo 档位且能放下的机会中分别否决约25.69%/27.01%；仍保留大量 defer。这证明过滤没有在 smoke 中完全变成 EDF，不能证明正式评测的 SLA 增益。不同组已经产生不同队列轨迹和采样动作，以上不同计数不是单个事件的因果对照。

## 版本与身份

- 四组早期源码集合 SHA：`a0ad8613179cb6fb4a0cd06db8080587b1888c83e23f873404a46d0835305c5a`。
- 四组原始预注册配置文件 SHA：`8ef7a5b0e65efee251b684e6ce011be9ebcafdc94b7ac9a328b64f03ba1f1d96`。
- 四组输入/预算 SHA：`00ea3729a6677b2d3a27eda7226fadbff70f22b6d81cbada41b70d1bc5cef47a`。
- Plain checkpoint SHA：`6b982ac0040c6483a8f173d2cd2710b88961132da7ad5b14dceeb1dc8676cf11`。
- State checkpoint SHA：`b4c0328a4431fead8db5c9ea6aa98522e33bd1ca59f94d99187383cb8ef23b5d`。
- Shield checkpoint SHA：`0375a53d1882dbdf597ed151419c891e85e64932a92a86fd09a308d924390b41`。
- Combined checkpoint SHA：`914a9503e8f8c112123e76264d105ddd058911183fb6c892c3f3ec1df3ebacb3`。

模型、逐拍 gzip、完整 episode 记录及源码快照仅存本地，不上传 GitHub。全部训练文件清单和训练后参数 SHA 位于各 `smoke-*/training_summary.json`。

## 实际命令

在科研目录、WSL Ubuntu-22.04 中，四次调用的 group 分别为 plain/state/shield/combined，每次输出目录与日志都不同：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 \
  PYTHONPATH=<WSL_RESEARCH_ROOT>/research_notes/E1/E1-5B/runtime-deps \
  <CONDA_ENV_PREFIX>/bin/python -m experiments.e1.adaptive_train \
  --config experiments/e1/configs/e1_6b_ppo.json --group plain --seed 55100 \
  --steps 1024 --output research_notes/E1/E1-6B/smoke-plain-001
```

其余三组完全相同替换 group/输出目录。实际日志分别为 `smoke-plain-001.txt`、`smoke-state-001.txt`、`smoke-shield-001.txt`、`smoke-combined-001.txt`；每个外层命令实际 exit0，WSL 代理提示/NUL前缀不改变 Python 的真实退出状态。

只读 gate 审计实际命令（Windows PowerShell）：

```powershell
& '<LOCAL_PYTHON>' -X utf8 -B `
  '<LOCAL_RESEARCH_ROOT>\research_notes\E1\E1-6B\verify_smoke.py'
```

输出 `groups=4, passed=true`，exit0。复跑训练或审计须换全新输出编号，原函数与脚本均拒绝覆盖已有成果。

## TDD 与旧训练兼容

新训练测试先得到 **11 failed**（独立模块尚未实现），补元数据/manifest两项时得到 **11 passed、2 failed**（实际缺少训练身份、未检查声明的manifest），修实现后新13项与旧9项共同 **22 passed / 17.98秒 / exit0**。日志为 `adaptive-train-red-001.txt`、`adaptive-train-red-002.txt`、`adaptive-train-green-001.txt`。

微型测试保留真实 LocalExecutor 与 Reward，仅隔离物理依赖；实际64步 PPO 测试验证参数更新、优化器、同 seed 确定性、四组相同初始参数/参数量/预算、真实 pre-action trace、scene reset的新物理对象与跨拍持久对象、角色拒绝、首次违约去重及安全截止负债。旧40维三动作代码未修改。

本报告只授权成本判断的证据，不自动启动正式训练；正式20,480步/seed前仍须由根执行器完成新增源码/测试统一冻结。
