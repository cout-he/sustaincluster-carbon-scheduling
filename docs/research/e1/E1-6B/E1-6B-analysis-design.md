# E1-6B 结果分析与图形契约（正式训练前）

目标是检验四动作、连续任务状态、冻结风险过滤各自的实际效果，不预写收益或将启发式称为已证明安全。分析输入只有保存的三档开发Validation结果和12个正式Train模型的摘要；不访问Test，不拟合或修改任何风险/Reward系数。

## 输入和强制完整性

- `analyze_campaign(group_dirs, training_dirs, output_dir)`要求3档1/1.5/2、各19个方法：4经典启发式、4组×3注册seed、3旧三动作PPO。
- 正式训练要求4组×seed55101/55102/55103，20480步、83维/4动作、19397参数；每模型参数确实改变、优化器状态存在、保存回读一致。同seed初始参数相同，共享网络/Reward/PPO配置相同，源码锁一致。
- 相同压力档位必须同冻结输入、任务数、完整Drain、同common_end、资源小时一致、空闲尾段计入、全部完成且无删失。缺失/重复seed、其他split、模式缺失、条件矛盾一律拒绝，而非silent drop。
- 新模型结果必须对应training_summary记录的checkpoint/steps；旧三动作PPO是外部基线，不纳入严格单因素配对。
- 读取并核验所用小CSV/JSON的已有hash记录，保存输入/输出/分析源码哈希，不覆盖既有目录或任何原数据。

## 输出表

- `all-methods.csv`：57行，每档19方法，完整任务状态、DMR/违约、IT/HVAC/总能碳、JCT/等待/队列、容量与主动等待、utilization、资源小时、drain、动作和过滤分母、运行/推理成本。
- `paired-effects.csv`：27行，3档×3seed×State/Shield/Combined相对Plain。记录违约数差、DMR百分点差、碳kg差、能耗/JCT/等待/队列差。匹配训练seed不等于匹配事件级因果或独立workload replicate。
- `group-summary.csv`：12行，各档各组3seed均值/最小/最大，保留各指标分档，不合并压力。
- `training-summary.csv`：12行，实际预算、参数数、参数hash更新、优化器/回读、开销、场景步数/episode比例、动作计数及partial episode披露。
- `training-updates.csv`：逐保存PPO更新数值，保留12模型的学习诊断；不等同科学评测指标。
- `training-episodes-summary.csv`：24行，每模型每压力场景完成episode的return/代价与长度范围；budget结束时未完成episode单独在training_summary披露。

## 图形契约

绘图后端沿用已确定Python/matplotlib，quantitative grid，7.2英寸宽、PNG300dpi、PDF矢量可编辑字体、SVG可编辑文字、最小字形6pt。不使用其他后端修图。

1. 绝对DMR：三档facet，共用0起点纵轴；四RL组各seed点、均值和min-max；4启发式参考线、旧三动作PPO均值/范围和5%目标。全局ylim覆盖全部19方法点，不裁掉不利结果。
2. 绝对公共窗口Carbon：同样三facet与全体点范围，纵轴0起点；只有通过完整公平门才能生成。
3. 配对差值：DMR百分点和Carbon kg两行，各3档facet；每seed差、均值和min-max，零参考线，signed纵轴允许负值。每行全局lim覆盖全部配对点，不用绝对图的0起点裁掉负变化。

图注声明3个训练seed用同一冻结workload，不是3独立真实到达样本；min-max不是置信区间，不做单次确定性运行的显著性宣称。图形对应原始CSV完整保存；科学归档/账本/模型与完整图片仅本地保留，发布只选小摘要。

正式结果出现之前，本契约不声称任一组优于Plain/RiskOnly/EDF。独立QA将核对每panel、字体、范围、source data和hash。
