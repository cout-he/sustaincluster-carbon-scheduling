# E1-6A 实际实验图表与源数据 QA

本记录针对 `analysis-001` 的真实六组实验输出。先前 `analysis-micro-qa-001` 是合成文件接口测试，不能作为本报告的性能证据。正式源码、测试、配置和原始分析目录均未由本次 QA 修改。

## 1. 数据完整性与公式

实际独立核验脚本 `qa_actual_figures.py` 已运行，receipt 为 `figure-actual-qa-001.json`，命令输出为 `figure-actual-qa-001.log`。

- 48行主统计：Train/开发 Validation × 1/1.5/2，每组8策略。
- 各组任务量依次为5542、8313、11084、8708、13062、17416，均完成且无 pending/running/right-censored。
- 同组 workload/environment 哈希相同、公共观察终点相同、CPU/GPU/MEM完成资源小时一致。
- Train三组共同终点为2023-05-03 14:00 UTC，Validation三组为2023-07-10 12:45 UTC。核算包括完整Drain与真实空闲尾段。
- DMR分母逐行等于完成违约数/全部同组任务数；总电能逐行等于IT+HVAC。
- 减碳百分比逐行按 **`100 × (C_EDF − C_policy) / C_EDF`** 重新计算并通过，C来自相同组公共观察窗口。不同倍率/窗口不构成配对碳对照。
- 30个新策略预算群体、660行首eligible/全部eligible直方图的计数与分母一致；未计算预算不编码为0。
- 三种图源CSV分别与主统计、预算直方图、资源主导分类统计逐字节一致。
- 原 `analysis-hashes.json` 的全部条目再次核验通过，manifest SHA256为 `ff7e0acdcfc0a8c7c69c0221c65127abb95c352dae0de3079c97c724fc551717`。

正式图注采用 `E1-6A-figure-captions.md`。原自动图注第2项存在变量文字歧义，但上述公式、数值和轴标签正确；原文件保留，不重写历史。

## 2. 原始图形核验

五张PNG已逐图打开，并分别检查每一面板、标题、轴、图例与整体布局。PDF真实字体操作符扫描均最低6pt，无小于5pt字形。SVG具有真实可编辑text节点，五图text节点数依次为29、27、52、51、50；并非所有文字转成轮廓。PNG/PDF/SVG哈希保存在实际receipt中。

| 原图 | 数据/单位 | 逐面板视觉结果 |
|---|---|---|
| Deadline miss rate | 48行数据和DMR正确，5%目标不变 | **FAIL：共享y轴错误固定在约0–8%，Validation 1.5×/2×高违约点被裁切，不能作为完整对照交付。** |
| Carbon relative EDF | matched EDF派生百分比正确 | PASS：最后统一以全部48行百分比确定对称范围，约±1.6%；0至约1.39%全部点可见，零线清晰，无裁切。 |
| High-pressure Pareto | 三个预定高压组，每组8策略 | PASS：独立横轴从0，纵轴含0并覆盖该组全部点；重合点没有人为抖动，图例完整，无布局碰撞。 |
| First-eligible buffer heatmap | 30群体×0–10拍，实际n逐行标明 | PASS：全部行/拍数保留，色标0–100%；未eligible者未当作零预算；标签可读且未裁切。 |
| Buffer by demand category | CPU/GPU/MEM容量需求比例主导、tie/zero | PASS：5策略面板0–150min覆盖全部均值；短标题、图例和旋转标签可读，无类别时保持missing。 |

原DMR缺陷产生于绘图接口：`sharey=True` 时，在Train面板只设置 `ylim(bottom=0)`，同时锁定了共享上限；随后Validation高值不能触发自动上限扩展。微型分析fixture六组DMR相同，未暴露真实Train/Validation压力范围差异。它不是任务数据、调度算法或指标计算变化，但会错误隐藏高压力结果，因此作为阻塞交付缺陷报告。

## 3. 修正方案与边界

原始 `analysis-001`、其源码快照、哈希和自动图注继续封存。根代理采用独立notes绘图脚本与全新 `corrected-figures-001` 输出目录，仅从原48行CSV显式确定公共DMR上限，以补充未裁切的正式交付图；不重运行或重算科研策略，不改变参数、数据、指标或原归档。

修正已形成并完成独立复核，正式交付使用 `corrected-figures-001/`。修正DMR的PNG已打开，Train/Validation两面板逐一检查：共享y轴从0至43.2142857143%，完整覆盖本轮最高39.2857142857%；原被裁切的Validation 1.5×和2×全部策略点现均可见。5%目标线、八方法图例和完整48个点保留，无标签或图例碰撞，未通过截断轴放大差异。

独立脚本 `qa_corrected_figures.py` 的receipt为 `figure-corrected-qa-001.json`，命令日志为 `figure-corrected-qa-001.log`：

- 原 `analysis-001` 的全部manifest条目再次通过，原始哈希未改。
- 修正目录的源summary与原48行summary逐字节一致，预算和主导类别源CSV逐字节一致。
- 其余四图PNG/PDF/SVG均逐字节复制，14项伴随图/CSV哈希完全相同；因此此前四图视觉审核仍适用。
- 修正目录的五张PDF仍均最低6pt，无小于5pt字体；全部SVG仍有可编辑text节点。
- 只有DMR呈现重绘，没有重新执行调度、改变数据、参数或统计。修正renderer另附输出provenance与自身源码哈希。

修正DMR输出 SHA256：PNG `3e46e5307b19078b0ee4d6c5941a8410ca082b0cd62f559ba2a1bbf5ff7ff451`；PDF `8bad236e05148ddafa7b6a08e43f4f1c41d0ba24a78aad9a587f313b676bfe4f`；SVG `fa6e5a784c7880764b6180265023a15117ca7bc2ec2fbac56c7d2c6693f770f5`。

**正式修正图：PASS。封存原DMR图：仍为FAIL，不可引用作完整对照。** 独立renderer与小源CSV应随精简材料发布，使读者能够重画正确图；不隐瞒或覆盖冻结分析源码原有的呈现缺陷。

## 4. 测试证据与统计限制

已有测试日志只读核对：分析8项通过；旧环境/固定动作关键回归101项通过（7.77s）；完整E1测试440项通过（989.24s）。本次实际图QA不重复运行这些测试，也不把测试通过等同模型科学正确。

每个cohort只有一次确定性冻结轨迹，没有独立重复样本，不添加误差棒或显著性结论。Validation已经用于开发；Test未评估。图中对照为同组完整工作量与公共观察窗口的实测描述，既非严格因果归因，也非未知未来deadline保证。正式图表不包含微型fixture数值。

静态绘图源preflight为17 PASS、3 WARN、0 FAIL。三项warning分别是未输出TIFF、PNG采用已固定300dpi而非skill默认600dpi、派生百分比轴范围用dropna检查有限值。PDF/SVG保留矢量正文，源CSV不删行；缺失百分比的处理不虚构等工作量优势。实际六组已全部排空，因此没有因该处理删除任何配对百分比。
