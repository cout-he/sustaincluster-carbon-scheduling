# A0 验证记录

基线源码 `bf8a35ef647f19b5ef10ce1b99f723370b7a0732`，新分支 `research/object-validity-audit`。

## 软件与数据边界

- 资源192/192次、碳规则54/54次、物理1440/1440次均完成。物理扫描含90个两家族重叠条件，不称1440个独立条件。
- 原受保护145个源码/配置文件逐字节等于基线；0个修改。清单见 `data/input-provenance.json`。
- Train96场景身份清单、manifest SHA保持一致；未加载Development/Validation/Test payload。
- 本轮未训练任何模型、未实施对象候选。原任务全部保留，所有规则完成率100%。
- 资源完整场景统计、后处理与碳规则使用不同目的的适配器；零功率资源探针未当作真实能耗结果。

## 测试

```text
python -m pytest tests/object_validity tests/joint/test_power.py tests/joint_v2/test_scenarios.py -q
.......................... [100%]
26 passed in 3.33s
```

14个新合成测试、12个现有合成/模型接口测试。该结果验证软件边界、恒等式与统计口径，不验证真实设备模型。

## 图表 QA

- `a0_figures.py` 静态检查：19 PASS、1 WARN、0 FAIL。WARN为提供PNG而未导出投稿TIFF；本轮为Markdown诊断，PDF/SVG作为可编辑矢量文件。
- 两张PDF的最小实际字形均6pt；5pt阈值以下0个，无解析警告。
- 输出4260×3060 PNG（600dpi），180.34×129.54mm的PDF/SVG；字体sans-serif，PDF type42，SVG保留文字。
- 已逐图检查：坐标/单位清楚，18/22°C交叉和CT分项来源相符，理想CI机会与规则结果均有适用边界。图1初稿图例接近MEM曲线，最终移至左上、以颜色标识资源、实线/虚线分别标均值/P99。
- 图中没有模拟填补、置信区间或显著性标记；所有点能追溯至聚合CSV和 `figures/figure-provenance.json`。
- 软件图形检查不代表论文投稿格式全面审定，更不代表物理模型已校准。

## 交付完整性与限制

只新增 `experiments/object_validity/`、`tests/object_validity/`、`docs/research/object-validity/`。所有任务明细、逐步资源曲线、冻结输入、大日志均留在被Git忽略的 `research_notes/object-validity/A0-001/`。合成物理扫描完整CSV不到0.8MB；公开整个报告/数据/图片约3.4MB。

本目录新增局部 `.gitattributes`，覆盖仓库继承的 `docs/research/** text eol=lf`：仅对A0的data/figures保留精确字节，防止PDF/PNG被文本换行归一化破坏，并保持CSV身份。没有修改原有属性文件或生产代码。

原运行脚本身份保留在各自metadata；后处理扩展后的最终代码身份由artifact manifest记录，不伪造历史相同源码。候选配置没有实测、没有新场景实验；数据单位/设备规格/真实Deadline等外部依据缺失均在主报告列出。

首次执行后因额度中断，用户恢复后只进行整理与合成验证。跨中断日历时间超过原30分钟wall预算；不声称符合整轮连续wall≤30分钟，也没有借恢复扩大实验范围。原实验完成时间和资源主运行耗时保留在metadata。
