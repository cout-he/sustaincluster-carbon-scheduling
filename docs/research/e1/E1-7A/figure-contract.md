# E1-7A 图形契约（实际A/B/C结果绘制前）

采用既有 Python/Matplotlib 工作流，独立新增 `plot_diagnostics.py`。只读取通过Train角色、complete状态、receipt和文件哈希检验的A/B/C输出；不加载原workload、Test或模型。输出目录必须全新；所有图保留原数据与绘图CSV、输入/输出SHA绑定，PNG300dpi、editable PDF/SVG。DejaVu Sans，文本最低7pt，双列宽约183mm；绝对指标轴从0开始，signed Δ轴显示0。没有显著性检验、p-value或将12个事件称独立真实负载样本。

## 主结论与证据逻辑

图只显示实际观察，不预先假设预约、低碳延迟或风险分数有效。A说明某个有限条件问题的资源时间安排；B同时显示完整IT/HVAC电能和碳排放及实际CI时间结构；C显示同状态一次动作干预后的系统差异和风险信号的反例/支持。属于 quantitative grid / mechanism timeline，图注写明所有条件、统计范围及局限。

### A资源时间安排图

仅从有已验证incumbent、可完整物理重放的真实A2案例选择展示，不混入12任务合成例。规则：最大 `edf_misses-oracle_misses`；平局稳定case_id；若均无正差，firstA2 stablecase_id。它只是display选择，不改变预注册案例/求解样本。绘制条件EDF与CP-SAT两分支真实CPU/GPU总占用（包含固定背景），和每一个所选候选的start、真实finish、ceilrelease、deadline。完整任务ID和所有绘图时点保留底层CSV。显示窗口以anchor前30min至最后候选释放后30min，限制在已观察覆盖内，不偷偷删除选中任务。

### B完整能碳与CI图

全部3压力×3基线的IT/HVAC堆叠电能及碳排放；保持9行完整数据和策略配对身份，不以代理代替实测。CI使用固定48h到达窗，核对三压力CI轨迹一致，标预定161.075阈值。辅助panel显示高/低CI连续runlength分布；不是预测曲线。无误差棒：每档一个冻结负载/确定性运行，不编造独立重复或CI。

### C配对时间线与风险散点

代表图规则：若有额外违约，最大positive Δmiss（平局case_id）；另firstnonincrease stablecase_id。若减碳且增违约的tradeoff存在且未被前两图包含，再firsttradeoff stablecase_id。不存在某类就明确记absent，不强迫构造。这个展示选择只依既有全部结果，不修改冻结12target，也不隐去零变化。

各代表图读取同case两分支完整intervals及目标账本：当前CI、CPU/GPU、pending、累计Carbon差（defer−execute）及目标task的start/finish/ceilrelease/deadline。资源与队列panel重点显示干预前60min至后至少360min或目标释放后60min，限制实际共同覆盖；完整55h等数据仍保留CSV和公共最终Δ。原始配对前缀/共同终点必须已审计。

风险图使用所有12case（或全部冻结有效case），两panel：连续v1 z 对 Δmiss、z 对 Δcarbon。按负载区分颜色，保留case index到ID映射，signed结果轴显示零。小样本只作描述，不拟合线/模型，不给p-values，不把differenttrajectory当单特征因果效应。

## 数据与API契约

`plot_campaigns(a_dir,b_dir,c_dir,output_dir,selection_manifest=None)`；reader只读科学输出，不接触求解器或执行器。辅助 `choose_a_display(rows)` / `choose_c_displays(rows)` 的稳定规则先用显式toy fixture测试。实际figure生成须root确认所有campaign完成并提供路径；未确认前只测试toy图（注明fixture，不作为科研结果）。

`figures-manifest.json`包含科学source CSV和receipt哈希、展示规则/被选case、每个图的source-data CSV和PNG/PDF/SVG哈希、轴范围/文本设置、图注及明确的缺类信息。完整任务账本和图片保持本地，不同步GitHub；公开只上传源码/测试和必要精简报告。

交付前执行nature-figure source preflight、PDF文本字号检查，并逐panel视觉检查；这些自动结果不替代人工可读性核对。保存任何QA告警，不因自动检查误报隐藏原始输出。
