# E1-6B 图形QA

五张最终图均已保存PNG/PDF/SVG并逐panel目视核查。对应原始科学表为all-methods.csv、paired-effects.csv、training-updates.csv及training-completed-episodes.csv，曲线不平滑，不删除seed。

绝对DMR和Carbon从0起；配对差值为签名坐标并包含0，三seed全部显示，mean/min–max不解释为CI或显著性。图注标注开发Validation、同一workload及训练随机性。碳差很小，绝对图如实显示，另有配对图辅助读数。

初版supplement-001的右端20480刻度存在裁切，完整初版保留。仅将notes绘图right margin .99改为.94，另存supplement-002；全部原始CSV SHA与001精确相同，未改坐标轴范围/数据/统计。第二轮PNG目视检查确认刻度、单位、图例和标题可见。

五PDF实际text audit均PASS，最小6.5pt，阈值5pt，无低于阈值glyph run。实际嵌入DejaVu Sans且CO₂e显示正确。源preflight的FONT-FAMILY FAIL源于验证器只认可部分sans字体名的误报，原报告保留，以实际字体和渲染检查解释，未谎称自动preflight全PASS。300dpi PNG/no TIFF的WARN保留；本轮注册的是PNG/PDF/SVG，优先论文使用矢量PDF。

本地最终图：analysis-001/figures/deadline-miss-rate、carbon-emissions、paired-effects；supplement-002/figures/training-optimization-diagnostics、training-episode-returns。图像归档不上GitHub；必要重画源码和精简科学CSV发布。figure-QA-001.json记录实际PDF文字审计、图哈希、数据不变及人工目视范围。
