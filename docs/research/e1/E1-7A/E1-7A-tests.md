# E1-7A测试与完整验证

实际命令为python -m pytest tests/e1 tests/test_model.py -q，包含全部E1关键测试及原模型测试；原test_training_loop.py会发起不在本轮范围的SAC训练，明确未调用，不能把本结果称为裸pytest全仓库通过。

|范围|通过|跳过|失败|实耗s|
|---|---:|---:|---:|---:|
|原科研目录|677|0|0|5138.49|
|独立发布副本|675|2|0|1096.18|

新增145项（A46、C23、B28、selection9、driver23、fair audit7、plot9）全部通过；原532项在原目录通过。发布副本两项跳过来自test_cars_experiment.py的私有历史归档和test_policy_runs.py的旧本地artifact，其余旧530项通过。合成测试不依赖原始大数据；保留旧PPO入口的微型合成参数更新回归，没有开展新的科研训练、选择模型或写入旧checkpoint。

测试包括三资源守恒、rawfinish/deadline与纳秒ceil release、整数/浮点映射边界、solver超时/状态、固定背景承诺、同拍回填、future-branch隔离、同状态单次动作、全任务完成和公共空闲积分、角色先验证、配置全哈希、覆写拒绝、绘图数据身份与积分时点。

实际实验验收：9历史基线×8文件byteexact；23 CP模型均OPTIMAL（仅映射有限问题），46原资源时间表及44完整物理分支验证，187662固定背景启动/释放核验；12对真实C全完成/同前缀/唯一干预；修复等价时区后192分支科学文件+72配对证据byteexact，60历史EDF控制验证实际执行。时间/性能日志不作为确定性科学值。

8图均PNG300dpi/PDF/SVG和对应CSV，实际PDF全部Tf字体≥7pt、DejaVu Sans及SVG可编辑文本核验通过，并人工查看8图。source regex preflight有3项静态识别失败（未识别rc_context中的STYLE和循环扩展名导出）及DPI识别告警，完整保留；实际导出文件提供反证，不将静态检查假称全部PASS。旋转字符pdfplumber bounding size不是Tf字号，使用独立Tf审计覆盖全部旋转/普通文字，不放宽7pt标准。

原目录224528旧文件metadata保持相同，158个旧当前源码/config/tests/M0内容SHA相同，.git元数据/HEAD/main/research及原dirty变化保留。只新增声明允许的文件与E1-7A输出。不是将所有旧原始数据都重新读入或SHA；不读封存Test payload，不获取Validation结果用于本轮选样/调参。原Train环境安全覆盖至June2，实际计算到May3 14:00；Validation/Test声明在July/September，不存在实际窗口重叠。

完整日志SHA：original `2f3e76d1e409a0170a2db92f82dcd9c97d4ae6611876edc8a48d0b7e0cbf7366`；publication `daa958c28d68edb10b9bcefef022a99731493fa925d266c2684664782efff69a`。RED/GREEN和失败调试记录保持本地，包括时区字符串审计漏检、C-replay-001源变化拒绝、test_model路径错误及图形QA运行时/几何口径错误。没有删失败测试或修改旧调度规则。

测试说明软件实现满足已测约束，不证明IT/HVAC实机准确、未知未来下的严格deadline保证、跨窗口泛化或学术创新成立。
