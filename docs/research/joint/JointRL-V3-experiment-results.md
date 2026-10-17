# JointRL-V3实验结果

当前完成的正式Development评价文件数：**0**；实际场景结果行数：**0**。流水线状态：`prepared`。没有正式结果时，不提供减碳率、优势判断或mean±std。

## 冻结比较

| 方法 | 任务控制 | 制冷控制 | 状态 |
|---|---|---|---|
| B0 | 资源允许即EDF启动 | 整个固定窗口22°C | 以analysis实际结果为准 |
| B1 | 资源允许即EDF启动 | 整个固定窗口18°C | 以analysis实际结果为准 |
| B2 | EDF立即启动 | V3 Cooling Actor | 以analysis实际结果为准 |
| B3 | 原V2第52轮Task Actor | 原V2第52轮Cooling Actor | 原权重只读评价 |
| B4 | V3 RAC Task Actor | 固定18°C | 以analysis实际结果为准 |
| B5 | V3 RAC Task Actor | V3 Cooling Actor | 以analysis实际结果为准 |

学习方法复用V2在排空后逐拍降至18°C的算子，并核算其电量；固定温度基线在整个运营窗口保持指定温度。所有方法共用相同任务、物理模型、资源容量、CI、天气与窗口。碳排主表包括全固定窗口总碳；增量碳统一相对于同一idle18背景，两个减排率分母分别计算。

五个消融分别训练：无队列风险、无边际碳特征、无鲁棒修正、独立局部Critic、无教师。每版本3个种子、同预算，不以测试时关闭模块替代重新训练。

## 已完成评价的实际数值

| 版本/方法 | 实际种子数 | 总碳kg | 增量碳kg | DMR |
|---|---:|---:|---:|---:|
| 尚无完成的正式评价 | 0 | 缺失 | 缺失 | 缺失 |

## 真实结果入口

- 汇总表：`/mnt/d/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001/research_notes/joint_v3/formal-001/analysis/aggregate.csv`。
- 分种子/分场景表、来源哈希与缺失说明：campaign/analysis。
- 碳、增量碳、DMR与IT/HVAC分解图：有完整实际评价后生成；没有结果时保持缺失。
- 多压力/SLA为同源增强场景，聚合结果不把它们视为独立实测。重复基线由算法种子产生的标准差通常为0，不能误称数据不确定性。
- 风险预测误差依据实际完成/违约结果；碳收益符号误差依据执行后独立真实CI/天气校准。已知队列反事实标签与最终实际策略减排是不同指标。

当前不能确认是否超过V2、Immediate+18°C或是否存在额外计算—制冷协同收益。 即便B0减排显著，也需检查B1、B2、B4与分项结果及各场景DMR。最终机制结论须等待所有预设实验完成，不能只看均值之比。

## September Test

本轮未执行Test。先完成18个冻结训练/消融与Development对照，然后生成最终评价freeze。需用户明确授权文件绑定该freeze SHA256，再运行：

```bash
python -m experiments.joint_v3.test_once --freeze <final-evaluation-freeze.json> --authorization <用户明确授权.json> --source-root /mnt/d/A-myprofile/SustainCluster/upstream --output <september-test-once目录>
```

授权文件必须含`authorize_september_test: true`及准确`freeze_sha256`。入口在授权、完整实验、源码身份校验通过前不会打开Test来源；随后先创建跨输出目录共享的一次尝试标记，再校验来源字节并执行。失败也不自动重试。原V1/V2/E1加载器仍永久拒绝Test。
