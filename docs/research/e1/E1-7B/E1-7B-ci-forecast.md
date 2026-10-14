# 严格在线CI预测与冻结后评估

F1优先、F0消融在Validation前固定；没有根据Validation重新选模型。源记录、滞后和当前观测时间由Forecast.sources逐项返回，测试所有来源<=now。相同时间同值observe幂等，冲突/倒退拒绝。数值负预测截0。CI为真实小时数据的上一观测持有；不是原生15min预测精度。滚动48h缓存仅历史，预热前缀不含窗口未来CI。

Train同一CI（三倍率不是独立重复），每个horizon有188/184/176个目标，重叠预测相关。MAE/RMSE单位gCO2e/kWh，方向为sign(pred-current)==sign(actual-current)，Persistence预测零变化，当前窗口这些跨度实际均改变，因此其方向一致为0。

| method | horizon_steps | samples | mae | rmse | direction_accuracy | cold_samples |
| --- | --- | --- | --- | --- | --- | --- |
| persistence | 4 | 188 | 10.661702 | 16.698009 | 0.000000 | 0 |
| persistence | 8 | 184 | 20.307391 | 29.927210 | 0.000000 | 0 |
| persistence | 16 | 176 | 34.913409 | 46.712418 | 0.000000 | 0 |
| seasonal | 4 | 188 | 7.971915 | 10.464805 | 0.765957 | 0 |
| seasonal | 8 | 184 | 13.216087 | 16.684145 | 0.782609 | 0 |
| seasonal | 16 | 176 | 17.118864 | 20.666507 | 0.886364 | 0 |

下面为方法全部冻结后的开发Validation诊断，未用于拟合或参数选择，也不代表独立Test：

| method | horizon_steps | samples | mae | rmse | direction_accuracy | cold_samples |
| --- | --- | --- | --- | --- | --- | --- |
| persistence | 4 | 188 | 11.269574 | 18.859369 | 0.000000 | 0 |
| persistence | 8 | 184 | 21.500000 | 33.318762 | 0.000000 | 0 |
| persistence | 16 | 176 | 39.495000 | 53.383631 | 0.000000 | 0 |
| seasonal | 4 | 188 | 3.486383 | 4.298680 | 0.893617 | 0 |
| seasonal | 8 | 184 | 4.635217 | 6.223916 | 0.869565 | 0 |
| seasonal | 16 | 176 | 4.778636 | 6.603927 | 0.931818 | 0 |

两窗口24h预热完整、cold_samples=0。长期>24h不合法外推，任务所需N+1超过96直接未知并执行。F1在本窗口优于F0，不据此引入复杂预测器。良好CI方向不等于整集群延迟收益方向正确。
