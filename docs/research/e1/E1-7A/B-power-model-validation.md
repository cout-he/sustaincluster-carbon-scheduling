# B物理模型验证与核算边界

旧power.py、upstreamIT/HVAC方程、reset seed、make_dc_env(month=1)及逐拍执行器完全不改。每次实际回放都新建一次物理模型，随后整个57h持续复用其状态。此处复用历史设置，仅说明既有研究物理实现，未验证真实硬件校准或增加关机模型。

基线9×8历史文件byteexact；A全部44真实物理分支及固定背景守恒；C全部24分支相同任务、同资源小时、完整Drain+真实空闲尾段。独立逐拍验证kW×h=kWh、IT+HVAC=总电能、kWh×g/kWh÷1000=kg，并核对零传输语义。纳秒ceil只影响占用释放，rawfinish决定deadline；不把累计reward当能碳。

额外空任务参考保持真实CI/weather连续，不使用固定末值或重复数据。该参考不是等工作量策略，HVAC不能按常数或可加差额处理。实际碳改善与局部IT代理之间的差异见B报告和C-carbon-components.csv。没有通过测试宣称物理模型准确反映真实机房。
