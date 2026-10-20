# V1.1 资料登记（执行前）

核对日期 2026-10-20。以下均为公开官方/作者来源，阅读范围按本轮实际检索列明；没有本设施实测资料。敏感性端点**全部为探索性假设**，这些来源不提供它们的统计置信区间。

| 来源 | 本轮读取范围及支持内容 | 不支持的内容 |
|---|---|---|
| [Intel 6338 规格](https://www.intel.com/content/www/us/en/products/sku/212285/intel-xeon-gold-6338-processor-48m-cache-2-00-ghz/specifications.html) | 官方规格表：32 核/64 线程、205 W TDP、八内存通道 | 不提供 45 W 空闲或工作负载功率曲线 |
| [Intel TDP 说明](https://www.intel.com/content/www/us/en/support/articles/000055611/processors.html?wapkw=tdp) | 定义、设计目标与 Turbo/AVX 可短时超 TDP 的说明 | 205 W 不是此观测使用量对应的测量值；240 W 探索点也不是实测 |
| [NVIDIA V100 数据表](https://images.nvidia.com/content/pdf/vgpu/data-sheet/nvidia-v100.pdf) | 官方两页规格：PCIe 250 W 与 SXM 300 W 应区分 | 没有给本平台 25 W 空闲或分数 GPU 性能/功率依据 |
| [Dell R750 规格](https://i.dell.com/sites/csdocuments/product_docs/en/poweredge-r750-spec-sheet.pdf) | 双路、32 DIMM 槽、内存类型 | 非 V1 整机认证清单 |
| [Dell R750xa 规格](https://i.dell.com/sites/csdocuments/Product_Docs/en/poweredge-R750xa-spec-sheet.pdf) | 双路2U、四个双宽300 W位、1400/1800/2400/2800 W PSU选项；报告撰写时补核对 | 不证明V100具体兼容SKU，也不自动指定V1的PSU冗余/效率曲线 |
| [Dell R750xa 内存指南](https://www.dell.com/support/manuals/en-sg/poweredge-r750xa/per750xa_ism_pub/system-memory-guidelines?guid=guid-51c3f7cc-c06f-4114-8547-ef319e3fc668&lang=en-us) | 每 CPU 八通道/16 槽，支持 8 GB 单 Rank 与 16 GB 双 Rank RDIMM | 不保证不同通道/Rank 的任务性能不变 |
| [Dell R750xa 内存规格](https://www.dell.com/support/manuals/en-ng/poweredge-r750xa/per750xa_ts_pub_ism/Memory-specifications?guid=guid-bef42bb9-d278-42fe-a34b-df7508d58f15&lang=en-us) | 8/16 GB 规格及双路最小/最大安装；8 GB 与 BPS 限制 | 不给 DIMM 3/5 W 依据 |
| [Dell 双路内存安装规则](https://www.dell.com/support/manuals/en-us/powerflex-custom-node/vxf-p-owners-guide-r750/general-memory-module-installation-guidelines?guid=guid-6db30d04-e447-48c6-8748-d6389f356fdb&lang=en-us) | R750 衍生系统官方手册：两路安装匹配、8/16 DIMM 可用，通道配置影响性能 | 不作为已认证的 V1 任意混插方案 |
| [CLEAResult 80 PLUS 计划](https://www.clearesult.com/80plus/program-details-information) | 官方认证与测试框架；核对计划页及其测试入口 | 未取得对应 V1 PSU 型号效率曲线，不用认证等级证明全时 94% |
| [LBL Modelica Buildings Chillers](https://simulationresearch.lbl.gov/modelica/releases/latest/help/Buildings_Fluid_Chillers.html) | Carnot_TEva/Carnot_y 文档：Carnot 温度关系、名义效率、部分负载修正及动态节点 | 示例名义效率不校准 V1 的 0.35，V1 也不是该完整模型 |
| [EnergyPlus 工程参考](https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v24.2.0/EngineeringReference.pdf) | 冷机章节/官方索引：容量与 EIR 的温度、部分负载曲线及厂家拟合依据 | 不为无名冷机的容量、COP 上下界或热容背书 |
| [Sun 等，2021，数据中心原型模型](https://eta-publications.lbl.gov/sites/default/files/prototype_energy_models_for_data_centers.pdf) | 原作者公开 32 页稿，设备类型与部分负载/变速风机段落 | V1 未采用其具体校准原型，不可搬用其现实验证 |
| [ASHRAE TC9.9 白皮书](https://resourcecenter.ashrae.org/File%20Library/Technical%20Resources/Bookstore/ASHRAE_TC0909_Power_White_Paper_22_June_2016_REVISED.pdf) | 推荐入口温度范围、进风环境与设备类型的说明 | 不能把单个平均温度≤27°C变成每架热点安全认证 |
| [Wang 等安全制冷作者稿](https://tanrui.github.io/pub/SafeDRL-TCPS.pdf) | 单节点假设、热状态拟合、安全修正和非均匀温度扩展段落 | 不提供本设施 C=500 kJ/K/host、UA=2 W/K/host 的依据 |

GPU 25 W 空闲的“论文先例”只继承 P1 对 DCcluster-Opt 补充材料的记录，未上升为独立产品测量。本轮不再追查 Alibaba 原始表。CPU/GPU/MEM 需求、Deadline 和处理后重复的局限按 P1 保留。
