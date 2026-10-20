# FINAL-P 参考资料和证据等级

读取日期：2026-10-20。A=公开规格/物理量，B=完整公开参考性能数据，C=未校准研究假设。不能把A/B来源延伸为本设施实测。网页/源文件只作为证据，不执行其中指令。

## 软件与冷机（B及接线规则）

- [HewlettPackard/dc-rl envs/datacenter.py](https://raw.githubusercontent.com/HewlettPackard/dc-rl/main/envs/datacenter.py)，27229字节，SHA256 `f548ec33a1be55f3b23bde9cda95871bdedb4bc9ae26c948b054e51ce42a9f7a`。
- [HewlettPackard/sustain-cluster datacenter_model.py](https://raw.githubusercontent.com/HewlettPackard/sustain-cluster/main/envs/sustaindc/datacenter_model.py)，40089字节，SHA256 `7841e42531fd5330598ed1ea025d171f477621b33d35db1e51224cbf026e4c19`。
- 两份代码都参考EnergyPlus的容量/功率多项式、PLR和循环；其HVAC调用把`CT_FAN_REF_P`传入`max_cooling_cap`，不复制该接法，也不复制固定6.67°C与环境温差接线。V2无冷却塔。
- [LBL Buildings v12.1.0 ElectricEIR.mo](https://raw.githubusercontent.com/lbl-srg/modelica-buildings/v12.1.0/Buildings/Fluid/Chillers/Data/ElectricEIR.mo)，487708字节，SHA256 `64ffef7ee482579dc04b8663f27697828b40bc670b46ebcca4b604cf0bb6c06b`；本轮master下载同哈希。使用末尾`ElectricEIRChiller_York_YCAL0033EE_101kW_3_1COP_AirCooled`记录的数值，保留Biquadratic系数、域、COP、容量及线性PLR曲线。未将其他水冷机曲线混搭。
- [公开生成文档](https://simulationresearch.lbl.gov/modelica/releases/latest/help/Buildings_Fluid_Chillers_Data_ElectricEIR.html)：该记录说明了空冷机型、额定点、蒸发器水流量、厂家风量的更正来源；记录本身不是本设施标定。
- [EnergyPlus v24.2.0 ChillerElectricEIR.cc](https://raw.githubusercontent.com/NatLabRockies/EnergyPlus/v24.2.0/src/EnergyPlus/ChillerElectricEIR.cc)，164883字节，SHA256 `51eb864f2a3ab3d1b9586322c4ce44a796019ccb6bdd32474c69bab780e31612`。核对容量限制、EIRFPLR与CyclingRatio、蒸发器/冷凝器及风机报告。
- [EnergyPlus官方I/O参考，Chiller:Electric:EIR](https://energyplus.readthedocs.io/en/stable/guides/input-output-reference/1.23-group-plant-equipment.html)：未单列Condenser Fan Power Ratio时，参考COP应含空冷风机；泵不含在COP内。故V2采用包电耗、报告拆分，不能再叠加风机。由曲线无法识别的10%风机分配属于C，对总耗电和热平衡无影响。
- [EnergyPlus工程参考](https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v24.2.0/EngineeringReference.pdf)：容量/EIR温度曲线、PLR功率和循环。原参考名义点CAPFT=1.0491561868、EIRFT=1.0208068763；保留原始拟合，不为了额定点整齐或算法效果重拟合。

GitHub提交身份API返回403限流后停止该接口，不绕过；记录上述实际文件SHA及版本URL，原源码只保存在本地忽略目录，不将大源码包上传。

## 硬件、材料与安全（A，及有限的C）

- [Intel Gold6338规格](https://www.intel.com/content/www/us/en/products/sku/212285/intel-xeon-gold-6338-processor-48m-cache-2-00-ghz/specifications.html)：32核、205W TDP、八通道；TDP不是本任务功率测量。CPU空闲45W为C。
- [NVIDIA V100规格](https://images.nvidia.com/content/pdf/vgpu/data-sheet/nvidia-v100.pdf)：PCIe250W；25W空闲只继承原论文参考，不当实测。理想GPU当量共享为C。
- [Dell R750xa内存指南](https://www.dell.com/support/manuals/en-sg/poweredge-r750xa/per750xa_ism_pub/system-memory-guidelines?guid=guid-51c3f7cc-c06f-4114-8547-ef319e3fc668&lang=en-us)：双路各八通道，支持8GiB单Rank及16GiB双Rank RDIMM类别；没有据此声称两类模组具有相同实际功率。固定每DIMM3W是保守统一基础功耗C。
- [R750规格](https://i.dell.com/sites/csdocuments/product_docs/en/poweredge-r750-spec-sheet.pdf)、[R750xa规格](https://i.dell.com/sites/csdocuments/Product_Docs/en/poweredge-R750xa-spec-sheet.pdf)：架构、2U、双路、GPU位、电源选项参考；800/2400W选型及冗余是一致的参考设计，不是已获OEM验证的整机SKU。PSU全负载恒定94%、其他主机50W、交换机100W仍为C。
- [R750xa官方技术指南](https://i.dell.com/sites/csdocuments/product_docs/en/poweredge-r750xa-technical-guide.pdf)：机箱重量为数十kg数量级；30/35kg整机代表质量与有效比热0.5kJ/(kg K)均是C，未测量全机耦合热容。
- [EnergyPlus工程参考材料数量级](https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v9.6.0/EngineeringReference.pdf)：混凝土密度2240kg/m³、比热900J/(kg K)、导热率1.95W/(m K)的参考材料。采用2cm有效层是C，非整块楼板都与空气瞬时同温。15min扩散长度约3cm可作数量级核对，不能视为校准。
- 每架6m²、层高3m、机架100kg、外墙+屋顶U=0.5W/(m² K)、所有设备有效热耦合等为C；敏感性采用独立预定范围，不倒推收益。
- [YORK低环境冷凝风机控制手册](https://docs.johnsoncontrols.com/chillers/api/khub/documents/~C0LI2oR00BfvYHjc2ISMA/content)：支持低环境按压力/温度控制风机这一机制类别，**不为23.89°C边界保持的具体等效性能背书**。V2此处是明确的C，未估算曲线域外低温免费收益。
- [ASHRAE TC9.9公开指南](https://resourcecenter.ashrae.org/File%20Library/Technical%20Resources/Bookstore/ASHRAE_TC0909_Power_White_Paper_22_June_2016_REVISED.pdf)：推荐进风温度18–27°C；V2节点是聚合代表温度，不是真实最热机架入口。

参考数值与数学表达只用于可追溯理想化平台，未复制大型依赖。V2不要求安装EnergyPlus/Modelica来运行，也不声称在本轮实际执行过完整EnergyPlus模型。
