# E1-5B PPO-v0 配置

预注册配置：`experiments/e1/configs/e1_5b_ppo.json`。该文件在真实策略预检查和 PPO 更新之前落盘。早期 CI 校准阈值固定 161.075，默认 E1-5A Reward 与 40 维工程归一化未调整。

框架为 Stable-Baselines3 2.9.0，MlpPolicy，两层各 64 单元；CPU，单 Torch 线程，确定性算法。learning_rate=3e-4、n_steps=256、batch_size=64、n_epochs=10、gamma=.99、gae_lambda=.95、clip_range=.2、ent_coef=0、vf_coef=.5、max_grad_norm=.5、normalize_advantage=true、target_kl=null。不使用 VecNormalize，不拟合 Validation/Test 分布。

预算与种子：smoke 1024 步/55100；pilot 20480 步/55101，运行正常时作为同预算正式候选；另外 55102/55103 各 20480 步。三个正式候选共用相同 Train 冻结输入和 stress seed510001，Validation stress seed520001。没有挑 seed 或根据结果调 Reward。完整 episode 自己 Drain 后终止；安全数据终点截断会停止正式训练并保留失败记录。

learn 预算到达时可能仍处于 episode 内；这一部分是 rollout 采样尾部，其价值按 PPO 常规 bootstrap 处理，不是人为环境截断，没有额外伪造终止罚分，也不属于整任务集合排空的评价。完整 episode 与预算尾部统计分开。

现有 Conda 的 Python3.10.22 / torch2.14.1 / Gymnasium1.4.0 / NumPy2.2.6 / pandas2.3.3 均保留。SB3 2.9.0 通过 `pip install --no-deps --target research_notes/E1/E1-5B/runtime-deps stable-baselines3==2.9.0` 新增到本地目录；实际版本见 dependencies.json。框架的依赖声明与现有版本相容。[SB3 依赖声明](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.0/setup.py)

使用标准 SB3 checkpoint 保存 actor/critic、优化器及算法状态；逐参数 canonical SHA、checkpoint SHA、256 个真实观察的确定性 roundtrip 和优化器 state 共同验证更新与加载。ZIP 文件 hash 是产物身份，不保证重跑时 ZIP 元数据完全一样。[SB3 保存格式](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/save_format.html)

Validation 模型准入：全部完成且 DMR <=5%；准入后依次以公共窗口碳、DMR、seed、checkpoint SHA 选择。不按训练 reward 或 own Drain 碳排序。资格标准固定在看到候选结果之前；没有合格模型时如实报告 none。
