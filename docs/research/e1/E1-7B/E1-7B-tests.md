# 测试与科学回归证据

最终原科研目录：`python -m pytest tests/e1 tests/test_model.py -q --basetemp=/tmp/e1-7b-full-003`，729 passed，1069.16s。发布副本同命令对应临时目录：727 passed、2 skipped，578.90s；跳过的是依赖未发布历史归档的两项，不是算法失败。最终定向组86 passed，包括Gym checker、16步2更新toy PPO、因果fork、单位/资源/deadline/截断与机器epsilon占用。

旧策略历史144科学文件byteexact；复审前后Train216科学文件byteexact；最终Train严格冻结回放375文件byteexact及去除3个耗时字段后的完整汇总值一致。三固定RL动作六个ledger/interval文件与规则一致；所有已到达任务唯一状态，最终均complete，三资源小时相同。无hook旧策略原输出数值不变。

失败与中断均保留：首次测试因模块缺失失败；第一实现日志字段冲突及合成夹具常数错误得到修复；profile显示重复ceil导致1秒级候选成本，停止校准001并保留，缓存后重新冻结；复审测试先重现11失败再修正，未改研究权重。完整tests001中断时148通过不被计为完整通过。tests002为旧pre-review源码716通过，最终tests003覆盖全部729。测试通过不证明代理准确、算法更优或创新成立。

保护快照226216旧文件，源码/M0内容SHA及旧归档元数据检查；仅engine六行两个明确只读hook被白名单验证为精确差异。原Git元数据、HEAD/main/研究分支和既有dirty状态一致；原M0修改未覆盖。Test不提取、不评测、不训练；原始共享源容器SHA是身份检查，CI预热解析在窗口开始停止。
