"""从已保存A1/E1汇总生成四张开题图；不加载任务数据、不运行仿真或训练。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
A1 = ROOT / 'docs/research/object-validity/data/a1/rules/candidate-summary.csv'
E1 = ROOT / 'docs/research/sca-ppo/e1-results/case-summary.csv'
ANALYSIS = ROOT / 'docs/research/sca-ppo/e1-results/analysis-summary.json'
UNITS = ROOT / 'docs/research/platform-v2-final/evidence/result-methods.json'
CAPS = ('OL', 'OM', 'OH')
METHODS = ('EDF-CG', 'CarbonRule-CG', 'CarbonRule-DDF-CG')
LABELS = ('EDF基线', '碳感知推迟', '碳感知推迟\n＋DDF保护')
COLORS = ('#78838D', '#B5834A', '#387D76')
CAP_COLORS = ('#A5B8C7', '#6A8DA8', '#365D7A')
STEMS = ('fig1_resource_deadline_validation', 'fig2_rule_compare_carbon_dmr',
         'fig3_tradeoff_carbon_qos', 'fig4_deadline_guard_behavior')
TITLES = ('图1 资源压力与 Deadline 风险验证',
          '图2 三种规则算法的碳排放与 Deadline 违约率对比',
          '图3 不同调度策略的碳排放—任务服务质量权衡',
          '图4 Deadline 保护过滤机制行为统计')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_sources():
    a1 = csv_rows(A1)
    chosen = [r for r in a1 if r['candidate_id'] in CAPS and r['policy'] == 'edf'
              and float(r['resource_pressure_factor']) == 1
              and float(r['sla_multiplier']) == 1.5]
    assert len(chosen) == 3 and {r['candidate_id'] for r in chosen} == set(CAPS)
    a = {r['candidate_id']: r for r in chosen}
    assert len({r['arrived'] for r in chosen}) == 1
    for r in chosen:
        assert int(r['scenarios']) == 16 and r['completed'] == r['arrived']
        assert np.isclose(float(r['capacity_waited_fraction']),
                          int(r['capacity_waited_tasks']) / int(r['arrived']))
        assert np.isclose(float(r['dmr']), int(r['deadline_misses']) / int(r['arrived']))
    e1 = csv_rows(E1)
    assert len(e1) == 27
    fragments = sorted({r['fragment'] for r in e1})
    assert fragments == ['train-20230302T0500-r1-s1.5',
                         'train-20230404T0500-r1-s1.5',
                         'train-20230501T0500-r1-s1.5']
    for fragment in fragments:
        group = [r for r in e1 if r['fragment'] == fragment]
        assert len(group) == 9
        assert {(r['capacity'], r['method']) for r in group} == {
            (cap, method) for cap in CAPS for method in METHODS}
        assert len({r['arrived'] for r in group}) == 1
        assert len({r['workload_sha256'] for r in group}) == 1
    for r in e1:
        assert r['valid_case'] == 'True' and r['verification_pass'] == 'True'
        assert int(r['unfinished']) == int(r['deadline_censored']) == 0
        assert r['completed'] == r['arrived'] and int(r['reference_guard_failure_count']) == 0
        assert r['actual_common_idle_tail_valid'] == 'True'
        assert float(r['horizon_hours']) == 72 and float(r['arrival_hours']) == 24
        assert int(r['attempts']) == int(r['approved']) + int(r['rejected'])
    analysis = json.loads(ANALYSIS.read_text(encoding='utf-8-sig'))
    assert analysis['matrix_complete'] and analysis['valid_case_count'] == 27
    assert json.loads(UNITS.read_text(encoding='utf-8-sig'))['units']['carbon'] == 'kgCO2e'
    return a1, a, e1, fragments


def aggregate(rows):
    n = sum(int(r['arrived']) for r in rows)
    miss = sum(int(r['deadline_missed']) for r in rows)
    return dict(tasks=n, miss=miss, dmr=100 * miss / n,
                carbon=sum(float(r['carbon_kg']) for r in rows),
                incremental=sum(float(r['common_incremental_carbon_kg']) for r in rows),
                attempts=sum(int(r['attempts']) for r in rows),
                approved=sum(int(r['approved']) for r in rows),
                rejected=sum(int(r['rejected']) for r in rows))


def configure(font_path):
    candidates = [font_path] if font_path else [Path('C:/Windows/Fonts/msyh.ttc'),
        Path('C:/Windows/Fonts/simhei.ttf'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    available = [Path(p) for p in candidates if Path(p).is_file()]
    if not available:
        raise RuntimeError('未找到中文字体；请用 --font 指定已安装的中文TTF/OTF/TTC字体。')
    font_manager.fontManager.addfont(available[0])
    family = font_manager.FontProperties(fname=available[0]).get_name()
    plt.rcParams.update({'font.family': [family, 'DejaVu Sans'],
        'font.sans-serif': [family, 'DejaVu Sans'], 'font.size': 13,
        'axes.labelsize': 14, 'axes.titlesize': 14, 'xtick.labelsize': 12,
        'ytick.labelsize': 12, 'legend.fontsize': 12, 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'svg.hashsalt': 'sca-ppo-proposal-2026-10-22', 'savefig.dpi': 600})
    return family


def canvas(number, subtitle, two=False):
    fig = plt.figure(figsize=(12, 5.8), layout=None)
    fig.suptitle(TITLES[number - 1], fontsize=18, fontweight='bold', y=.98)
    fig.text(.5, .894, subtitle, ha='center', fontsize=12, color='#4D555B')
    if two:
        axes = [fig.add_axes([.095, .23, .365, .525]),
                fig.add_axes([.605, .23, .365, .525])]
    else:
        axes = [fig.add_axes([.125, .23, .83, .525])]
    for ax in axes:
        ax.grid(axis='y', color='#DDE1E4', linewidth=.65, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    return fig, axes


def save(fig, number):
    # Explicit limits and margins: no cropping, broken bar axes, or hidden data.
    stem = OUT / STEMS[number - 1]
    fig.savefig(stem.with_suffix('.png'), dpi=600, facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    svg = stem.with_suffix('.svg')
    fig.savefig(svg, metadata={'Date': None})
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf-8').splitlines())
                   + '\n', encoding='utf-8', newline='\n')
    plt.close(fig)


def fig1(a):
    fig, axes = canvas(1, 'A1历史研究对象验证｜EDF规则｜SLA＝1.5，资源压力因子＝1.0｜16个Train窗口', True)
    x = np.arange(3)
    for ax, key, ylabel, panel in zip(axes, ('capacity_waited_fraction', 'dmr'),
            ('经历容量等待的任务比例（%）', 'Deadline违约率（%）'),
            ('(a) 资源竞争', '(b) Deadline风险')):
        values = [100 * float(a[c][key]) for c in CAPS]
        ax.bar(x, values, width=.55, color=CAP_COLORS, zorder=3)
        upper = max(values) * 1.3
        ax.set(xticks=x, xticklabels=['OL\n低压力', 'OM\n主场景', 'OH\n高压力'],
               ylim=(0, upper), ylabel=ylabel, xlabel='容量场景（由低至高压力）')
        ax.set_title(panel, loc='left', pad=15)
        for xx, yy in zip(x, values):
            ax.text(xx, yy + upper * .035, f'{yy:.2f}%', ha='center', fontsize=13)
    fig.text(.5, .035, '同一组52,339个任务；比例按任务数汇总，非跨窗口均值；不含主动推迟。',
             ha='center', fontsize=12, color='#4D555B')
    save(fig, 1)


def fig2(om):
    fig, axes = canvas(2, 'E1规则实验｜OM主场景｜全部3个Train片段｜每种方法9,098个相同任务', True)
    x = np.arange(3)
    for ax, key, ylabel, panel in zip(axes, ('carbon', 'dmr'),
            ('总碳排放（kg CO₂e）', 'Deadline违约率（%）'),
            ('(a) IT＋HVAC总碳排放', '(b) 任务服务质量')):
        values = [om[m][key] for m in METHODS]
        upper = max(values) * 1.28
        ax.bar(x, values, width=.55, color=COLORS, zorder=3)
        ax.set(xticks=x, xticklabels=LABELS, xlabel='调度策略', ylabel=ylabel, ylim=(0, upper))
        ax.set_title(panel, loc='left', pad=15)
        for xx, yy in zip(x, values):
            text = f'{yy:,.2f}' if key == 'carbon' else f'{yy:.2f}%'
            ax.text(xx, yy + upper * .035, text, ha='center', fontsize=12)
    fig.text(.5, .035, '左：3个完整72h回合的碳排合计；右：总违约任务数／总任务数。共同完成护栏均通过。',
             ha='center', fontsize=12, color='#4D555B')
    save(fig, 2)


def fig3(om):
    fig, (ax,) = canvas(3, '与图2完全相同的OM结果｜每个点代表一种策略｜无新增实验')
    xs = [om[m]['dmr'] for m in METHODS]
    ys = [om[m]['carbon'] for m in METHODS]
    for idx, (xx, yy, label, color, marker, offset) in enumerate(zip(xs, ys,
            ('EDF基线', '碳感知推迟', '碳感知推迟＋DDF保护'), COLORS,
            ('o', 's', 'D'), ((14, 0), (-14, 18), (14, 0)))):
        ax.scatter(xx, yy, s=115, c=color, marker=marker, zorder=4,
                   edgecolors='white', linewidths=.8)
        ax.annotate(label, (xx, yy), xytext=offset, textcoords='offset points',
                    ha='right' if idx == 1 else 'left', va='center', fontsize=14)
    dx = max(xs) - min(xs); dy = max(ys) - min(ys)
    ax.set(xlim=(min(xs) - .23 * dx, max(xs) + .17 * dx),
           ylim=(min(ys) - .3 * dy, max(ys) + .28 * dy),
           xlabel='Deadline违约率（%）', ylabel='总碳排放（kg CO₂e）')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
    ax.grid(axis='x', color='#E5E7E9', linewidth=.65)
    ax.annotate('更优方向', xy=(.72, .68), xytext=(.84, .86),
                xycoords='axes fraction', fontsize=12, color='#626A70', ha='center',
                arrowprops={'arrowstyle': '->', 'color': '#93999F', 'lw': 1})
    fig.text(.5, .035, '左下方表示更低碳、更少违约；纵轴局部放大，三种策略的总碳差异小于0.3%。',
             ha='center', fontsize=12, color='#4D555B')
    save(fig, 3)


def fig4(guards):
    fig, (ax,) = canvas(4, 'E1规则实验｜CarbonRule-DDF-CG自身轨迹｜每种容量均汇总相同3个Train片段')
    x = np.arange(3)
    approved = np.array([guards[c]['approved'] for c in CAPS])
    rejected = np.array([guards[c]['rejected'] for c in CAPS])
    totals = approved + rejected
    ax.bar(x, approved, width=.5, color='#709DA8', label='批准推迟', zorder=3)
    ax.bar(x, rejected, bottom=approved, width=.5, color='#B5836E', label='拒绝推迟',
           hatch='///', edgecolor='#725347', linewidth=.35, zorder=3)
    for xx, aa, rr, total in zip(x, approved, rejected, totals):
        ax.text(xx, aa / 2, f'{aa:,}', ha='center', va='center', fontsize=15)
        ax.text(xx, aa + rr / 2, f'{rr:,}', ha='center', va='center', fontsize=15)
        ax.text(xx, total + max(totals) * .055, f'请求 {total:,}', ha='center', fontsize=14)
    ax.set(xticks=x, xticklabels=['OL（低压力）', 'OM（主场景）', 'OH（高压力）'],
           xlabel='容量场景', ylabel='候选任务请求次数（次）', ylim=(0, max(totals) * 1.25))
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=2,
               bbox_to_anchor=(.5, .005), frameon=False)
    save(fig, 4)


def write_notes(a1, a, e1, om, guards, family):
    sources = [A1, E1, ANALYSIS, UNITS]
    source_table = '\n'.join(f'| `{p.relative_to(ROOT).as_posix()}` | `{sha(p)}` |' for p in sources)
    a_table = '\n'.join(f"| {c} | {a[c]['cpu_capacity']} / {a[c]['gpu_capacity']} / {a[c]['mem_capacity']} | "
        f"{int(a[c]['capacity_waited_tasks']):,} / {int(a[c]['arrived']):,} | "
        f"{100*float(a[c]['capacity_waited_fraction']):.6f}% | {int(a[c]['deadline_misses'])} | "
        f"{100*float(a[c]['dmr']):.6f}% |" for c in CAPS)
    e_table = '\n'.join(f"| {m} | {om[m]['carbon']:.6f} | {om[m]['incremental']:.6f} | "
        f"{om[m]['miss']} / {om[m]['tasks']} | {om[m]['dmr']:.6f}% |" for m in METHODS)
    g_table = '\n'.join(f"| {c} | {guards[c]['attempts']} | {guards[c]['approved']} | "
        f"{guards[c]['rejected']} | {100*guards[c]['rejected']/guards[c]['attempts']:.3f}% |" for c in CAPS)
    notes = f'''# 开题报告科研图说明

日期：2026-10-22。仅生成4张图，每张PNG/PDF/SVG，共12个图文件；没有新调度、模型评价或训练。
全部图为定量比较图：研究问题验证 → 已完成规则实验 → 碳与服务质量权衡 → 保护机制行为。
正文优先推荐图1与图3；说明方法实现进度时可配图4。不得把这些开发性结果称为独立泛化测试或PPO优势。

## 共同数据、统计与复现口径

- Python/matplotlib；字体统一为{family}。画布12×5.8英寸，白底，标题18pt、坐标轴14pt、刻度/注释12pt及以上；PNG为600dpi（7200×3480像素），PDF嵌入TrueType字体，SVG保留可编辑文字。跨机器查看SVG需安装对应中文字体。
- 柱图均从0起；图3使用明确标记的局部纵轴，不用双y轴、不连接成最优前沿、不画训练曲线。
- 采用完整任务计数的汇总比例和完整回合碳合计，不是随机重复实验均值。因此不添加虚构SD/SE/CI或显著性标记；跨窗口、容量、方法并非独立随机样本。
- 图1：A1候选汇总原有{len(a1)}行，固定选择Observed×1、OL/OM/OH、EDF、压力因子1.0、SLA1.5，共3行；其余87行为不同映射/压力/SLA/规则，非删除困难任务。每行含相同16窗52,339任务，全部完成。
- 图2/3：E1原有27行，按用户优先OM的要求选择全部3片段×3方法共9行；未选择的18行是OL/OH容量，图4另使用其中DDF方法。OM未挑选日期或seed，每种方法9,098个相同任务。
- 图4：同一E1原表选择全部3容量×3片段的DDF方法9行，其他18行为无DDF对照，不能混入DDF机制统计。所有27回合的完整性/CG/尾段验证通过；脚本读取时逐项断言。
- E1窗口：2023-03-02、2023-04-04、2023-05-01，均05:00起，24h任务到达、72h固定评价、SLA1.5、HVAC请求20°C。它们已用于平台验收/方法开发，非独立泛化样本。
- A1是历史逻辑容量与规则诊断阶段，E1是冻结V2执行及能耗口径；即使容量标签相同，也不能把图1与图2的DMR差直接归因为某项算法。所有需求保留观测使用量代理、合成Deadline与原始重复任务的既有限制。
- 未读取原始工作负载、年度PKL或任何封存Development/Validation/Test输入；仅读取以下已存在的轻量Train结果表。

| 原始结果文件（仓库相对路径） | SHA256 |
|---|---|
{source_table}

来源结果沿用既有A1和E1源码身份，E1逐回合source_sha/workload_sha256保留在源CSV；E1执行源码SHA为`{', '.join(sorted({r['source_sha'] for r in e1}))}`。本次绘图不改变任何输入；单位由冻结V2结果方法清单独立核对。

## 图1：资源压力与 Deadline 风险验证

![图1](fig1_resource_deadline_validation.png)

**数据来源：** `{A1.relative_to(ROOT).as_posix()}`，与既有A1-capacity-competition图相同的汇总源；固定EDF/Observed×1/压力1/SLA1.5。

**Caption：** 相同16个Train窗口下，低至高压力容量的资源等待率与Deadline违约率。左图为至少经历一次容量不足等待的任务数／到达任务数，右图为违约任务数／到达任务数，均按任务汇总。等待不包括策略主动推迟。

**结论：** OL无容量等待，OM出现少量竞争，OH的容量等待和Deadline风险明显上升。这支持有限资源与Deadline耦合这一研究问题，而不是证明某个算法能够解决它。

**放置位置：** 研究背景后的“研究对象与问题验证”；答辩PPT的科学问题动机页。优先使用。

**局限：** 16窗非16个随机种子；汇总率可能掩盖短期峰值。OH的最坏窗口DMR为{100*float(a['OH']['worst_fragment_dmr']):.2f}%，有{a['OH']['fragments_with_dmr_gt5_percent']}/16窗超过5%；不能只因汇总DMR低于5%就称全部场景达标。A1不是当前V2的性能对比。

| 场景 | CPU / GPU等价 / MEM容量 | 容量等待任务 / 总任务 | 等待率 | 违约数 | DMR |
|---|---|---|---|---:|---|
{a_table}

## 图2：三种规则算法的碳排放与 Deadline 违约率对比

![图2](fig2_rule_compare_carbon_dmr.png)

**数据来源：** `{E1.relative_to(ROOT).as_posix()}` 的全部OM记录；`analysis-summary.json`核对矩阵完整性。真实方法为EDF-CG、CarbonRule-CG、CarbonRule-DDF-CG，E1没有FIFO-CG，不伪造或混用其他阶段结果。

**Caption：** OM主场景下三种已实现规则的完整72h总IT＋HVAC碳排和任务加权DMR。碳为3个相同片段回合的合计，不是单回合或均值；DMR为三回合总违约数／9,098。CG表示共同完成护栏，DDF表示Deadline非恶化过滤器。

**结论：** 碳感知推迟比EDF略低碳，但最终DMR上升；增加DDF后违约显著减少，同时放弃部分减碳机会。它展示了已实现且完整评价的规则基线，不意味着PPO已经胜出。

**放置位置：** “已完成工作／前期实验”；规则基线实现与平台验收进展页。

**局限：** 总碳中包含公共空闲背景，差异小；柱图从0开始不会夸大小幅变化，图3再展示细节。不能将局部预测保护称为未知未来到达下的全局安全保证。背景增量碳仍完整列于下表，未用增量口径替代总碳。

| 源方法名 | 总碳/kg CO₂e | 背景增量碳/kg CO₂e | 违约数 / 总任务 | DMR |
|---|---:|---:|---:|---|
{e_table}

图中文字对应：EDF基线＝EDF-CG；碳感知推迟＝CarbonRule-CG；碳感知推迟＋DDF保护＝CarbonRule-DDF-CG。碳单位kg CO₂e与冻结V2的`result-methods.json`中`units.carbon=kgCO2e`一致；仅表示既有CI积分的运行碳核算，不额外包含设备生命周期排放。

## 图3：不同调度策略的碳排放—任务服务质量权衡

![图3](fig3_tradeoff_carbon_qos.png)

**数据来源：** 与图2完全相同的9行OM源记录、同一张数值表、同一汇总方法，无额外实验或重选窗口。

**Caption：** 每个点为一种规则在相同三个OM片段上的任务加权DMR和完整回合总碳合计，左下表示更低碳与更少违约。纵轴局部放大，以显示总碳不足0.3%的细微差异；点不构成理论最优前沿。

**结论：** 无DDF推迟获得最低总碳，却具有最高违约率；带DDF规则相对它增加{om[METHODS[2]]['carbon']-om[METHODS[1]]['carbon']:.3f}kg总碳、减少{om[METHODS[1]]['miss']-om[METHODS[2]]['miss']}个违约。这给出了“碳收益与服务风险如何权衡”的直接证据，DDF只保护已知任务的局部参考结果。

**放置位置：** 核心科学问题、研究目标或方法动机页；最适合作为答辩主图之一，与图1优先组合。

**局限：** 仅三种确定性规则与一个容量的描述性比较，不提供因果机制消融、统计显著性或跨容量最优结论。不能把箭头理解为任何新算法已达到的改进轨迹。

## 图4：Deadline保护过滤机制行为统计

![图4](fig4_deadline_guard_behavior.png)

**数据来源：** `{E1.relative_to(ROOT).as_posix()}` 的CarbonRule-DDF-CG全部9回合；次数来自attempts/approved/rejected。

**Caption：** 每个容量汇总三片段DDF自身真实轨迹中的候选推迟请求。柱总高为请求总数，蓝色为批准，棕色斜线为拒绝；两段恰好相加为全部请求。次数包含同一任务跨多个时槽的重复请求，不是唯一任务数。

**结论：** DDF在三个压力水平下都拒绝约三至四成请求，表明机制确实进行筛选。高压力OH的拒绝比例更高；批准数不等于实际主动等待，因为真实执行还遵守容量先判断与回填规则。

**放置位置：** “拟采用方法／已实现保护机制”；说明算法不是盲目推迟的机制证据页。

**局限：** 每种容量的请求来自其自身轨迹，不能把不同轨迹下总请求数的差都当作DDF直接拦截的效果。源CSV中拒绝原因标记存在重叠（例如同时破坏按时集合并增加超期），因此本图使用互斥的批准／拒绝，而不直接堆叠这些原因。全局primary_reason_partition可互斥分类，但不是现成的逐容量分类表。

| 场景 | 全部请求 | 批准 | 拒绝 | 拒绝比例 |
|---|---:|---:|---:|---:|
{g_table}

## 复现与导出自检

```powershell
python docs/research/sca-ppo/proposal-figures/generate_proposal_figures.py
# 如需指定其他已安装中文字体：在同一命令后添加 --font 字体文件路径
```

脚本只读取四个既有轻量结果/方法文件，断言A1选择唯一、E1全部27回合完整且相同任务、CG和尾段有效、请求＝批准＋拒绝，缺失或不一致直接报错。SVG保留文字，PDF嵌入字体；PNG保留完整画布与600dpi，不对某个策略单独裁剪、修饰或重算结果。没有p值、置信区间、3D或双y轴。四图所有标签和图例使用同一中文字体（₂符号使用兼容无衬线回退字体），方法原名、数据单位和局限在本说明中明确对应。

最终源码预检18项通过、0失败；两项提示分别为PNG而非TIFF，以及画布比期刊单栏宽，均按本次PPT/开题用途和指定格式保留，不增加多余版本。PDF逐图最小文字12pt；缩放至183mm正文宽度时相当于约7.2pt。已逐图检查标题、子图、数值、图例与点旁标注，不存在重叠或截断；图3整数刻度与真实坐标一致。PNG尺寸与DPI、12个导出文件、全部输入源在Git中无变更均另行复核。上表记录实际读取字节的SHA256；CSV文本的CRLF/LF行尾可能随Git检出平台转换，不改变数据值。
'''
    (OUT / 'figure_notes.md').write_text(notes, encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path)
    args = parser.parse_args()
    a1, a, e1, _ = load_sources()
    family = configure(args.font)
    om = {m: aggregate([r for r in e1 if r['capacity'] == 'OM' and r['method'] == m])
          for m in METHODS}
    guards = {c: aggregate([r for r in e1 if r['capacity'] == c and r['method'] == METHODS[2]])
              for c in CAPS}
    fig1(a); fig2(om); fig3(om); fig4(guards)
    write_notes(a1, a, e1, om, guards, family)
    for stem in STEMS:
        assert all((OUT / (stem + suffix)).is_file() for suffix in ('.png', '.pdf', '.svg'))
    print('已生成4张图（PNG/PDF/SVG共12文件）及figure_notes.md；无新实验。')


if __name__ == '__main__':
    main()
