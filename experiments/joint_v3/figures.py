"""Method diagram from implemented data flow; quantitative data are never invented."""
import argparse
from pathlib import Path


def framework(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(7.1, 4.2), layout='constrained')
    ax.set(xlim=(0, 10), ylim=(0, 6)); ax.axis('off')
    boxes = {}
    def box(name, xy, text, color='#e8eef3', width=2.6, height=.8):
        x, y = xy; boxes[name] = (x, y, width, height)
        ax.add_patch(FancyBboxPatch(xy, width, height, boxstyle='round,pad=0.05',
            facecolor=color, edgecolor='#607080', linewidth=.65))
        ax.text(x+width/2, y+height/2, text, ha='center', va='center', fontsize=7)
    def arrow(a, b):
        x, y, w, h = boxes[a]; xx, yy, ww, hh = boxes[b]
        start = (x+w/2, y) if yy < y else (x+w, y+h/2)
        end = (xx+ww/2, yy+hh) if yy < y else (xx, yy+hh/2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=8,
                                     linewidth=.7, color='#566573'))
    box('online', (3.7, 4.95), 'Observed jobs and resources\nCurrent CI and weather')
    box('risk', (.35, 3.7), 'Paired EDF reservations\nQueue risk and safe wait budget')
    box('carbon', (6.95, 3.7), 'Train-only CI forecast\nIT + HVAC robust carbon gain')
    box('task', (3.7, 2.6), 'Risk shield + Task Actor (40)\nSequential task actions', '#dbe8ef')
    box('cooling', (3.7, 1.35), 'Post-task Cooling Actor (17)\nOne legal temperature action', '#dbe8ef')
    box('plant', (3.7, .1), 'Unchanged physical executor\nOne 15-minute system feedback', '#e8e5dc')
    box('critic', (.35, .7), 'Boundary-specific global critics\nCarbon and Deadline costs', '#e8e5dc', height=1.1)
    box('ppo', (6.95, .7), 'Complete-episode masked PPO\nGrouped constraints + auxiliary loss', '#e8e5dc', height=1.1)
    for a, b in [('online','risk'), ('online','carbon'), ('risk','task'), ('carbon','task'),
                 ('task','cooling'), ('cooling','plant'), ('critic','ppo')]: arrow(a,b)
    ax.add_patch(FancyArrowPatch((3.65, .5), (3.02, 1.05), arrowstyle='-|>', mutation_scale=8,
                                 linewidth=.7, color='#566573'))
    ax.text(.4, 2.65, 'Conditional safety on known jobs;\nno unknown-arrival guarantee', fontsize=7,
             ha='left', va='center', color='#45515c')
    ax.text(7.0, 2.65, 'Matched full-work responsibility;\nuncertain tails disable positive guidance',
             fontsize=7, ha='left', va='center', color='#45515c')
    ax.text(5, 5.95, 'RAC-MAPPO: causal risk and carbon guidance', ha='center', va='center', fontsize=9)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    paths = [str(root/('method-framework.'+s)) for s in ('pdf', 'svg', 'png')]
    fig.savefig(root/'method-framework.pdf')
    fig.savefig(root/'method-framework.svg')
    fig.savefig(root/'method-framework.png', dpi=600)
    plt.close(fig)
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    a = p.parse_args(argv); print(framework(a.output))


if __name__ == '__main__': main()
