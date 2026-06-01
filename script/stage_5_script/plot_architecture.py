'''
Draws the Stage 5 two-layer GCN architecture diagram.

Run from the repo root:
    python3 -m script.stage_5_script.plot_architecture
'''

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULT_DIR = os.path.join(REPO_ROOT, 'result', 'stage_5_result')


def block(ax, x, y, w, h, text, color):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.08',
        linewidth=1.5, edgecolor='#333333', facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=9.5, wrap=True)


def arrow(ax, x1, y1, x2, y2, text=''):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16,
        linewidth=1.4, color='#333333'))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.07, text,
                ha='center', va='bottom', fontsize=8.5, color='#444444')


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    ax.axis('off')

    y, h, w = 1.3, 1.0, 1.7
    gap = 0.45
    x = 0.2

    blocks = [
        ('Input\nX  (N x F)\nA_hat (N x N)', '#dbeafe'),
        ('GraphConv 1\nA_hat X W1\nF -> 16', '#bfdbfe'),
        ('ReLU', '#fde68a'),
        ('Dropout\np = 0.5', '#fde68a'),
        ('GraphConv 2\nA_hat H W2\n16 -> C', '#bfdbfe'),
        ('LogSoftmax\n(C classes)', '#bbf7d0'),
        ('Node\nPredictions', '#dbeafe'),
    ]

    centers = []
    for i, (text, color) in enumerate(blocks):
        block(ax, x, y, w, h, text, color)
        centers.append((x, x + w))
        x += w + gap

    for i in range(len(blocks) - 1):
        arrow(ax, centers[i][1], y + h / 2, centers[i + 1][0], y + h / 2)

    ax.text(6.5, 3.2, 'Stage 5  -  Two-Layer Graph Convolutional Network (GCN)',
            ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(6.5, 0.45,
            'N = #nodes,  F = #input features,  C = #classes.  '
            'A_hat = D^(-1/2)(A + I)D^(-1/2) is the symmetrically normalized '
            'adjacency with self-loops.',
            ha='center', va='center', fontsize=8.5, color='#555555')

    fig.tight_layout()
    out_path = os.path.join(RESULT_DIR, 'gcn_architecture_plot.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'architecture diagram saved to {out_path}')


if __name__ == '__main__':
    main()
