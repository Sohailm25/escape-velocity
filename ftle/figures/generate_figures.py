#!/usr/bin/env python3
"""Generate FTLE figures in Everforest dark style."""
import json, sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Everforest palette
EF = {
    'bg': '#2d353b', 'bg_dim': '#232a2e', 'bg_card': '#343f44',
    'text': '#d3c6aa', 'muted': '#859289',
    'green': '#A7C080', 'aqua': '#83C092', 'blue': '#7fbbb3',
    'red': '#e67e80', 'orange': '#e69875', 'yellow': '#dbbc7f',
    'purple': '#d699b6', 'border': '#4a555b',
}
CONDITION_COLORS = {
    'HOMO_A': EF['blue'], 'HOMO_B': EF['orange'],
    'HOMO_C': EF['purple'], 'HETERO_ROT': EF['aqua'],
}
CONDITION_LABELS = {
    'HOMO_A': 'Llama-3.1-8B', 'HOMO_B': 'Qwen2.5-7B',
    'HOMO_C': 'Mistral-7B', 'HETERO_ROT': 'HETERO_ROT',
}

def setup_style():
    plt.rcParams.update({
        'figure.facecolor': EF['bg'], 'axes.facecolor': EF['bg'],
        'text.color': EF['text'], 'axes.labelcolor': EF['text'],
        'xtick.color': EF['muted'], 'ytick.color': EF['muted'],
        'axes.edgecolor': EF['border'], 'grid.color': EF['border'],
        'grid.alpha': 0.4, 'font.size': 11, 'font.family': 'sans-serif',
        'legend.facecolor': EF['bg_card'], 'legend.edgecolor': EF['border'],
        'legend.labelcolor': EF['text'],
    })

def save(fig, name, outdir):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(outdir, f'{name}.{ext}'), dpi=200, bbox_inches='tight',
                    facecolor=EF['bg'], edgecolor='none')
    print(f'  Saved {name}')

# === Paths ===
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(base)
ftle = pd.read_csv(os.path.join(root, 'results/phase2/ftle_bundle.csv'))
papera = pd.read_csv(os.path.join(root, 'results/internal/paper-a/phase3_baseline/analysis_bundle.csv'))
with open(os.path.join(root, 'results/phase3/bridge_results.json')) as f:
    bridge = json.load(f)
with open(os.path.join(root, 'results/phase2/raw_results.json')) as f:
    raw = json.load(f)
outdir = os.path.dirname(os.path.abspath(__file__))
setup_style()

# Merge datasets
merged = ftle.merge(papera, on=['condition', 'seed_id', 'repeat_index'], suffixes=('_ftle', '_pa'))

# ===================== FIGURE 1: Correlation Heatmap =====================
print('Figure 1: correlation heatmap')
tests = bridge['primary_results']['tests']
summaries = ['lambda1_mean', 'lambda1_var', 'lambda1_slope']
metrics = ['collapse_rate', 'first_collapse_turn', 'collapse_incidence']
summary_labels = ['λ₁ mean', 'λ₁ variance\n(layerwise)', 'λ₁ slope\n(depth profile)']
metric_labels = ['Collapse\nrate', 'First collapse\nturn', 'Collapse\nincidence']

rho_grid = np.full((3, 3), np.nan)
pass_grid = np.full((3, 3), False)
test_map = {(t['lambda1_summary'], t['paper_a_metric']): t for t in tests}
for i, s in enumerate(summaries):
    for j, m in enumerate(metrics):
        t = test_map.get((s, m))
        if t:
            rho_grid[i, j] = t['rho']
            pass_grid[i, j] = t['pass_prereg']

fig, ax = plt.subplots(figsize=(7, 4.5))
cmap = LinearSegmentedColormap.from_list('ef', [EF['blue'], EF['bg_card'], EF['red']], N=256)
im = ax.imshow(rho_grid, cmap=cmap, vmin=-0.6, vmax=0.6, aspect='auto')
for i in range(3):
    for j in range(3):
        val = rho_grid[i, j]
        if not np.isnan(val):
            marker = '✓' if pass_grid[i, j] else '✗'
            color = EF['green'] if pass_grid[i, j] else EF['red']
            ax.text(j, i, f'ρ={val:+.3f}\n{marker}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)
ax.set_xticks(range(3)); ax.set_xticklabels(metric_labels, fontsize=9)
ax.set_yticks(range(3)); ax.set_yticklabels(summary_labels, fontsize=9)
ax.set_title('Bridge Correlations: FTLE Summaries × Collapse Metrics', color=EF['green'], fontsize=12, pad=12)
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Spearman ρ', color=EF['text'])
cbar.ax.yaxis.set_tick_params(color=EF['muted'])
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=EF['muted'])
fig.text(0.5, -0.02, '✓ = prereg pass (|ρ| ≥ 0.40, p_holm < 0.05)  ·  κ=0.566 caveat applies',
         ha='center', fontsize=8, color=EF['muted'])
save(fig, 'figure_1_bridge_heatmap', outdir)
plt.close()

# ===================== FIGURE 2: Scatter plots for passing tests =====================
print('Figure 2: scatter plots (passing tests)')
pass_tests = [(4, 'layerwise_variance_mean', 'collapse_rate', 'λ₁ variance (layerwise)', 'Collapse rate'),
              (5, 'profile_slope_mean', 'collapse_rate', 'λ₁ slope (depth profile)', 'Collapse rate'),
              (6, 'profile_slope_mean', 'first_collapse_turn', 'λ₁ slope (depth profile)', 'First collapse turn')]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for idx, (tid, xcol, ycol, xlab, ylab) in enumerate(pass_tests):
    ax = axes[idx]
    t = tests[tid - 1]
    # Censor first_collapse_turn for non-collapsing
    y_data = merged[ycol].copy() if ycol != 'first_collapse_turn' else merged['first_collapse_turn'].fillna(40)
    if ycol == 'first_collapse_turn':
        y_data = merged['first_collapse_turn'].copy()
        y_data = y_data.fillna(40)

    for cond in ['HOMO_A', 'HOMO_B', 'HOMO_C', 'HETERO_ROT']:
        mask = merged['condition'] == cond
        ax.scatter(merged.loc[mask, xcol], y_data[mask], c=CONDITION_COLORS[cond],
                   alpha=0.5, s=12, label=CONDITION_LABELS[cond], edgecolors='none')
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(f'Test #{tid}: ρ={t["rho"]:+.3f}', color=EF['green'], fontsize=10)
    ax.text(0.97, 0.03, f'p={t["p_holm"]:.1e}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color=EF['muted'])
    ax.grid(True, alpha=0.3)
axes[0].legend(fontsize=7, loc='upper left', framealpha=0.8)
fig.suptitle('Passing Pre-Registered Tests (n=720, predictive association only)',
             color=EF['green'], fontsize=12, y=1.02)
fig.tight_layout()
save(fig, 'figure_2_passing_scatters', outdir)
plt.close()

# ===================== FIGURE 3: Layerwise λ₁ profiles =====================
print('Figure 3: layerwise profiles')
# Aggregate profiles by condition (mean across tangent seeds and trajectories)
profiles = {}
for r in raw:
    cond = r['condition']
    if cond not in profiles:
        profiles[cond] = []
    profiles[cond].append(r['layerwise_profile'])

fig, ax = plt.subplots(figsize=(8, 5))
for cond in ['HOMO_A', 'HOMO_B', 'HOMO_C', 'HETERO_ROT']:
    arr = np.array(profiles[cond])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    layers = np.arange(1, len(mean) + 1)
    ax.plot(layers, mean, color=CONDITION_COLORS[cond], label=CONDITION_LABELS[cond], linewidth=2)
    ax.fill_between(layers, mean - std, mean + std, color=CONDITION_COLORS[cond], alpha=0.15)

ax.set_xlabel('Layer index', fontsize=10)
ax.set_ylabel('Layerwise λ₁ (log growth rate)', fontsize=10)
ax.set_title('λ₁ Depth Profiles by Condition (mean ± 1σ)', color=EF['green'], fontsize=12)
ax.legend(fontsize=9, framealpha=0.8)
ax.grid(True, alpha=0.3)
fig.text(0.5, -0.02, 'Computed on first assistant turn · float32 · cadence=4 · 10 tangent seeds per trajectory',
         ha='center', fontsize=8, color=EF['muted'])
save(fig, 'figure_3_depth_profiles', outdir)
plt.close()

print('All figures generated.')
