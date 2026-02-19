#!/usr/bin/env python3
# ABOUTME: Generate Paper A figures with sohail_research theme.
# ABOUTME: Reads frozen analysis bundle, produces themed figures matching activation-steering style.
"""Generate Paper A figures with sohail_research theme.

Reads from the frozen analysis bundle and produces themed figures
matching the Inverse Scaling in Activation Steering style.

Usage:
    python3 scripts/generate_figures_themed.py
"""
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from theme import CONDITION_COLORS, COLORS, apply_theme, save_figure

# Apply theme globally
apply_theme()

# Load data
BUNDLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'analysis_bundle.csv')


def load_data():
    df = pd.read_csv(BUNDLE_PATH)
    return df


def figure_1_mean_collapse_by_condition(df):
    """Figure 1: Mean collapse rate by condition (bar chart)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    conditions = ['HOMO_A', 'HOMO_B', 'HOMO_C', 'HETERO_ROT']
    labels = ['HOMO_A\n(Llama-3.1-8B)', 'HOMO_B\n(Qwen2.5-7B)',
              'HOMO_C\n(Mistral-7B)', 'HETERO_ROT\n(Rotation)']
    means = [df[df['condition'] == c]['collapse_rate'].mean() for c in conditions]
    sds = [df[df['condition'] == c]['collapse_rate'].std() for c in conditions]
    colors = [CONDITION_COLORS[c] for c in conditions]

    bars = ax.bar(range(len(conditions)), means, yerr=sds, capsize=4,
                  color=colors, edgecolor='white', linewidth=0.5, alpha=0.9)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean Collapse Rate')
    ax.set_title('Collapse Rate by Condition (N=180 each)')
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for bar, mean, sd in zip(bars, means, sds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + sd + 0.03,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight=500)

    save_figure(fig, 'figure_1_mean_collapse_by_condition')
    plt.close()


def figure_2_collapse_distribution(df):
    """Figure 2: Collapse rate distribution by condition (box + strip)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    conditions = ['HOMO_A', 'HOMO_B', 'HOMO_C', 'HETERO_ROT']
    labels = ['HOMO_A\n(Llama-3.1-8B)', 'HOMO_B\n(Qwen2.5-7B)',
              'HOMO_C\n(Mistral-7B)', 'HETERO_ROT\n(Rotation)']

    data = [df[df['condition'] == c]['collapse_rate'].values for c in conditions]
    colors = [CONDITION_COLORS[c] for c in conditions]

    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.5,
                    medianprops=dict(color=COLORS['charcoal'], linewidth=1.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Jittered strip plot overlay
    for i, (d, color) in enumerate(zip(data, colors)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(d))
        ax.scatter(np.full_like(d, i + 1) + jitter, d,
                   c=color, alpha=0.15, s=8, edgecolors='none', zorder=2)

    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Collapse Rate')
    ax.set_title('Collapse Rate Distribution by Condition')
    ax.set_ylim(-0.05, 1.05)

    save_figure(fig, 'figure_2_collapse_distribution')
    plt.close()


def figure_3_collapse_timeline_heatmap(df):
    """Figure 3: When does collapse begin? First-collapse-turn distribution."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    conditions = ['HOMO_A', 'HOMO_B', 'HOMO_C', 'HETERO_ROT']
    labels = ['HOMO_A', 'HOMO_B', 'HOMO_C', 'HETERO_ROT']
    colors = [CONDITION_COLORS[c] for c in conditions]

    if 'first_collapse_turn' in df.columns:
        for cond, color, label in zip(conditions, colors, labels):
            subset = df[(df['condition'] == cond) & (df['first_collapse_turn'].notna())]
            if len(subset) > 0:
                vals = subset['first_collapse_turn'].values
                ax.hist(vals, bins=range(0, 42, 2), alpha=0.5, color=color,
                        label=f'{label} (n={len(subset)})', edgecolor='white',
                        linewidth=0.3)

        ax.set_xlabel('First Collapse Turn')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of First Collapse Turn by Condition')
        ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=8)
        save_figure(fig, 'figure_3_first_collapse_turn')
    else:
        ax.text(0.5, 0.5, 'first_collapse_turn column not available\nin analysis bundle',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        save_figure(fig, 'figure_3_first_collapse_turn_PLACEHOLDER')

    plt.close()


if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    print("Generating Figure 1: Mean collapse by condition...")
    figure_1_mean_collapse_by_condition(df)

    print("Generating Figure 2: Collapse distribution...")
    figure_2_collapse_distribution(df)

    print("Generating Figure 3: First collapse turn...")
    figure_3_collapse_timeline_heatmap(df)

    print("Done. Figures saved to figures/")
