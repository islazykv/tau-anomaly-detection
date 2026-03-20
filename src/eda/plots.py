from __future__ import annotations

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_NON_TRAINING_COLS = {"tau_n", "eventOrigin", "weight", "sample_type"}


def plot_sample_balance(
    df: pd.DataFrame,
    sample_col: str = "eventOrigin",
) -> plt.Figure:
    """Plot unweighted and weighted event counts per sample type."""
    counts = df[sample_col].value_counts().sort_index()
    n = len(counts)

    has_weights = "weight" in df.columns
    n_panels = 2 if has_weights else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(12 * n_panels, 6))
    axes = np.array(axes).reshape(-1)

    axes[0].bar(range(n), counts.values, width=0.5)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(counts.index, fontsize=10, rotation=45, ha="right")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Event count")
    axes[0].set_title("Unweighted event counts")
    axes[0].ticklabel_format(axis="y", style="plain")
    ampl.draw_atlas_label(0.05, 0.97, ax=axes[0])

    if has_weights:
        weighted = df.groupby(sample_col)["weight"].sum().sort_index()
        axes[1].bar(range(n), weighted.values, width=0.5)
        axes[1].set_xticks(range(n))
        axes[1].set_xticklabels(weighted.index, fontsize=10, rotation=45, ha="right")
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel("Weighted event count")
        axes[1].set_title("Weighted event counts")
        axes[1].ticklabel_format(axis="y", style="plain")
        ampl.draw_atlas_label(0.05, 0.97, ax=axes[1])

    plt.subplots_adjust(wspace=0.2)
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    features: list[str] | None = None,
    exclude: list[str] | None = None,
) -> plt.Figure:
    """Plot a Pearson correlation heatmap for numeric training features."""
    excluded = _NON_TRAINING_COLS | set(exclude or [])
    if features is None:
        features = [
            c for c in df.select_dtypes(include="number").columns if c not in excluded
        ]

    corr = df[features].corr()
    n = len(features)
    annotate = n <= 30
    size = max(8, n * 0.4)

    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=annotate,
        fmt=".2f" if annotate else "",
        linewidths=0.3 if annotate else 0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    features: list[str],
    group_col: str = "eventOrigin",
    n_cols: int = 3,
    n_bins: int = 50,
) -> plt.Figure:
    """Plot per-sample normalized histograms for each feature in a grid layout."""
    groups = sorted(df[group_col].unique())
    colors = [plt.cm.tab10(i % 10) for i in range(len(groups))]
    n = len(features)
    n_rows = max(1, (n + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    axes_flat = np.array(axes).reshape(-1)

    for ax, feature in zip(axes_flat, features):
        for group, color in zip(groups, colors):
            values = df.loc[df[group_col] == group, feature].dropna()
            ax.hist(
                values,
                bins=n_bins,
                density=True,
                alpha=0.6,
                color=color,
                label=group,
                histtype="stepfilled",
            )
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Feature distributions by sample")
    fig.tight_layout()
    return fig
