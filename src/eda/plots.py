from __future__ import annotations

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_NON_TRAINING_COLS = {"tau_n", "eventOrigin", "weight", "sample_type"}

_DEFAULT_SAMPLE_ORDER = ["topquarks", "wtaunu", "ztautau", "diboson", "other", "signal"]


def _sort_samples(
    groups: list[str],
    order: list[str] | None = None,
) -> list[str]:
    """Sort sample names by a custom order, with unrecognized names appended alphabetically."""
    if order is None:
        order = _DEFAULT_SAMPLE_ORDER
    order_map = {name: i for i, name in enumerate(order)}
    known = sorted((g for g in groups if g in order_map), key=lambda g: order_map[g])
    unknown = sorted(g for g in groups if g not in order_map)
    return known + unknown


def plot_sample_balance(
    df: pd.DataFrame,
    sample_col: str = "sample_type",
    sample_order: list[str] | None = None,
    display_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Plot unweighted and weighted event counts per sample type as bar charts."""
    ordered = _sort_samples(df[sample_col].unique().tolist(), sample_order)
    counts = df[sample_col].value_counts().reindex(ordered)
    tick_labels = [
        display_labels.get(name, name) if display_labels else name for name in ordered
    ]
    n = len(counts)

    has_weights = "weight" in df.columns
    n_panels = 2 if has_weights else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(12 * n_panels, 6))
    axes = np.array(axes).reshape(-1)

    axes[0].bar(range(n), counts.values, width=0.5)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(tick_labels, fontsize=14)
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Event count")
    axes[0].set_title("Unweighted event counts")
    axes[0].ticklabel_format(axis="y", style="plain")
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=axes[0])

    if has_weights:
        weighted = df.groupby(sample_col)["weight"].sum().reindex(ordered)
        axes[1].bar(range(n), weighted.values, width=0.5)
        axes[1].set_xticks(range(n))
        axes[1].set_xticklabels(tick_labels, fontsize=14)
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel("Weighted event count")
        axes[1].set_title("Weighted event counts")
        axes[1].ticklabel_format(axis="y", style="plain")
        ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=axes[1])

    plt.subplots_adjust(wspace=0.2)
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    features: list[str] | None = None,
    exclude: list[str] | None = None,
) -> plt.Figure:
    """Plot a Pearson correlation heatmap for numeric features."""
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
    group_col: str = "class",
    class_labels: list[str] | None = None,
    n_cols: int = 3,
    n_bins: int = 50,
) -> plt.Figure:
    """Plot per-class normalized histograms for each feature in a grid."""
    class_order = sorted(df[group_col].unique())
    if class_labels is None:
        class_labels = [str(c) for c in class_order]

    colors = [plt.cm.tab10(i % 10) for i in range(len(class_order))]
    n = len(features)
    n_rows = max(1, (n + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    axes_flat = np.array(axes).reshape(-1)

    for ax, feature in zip(axes_flat, features):
        for cls, name, color in zip(class_order, class_labels, colors):
            values = df.loc[df[group_col] == cls, feature].dropna()
            ax.hist(
                values,
                bins=n_bins,
                density=True,
                alpha=0.6,
                color=color,
                label=name,
                histtype="stepfilled",
            )
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()
        ampl.draw_atlas_label(0.05, 0.95, simulation=True, status="final", ax=ax)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig
