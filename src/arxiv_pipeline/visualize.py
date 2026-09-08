"""Stage 4 — render the four analysis charts from the aggregate tables.

All plots share one category colour map and one rcParams block, so the set
reads as a single design system. The whole theme lives in this module —
nothing else in the package draws anything.
"""
import matplotlib

matplotlib.use("Agg")  # render to file; never needs a display

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from . import config
from .db import connect

# One colour per category, so a category keeps its identity across all four
# charts. Keyed on the categories that actually occur as `primary_category`
# (which is why stat.ML appears and cs.RO/cs.NE do not — see TARGET_CATEGORIES).
CATEGORY_COLORS = {
    "cs.AI": "#4C72B0",
    "cs.LG": "#DD8452",
    "cs.CV": "#55A868",
    "cs.CL": "#C44E52",
    "stat.ML": "#8172B2",
}
FALLBACK_COLOR = "#999999"

BG = "#F8F9FA"
GRID = "#E0E0E0"
ACCENT = "#E63946"
LABEL_GREY = "#333333"


def thousands() -> ticker.Formatter:
    """A fresh thousands-separator formatter for one axis.

    Deliberately a factory rather than a shared module-level instance:
    attaching a formatter calls `set_axis`, so a single shared instance keeps
    the last axis it touched — and therefore that axis's closed Figure and
    every artist on it — reachable for the life of the process. Verified: with
    a shared instance the Figure survives `plt.close()`; with a per-axis one it
    does not. The formatter class makes no difference; the sharing does.
    """
    return ticker.StrMethodFormatter("{x:,.0f}")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def color_for(category: str) -> str:
    """The fixed colour for one category, neutral grey if it has none."""
    return CATEGORY_COLORS.get(category, FALLBACK_COLOR)


def colors_for(categories) -> list[str]:
    return [color_for(category) for category in categories]


def save(fig, filename: str) -> None:
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.PLOT_DIR / filename, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved -> {filename}")


def plot_papers_per_category(conn) -> None:
    """Volume per category as bars, with publish rate overlaid on a second axis."""
    df = pd.read_sql("SELECT * FROM category_stats ORDER BY total_papers DESC", conn)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        x, df["total_papers"], width=0.55, color=colors_for(df["category"]),
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=10, fontweight="bold")

    rate_ax = ax.twinx()
    rate_ax.plot(
        x, df["publish_rate_pct"], color=ACCENT, marker="o", markersize=7,
        linewidth=2, linestyle="--", label="Publish rate %", zorder=4,
    )
    rate_ax.set_ylabel("Publish Rate (%)", color=ACCENT)
    rate_ax.tick_params(axis="y", labelcolor=ACCENT, labelsize=10)
    rate_ax.set_ylim(0, df["publish_rate_pct"].max() * 1.6)
    rate_ax.spines["top"].set_visible(False)
    rate_ax.grid(False)
    rate_ax.legend(loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels(df["category"])
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Papers")
    ax.yaxis.set_major_formatter(thousands())
    ax.set_title("Papers per Category with Publish Rate")
    save(fig, "01_papers_per_category.png")


def plot_submission_trend(conn) -> None:
    """One line per category over time, labelled at its right-hand end."""
    df = pd.read_sql(
        "SELECT year, category, paper_count FROM yearly_trends ORDER BY year", conn
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    for category, group in df.groupby("category"):
        color = color_for(category)
        ax.plot(
            group["year"], group["paper_count"], marker="o", markersize=4,
            linewidth=2, color=color, label=category, zorder=3,
        )
        last = group.iloc[-1]
        ax.annotate(
            category, xy=(last["year"], last["paper_count"]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=9, color=color, va="center",
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(title="Category", loc="upper left", framealpha=0.85)
    ax.set_title("arXiv Submission Trend Over Time by Category")
    save(fig, "02_submission_trend_over_time.png")


def plot_publication_status(conn) -> None:
    """Preprint vs published per category, stacked, annotated with publish share."""
    df = pd.read_sql("SELECT * FROM publication_status", conn)
    pivot = df.pivot_table(
        index="category", columns="pub_status", values="paper_count",
        aggfunc="sum", fill_value=0,
    ).reindex(columns=["Preprint", "Published"], fill_value=0).reset_index()
    pivot = pivot.sort_values("Preprint", ascending=False).reset_index(drop=True)

    x = np.arange(len(pivot))
    colors = colors_for(pivot["category"])

    fig, ax = plt.subplots(figsize=(10, 6))
    # Same hue per category; opacity distinguishes the two statuses.
    ax.bar(
        x, pivot["Preprint"], width=0.55, color=colors,
        edgecolor="white", linewidth=0.8, alpha=0.75, zorder=3,
    )
    ax.bar(
        x, pivot["Published"], width=0.55, bottom=pivot["Preprint"], color=colors,
        edgecolor="white", linewidth=0.8, zorder=3,
    )

    totals = pivot["Preprint"] + pivot["Published"]
    for i, total in enumerate(totals):
        if total:
            ax.text(
                x[i], total + 12, f"{pivot['Published'][i] / total * 100:.1f}% pub",
                ha="center", va="bottom", fontsize=9, color=LABEL_GREY,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot["category"])
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Papers")
    ax.yaxis.set_major_formatter(thousands())
    # Hue carries the category, so the legend shows status as neutral swatches.
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=FALLBACK_COLOR, alpha=alpha)
            for alpha in (0.75, 1.0)
        ],
        labels=["Preprint", "Published"], framealpha=0.85, loc="upper right",
    )
    ax.set_ylim(0, totals.max() * 1.18)
    ax.set_title("Publication Status Breakdown by Category\n(Preprint vs Published)")
    save(fig, "03_publication_status_breakdown.png")


def plot_abstract_length(conn) -> None:
    """Boxplot per category with a jittered sample of the raw points behind it."""
    df = pd.read_sql(
        "SELECT primary_category AS category, abstract_word_count FROM papers "
        "WHERE abstract_word_count IS NOT NULL",
        conn,
    )
    # One grouper serves both the ordering and the per-category values.
    grouped = df.groupby("category")["abstract_word_count"]
    groups = {name: group.values for name, group in grouped}
    medians = grouped.median().sort_values(ascending=False)
    categories = list(medians.index)
    series = [groups[category] for category in categories]

    fig, ax = plt.subplots(figsize=(11, 6))
    box = ax.boxplot(
        series, patch_artist=True, widths=0.45,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=2, alpha=0.3, linestyle="none"),
    )
    for patch, color in zip(box["boxes"], colors_for(categories)):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    for i, category in enumerate(categories):
        values = series[i]
        rng = np.random.default_rng(i)
        sample = rng.choice(values, size=min(len(values), 300), replace=False)
        jitter = rng.uniform(-0.18, 0.18, size=len(sample))
        ax.scatter(
            np.full_like(sample, i + 1, dtype=float) + jitter, sample,
            alpha=0.18, s=8, color=color_for(category), zorder=2,
        )
        # Sit the label just above the white median line so neither obscures
        # the other.
        ax.annotate(
            f"{int(medians[category])}",
            xy=(i + 1, medians[category]), xytext=(0, 6),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="white",
        )

    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(categories)
    ax.set_xlabel("Category")
    ax.set_ylabel("Abstract Word Count")
    ax.set_title("Abstract Length Distribution by Category\n(median shown above box line)")
    save(fig, "04_abstract_length_distribution.png")


PLOTS = [
    plot_papers_per_category,
    plot_submission_trend,
    plot_publication_status,
    plot_abstract_length,
]


def run() -> None:
    print("Rendering charts...")
    with connect() as conn:
        for plot in PLOTS:
            plot(conn)
    print(f"\n{len(PLOTS)} plots saved to {config.PLOT_DIR}")


if __name__ == "__main__":
    run()
