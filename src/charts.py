"""Charts over the firm-year aggregation.

One figure, one panel per audit year, bars ranked largest-to-smallest within
each panel.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

# Single measure per panel, so one hue throughout: colour identifies the
# measure, never the rank of a bar.
PALETTE = {
    "surface": "#fcfcfb",
    "series": "#2a78d6",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
}

FIRM_COLUMN = "auditor_firm_name"
YEAR_COLUMN = "audit_year"


def top_n_per_year(
    df: pd.DataFrame,
    *,
    value_column: str,
    top_n: int = 15,
    firm_column: str = FIRM_COLUMN,
    year_column: str = YEAR_COLUMN,
) -> pd.DataFrame:
    """The `top_n` highest-scoring firms within each year, ranked descending.

    Ranking is per year, so the firms shown in one panel need not be the firms
    shown in another. That answers "who was largest in this year"; it does not
    let you trace one firm across panels.
    """
    for col in (firm_column, year_column, value_column):
        if col not in df.columns:
            raise KeyError(f"column absent from frame: {col}")
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}")

    return (
        df.sort_values([year_column, value_column], ascending=[True, False])
        .groupby(year_column, group_keys=False, dropna=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _truncate(label: str, limit: int) -> str:
    text = str(label)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def plot_firms_by_year(
    df: pd.DataFrame,
    *,
    value_column: str = "n_auditees",
    top_n: int = 15,
    firm_column: str = FIRM_COLUMN,
    year_column: str = YEAR_COLUMN,
    share_y: bool = True,
    label_limit: int = 28,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Draw one bar panel per audit year, ranked largest to smallest.

    Parameters
    ----------
    share_y
        Panels share a y-scale so bar heights are comparable across years.
        Set False to let each year use its own scale.
    output_path
        If given, the figure is written there as well as returned.

    Returns
    -------
    The figure, so a caller can adjust it or save it elsewhere.
    """
    ranked = top_n_per_year(
        df,
        value_column=value_column,
        top_n=top_n,
        firm_column=firm_column,
        year_column=year_column,
    )

    years = sorted(ranked[year_column].dropna().unique())
    if not years:
        raise ValueError("no rows to plot")

    fig, axes = plt.subplots(
        nrows=len(years),
        ncols=1,
        figsize=(11, 3.6 * len(years)),
        sharey=share_y,
        constrained_layout=True,
    )
    # subplots returns a bare Axes when nrows == 1
    axes = axes if len(years) > 1 else [axes]

    fig.patch.set_facecolor(PALETTE["surface"])

    for ax, year in zip(axes, years):
        panel = ranked[ranked[year_column] == year]
        labels = [_truncate(name, label_limit) for name in panel[firm_column]]

        ax.bar(
            range(len(panel)),
            panel[value_column],
            width=0.8,  # leaves a gap between adjacent bars
            color=PALETTE["series"],
        )

        ax.set_xticks(range(len(panel)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{year}", loc="left", fontsize=11, color=PALETTE["ink"])
        ax.set_ylabel(value_column, fontsize=9, color=PALETTE["ink_secondary"])

        ax.set_facecolor(PALETTE["surface"])
        ax.grid(axis="y", color=PALETTE["gridline"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(PALETTE["baseline"])
        ax.tick_params(colors=PALETTE["muted"], length=0)

    heading = title or f"Top {top_n} audit firms by {value_column}, per audit year"
    fig.suptitle(heading, x=0.01, ha="left", fontsize=13, color=PALETTE["ink"])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())

    return fig
