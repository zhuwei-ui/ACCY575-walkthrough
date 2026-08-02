import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede pyplot import

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.charts import PALETTE, plot_firms_by_year, top_n_per_year


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def make_agg():
    """Four firms across three years, deliberately not in sorted order."""
    rows = []
    for year, values in {
        2016: [("A LLP", 10), ("B PC", 30), ("C LLC", 20), ("D CPA", 5)],
        2017: [("A LLP", 40), ("B PC", 15), ("C LLC", 25), ("D CPA", 35)],
        2018: [("A LLP", 7), ("B PC", 9), ("C LLC", 8), ("D CPA", 6)],
    }.items():
        for firm, value in values:
            rows.append({"auditor_firm_name": firm, "audit_year": year, "n_auditees": value})
    return pd.DataFrame(rows)


# --- top_n_per_year ----------------------------------------------------------


def test_keeps_only_top_n_within_each_year():
    result = top_n_per_year(make_agg(), value_column="n_auditees", top_n=2)

    assert len(result) == 6  # 2 per year x 3 years
    assert result.groupby("audit_year").size().tolist() == [2, 2, 2]


def test_sorted_descending_within_each_year():
    result = top_n_per_year(make_agg(), value_column="n_auditees", top_n=4)

    for _, panel in result.groupby("audit_year"):
        values = panel["n_auditees"].tolist()
        assert values == sorted(values, reverse=True)


def test_ranking_is_per_year_not_global():
    """The leader differs by year, so panels need not show the same firms."""
    result = top_n_per_year(make_agg(), value_column="n_auditees", top_n=1)
    leaders = dict(zip(result["audit_year"], result["auditor_firm_name"]))

    assert leaders[2016] == "B PC"
    assert leaders[2017] == "A LLP"
    assert leaders[2018] == "B PC"


def test_top_n_larger_than_available_keeps_everything():
    result = top_n_per_year(make_agg(), value_column="n_auditees", top_n=99)
    assert len(result) == 12


def test_missing_column_raises():
    with pytest.raises(KeyError, match="nope"):
        top_n_per_year(make_agg(), value_column="nope")


def test_top_n_below_one_raises():
    with pytest.raises(ValueError, match="at least 1"):
        top_n_per_year(make_agg(), value_column="n_auditees", top_n=0)


# --- plot_firms_by_year ------------------------------------------------------


def test_one_panel_per_year():
    fig = plot_firms_by_year(make_agg(), top_n=3)
    assert len(fig.axes) == 3


def test_a_single_year_still_produces_one_panel():
    """subplots() returns a bare Axes when nrows == 1; that must not break."""
    df = make_agg()
    fig = plot_firms_by_year(df[df["audit_year"] == 2016], top_n=3)
    assert len(fig.axes) == 1


def test_bars_descend_left_to_right_in_every_panel():
    fig = plot_firms_by_year(make_agg(), top_n=4)

    for ax in fig.axes:
        heights = [patch.get_height() for patch in ax.patches]
        assert heights == sorted(heights, reverse=True)


def test_bar_count_matches_top_n():
    fig = plot_firms_by_year(make_agg(), top_n=2)
    for ax in fig.axes:
        assert len(ax.patches) == 2


def test_every_bar_uses_the_same_colour():
    """Colour identifies the measure; it must never encode a bar's rank."""
    fig = plot_firms_by_year(make_agg(), top_n=4)
    colours = {patch.get_facecolor() for ax in fig.axes for patch in ax.patches}
    assert len(colours) == 1


def test_y_axis_starts_at_zero():
    """Bar length encodes magnitude, so a truncated baseline would mislead."""
    fig = plot_firms_by_year(make_agg(), top_n=4)
    for ax in fig.axes:
        assert ax.get_ylim()[0] == 0


def test_share_y_gives_panels_one_scale():
    fig = plot_firms_by_year(make_agg(), top_n=4, share_y=True)
    limits = {ax.get_ylim() for ax in fig.axes}
    assert len(limits) == 1


def test_independent_scales_when_share_y_is_false():
    fig = plot_firms_by_year(make_agg(), top_n=4, share_y=False)
    limits = {ax.get_ylim() for ax in fig.axes}
    assert len(limits) > 1


def test_long_labels_are_truncated():
    df = make_agg()
    df.loc[df["auditor_firm_name"] == "A LLP", "auditor_firm_name"] = "X" * 60

    fig = plot_firms_by_year(df, top_n=4, label_limit=10)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]

    assert all(len(label) <= 10 for label in labels)
    assert any(label.endswith("…") for label in labels)


def test_short_labels_are_left_alone():
    fig = plot_firms_by_year(make_agg(), top_n=4, label_limit=28)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "A LLP" in labels


def test_panels_are_titled_by_year():
    fig = plot_firms_by_year(make_agg(), top_n=2)
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert titles == ["2016", "2017", "2018"]


def test_writes_a_file_when_asked(tmp_path):
    out = tmp_path / "nested" / "figure.png"
    plot_firms_by_year(make_agg(), top_n=3, output_path=out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_no_file_written_without_output_path(tmp_path):
    plot_firms_by_year(make_agg(), top_n=3)
    assert list(tmp_path.iterdir()) == []


def test_empty_frame_raises():
    empty = make_agg().iloc[0:0]
    with pytest.raises(ValueError, match="no rows to plot"):
        plot_firms_by_year(empty)


def test_custom_title_is_used():
    fig = plot_firms_by_year(make_agg(), top_n=2, title="Custom heading")
    assert fig._suptitle.get_text() == "Custom heading"


def test_palette_has_one_series_colour():
    """A single measure per panel means one hue, so there is nothing to cycle."""
    assert PALETTE["series"] == "#2a78d6"
