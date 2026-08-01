import pandas as pd
import pytest

from src.analysis import monthly_totals_by_category


def make_df(rows):
    """rows: list of (date, amount, category)."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime([r[0] for r in rows]),
            "amount": [r[1] for r in rows],
            "category": [r[2] for r in rows],
        }
    )


def test_handles_missing_category_in_a_month():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-15", "2026-02-10"]),
        "amount": [100.0, 200.0],
        "vendor": ["a", "b"],
        "category": ["food", "travel"],
    })
    result = monthly_totals_by_category(df)
    # Both months and both categories should appear.
    # Missing combos should be 0, not NaN.
    assert result.loc["2026-01-31", "food"] == 100.0
    assert result.loc["2026-01-31", "travel"] == 0.0
    assert result.loc["2026-02-28", "food"] == 0.0
    assert result.loc["2026-02-28", "travel"] == 200.0


def test_returns_one_column_per_category():
    df = make_df(
        [
            ("2024-01-10", 10.0, "books"),
            ("2024-01-20", 5.0, "food"),
            ("2024-01-25", 2.0, "travel"),
        ]
    )
    result = monthly_totals_by_category(df)

    assert sorted(result.columns) == ["books", "food", "travel"]
    assert len(result) == 1


def test_amounts_are_summed_per_month_and_category():
    df = make_df(
        [
            ("2024-01-10", 10.0, "books"),
            ("2024-01-20", 5.0, "books"),
            ("2024-02-03", 7.0, "books"),
            ("2024-02-14", 1.5, "food"),
        ]
    )
    result = monthly_totals_by_category(df)

    jan, feb = result.index
    assert result.loc[jan, "books"] == pytest.approx(15.0)
    assert result.loc[feb, "books"] == pytest.approx(7.0)
    assert result.loc[feb, "food"] == pytest.approx(1.5)


def test_index_is_month_end_timestamps():
    df = make_df([("2024-01-10", 10.0, "books"), ("2024-02-03", 7.0, "books")])
    result = monthly_totals_by_category(df)

    assert list(result.index) == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),  # 2024 is a leap year
    ]


def test_axis_names():
    df = make_df([("2024-01-10", 10.0, "books")])
    result = monthly_totals_by_category(df)

    assert result.index.name == "month"
    assert result.columns.name is None


def test_months_with_no_activity_at_all_are_omitted():
    """The docstring claims the index spans every month between the first and
    last transaction. It does not: a month with zero rows forms no group, so it
    is absent from the result rather than present as a row of zeros.
    """
    df = make_df(
        [
            ("2024-01-10", 10.0, "books"),
            ("2024-03-05", 7.0, "books"),  # nothing at all in February
        ]
    )
    result = monthly_totals_by_category(df)

    assert list(result.index) == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-03-31"),
    ]
    assert pd.Timestamp("2024-02-29") not in result.index


def test_negative_amounts_net_against_positives():
    df = make_df(
        [
            ("2024-01-10", 100.0, "books"),
            ("2024-01-11", -30.0, "books"),  # refund
        ]
    )
    result = monthly_totals_by_category(df)

    assert result.loc[result.index[0], "books"] == pytest.approx(70.0)


def test_zero_net_is_indistinguishable_from_no_activity():
    """`fillna(0.0)` means a 0.0 cell reads the same whether the category had no
    transactions or had transactions netting exactly zero. Callers cannot tell
    the two apart.
    """
    netted = make_df(
        [
            ("2024-01-10", 50.0, "books"),
            ("2024-01-11", -50.0, "books"),
        ]
    )
    absent = make_df(
        [
            ("2024-01-10", 1.0, "food"),
            ("2024-01-11", 0.0, "books"),
        ]
    )

    assert monthly_totals_by_category(netted)["books"].iloc[0] == 0.0
    assert monthly_totals_by_category(absent)["books"].iloc[0] == 0.0


def test_result_is_all_float():
    df = make_df([("2024-01-10", 10.0, "books"), ("2024-02-03", 7.0, "food")])
    result = monthly_totals_by_category(df)

    assert (result.dtypes == "float64").all()