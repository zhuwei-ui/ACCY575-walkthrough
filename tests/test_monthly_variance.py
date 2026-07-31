import pandas as pd
import pytest

from src.analysis import monthly_variance


def test_monthly_variance_matches_snapshot():
    df = pd.read_csv("tests/fixtures/sample_transactions.csv", parse_dates=["date"])
    result = monthly_variance(df)
    expected = pd.read_csv("tests/fixtures/expected_monthly_volatility.csv")
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def test_monthly_variance_computes_sample_variance():
    """Variance of [10, 20] is 50.0 with ddof=1 (pandas default)."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-10"]),
            "amount": [10.0, 20.0, 7.0],
        }
    )
    result = monthly_variance(df)

    assert list(result["month"]) == ["2024-01", "2024-02"]
    assert result.loc[0, "amount"] == pytest.approx(50.0)
    assert pd.isna(result.loc[1, "amount"]), "a single observation has no sample variance"


def test_monthly_variance_groups_by_calendar_month_not_30_days():
    """Rows 20 days apart still land in different months if they straddle a boundary."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-25", "2024-02-14", "2024-02-20"]),
            "amount": [1.0, 3.0, 9.0],
        }
    )
    result = monthly_variance(df)

    assert list(result["month"]) == ["2024-01", "2024-02"]
    assert result.loc[1, "amount"] == pytest.approx(18.0)


def test_monthly_variance_accepts_a_different_column():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-01", "2024-03-15"]),
            "amount": [1.0, 2.0],
            "units": [4.0, 10.0],
        }
    )
    result = monthly_variance(df, column="units")

    assert list(result.columns) == ["month", "units"]
    assert result.loc[0, "units"] == pytest.approx(18.0)
