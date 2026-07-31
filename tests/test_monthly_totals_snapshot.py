import pandas as pd
from src.analysis import compute_monthly_totals

def test_monthly_totals_match_snapshot():
    df = pd.read_csv("tests/fixtures/sample_transactions.csv", parse_dates=["date"])
    result = compute_monthly_totals(df)
    expected = pd.read_csv("tests/fixtures/expected_monthly_totals.csv")
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )
