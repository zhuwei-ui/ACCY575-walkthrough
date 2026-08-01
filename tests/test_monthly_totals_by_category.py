import pandas as pd
from src.analysis import monthly_totals_by_category

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