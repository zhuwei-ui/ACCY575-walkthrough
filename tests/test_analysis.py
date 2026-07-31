import pandas as pd
from src.analysis import summary_stats   # written in Module 3; moved to src/analysis.py in Module 4

def test_summary_stats_returns_a_row_per_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = summary_stats(df)
    assert len(result) == 2