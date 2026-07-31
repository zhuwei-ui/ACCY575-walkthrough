import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]
print(f"loaded a key of length {len(api_key)}")

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean and median for each numeric column."""
    return df.select_dtypes(include="number").agg(["mean", "median"]).T

def compute_monthly_totals(df):
    """Total `amount` per calendar month."""
    return (
        df.assign(month=df["date"].dt.to_period("M").astype(str))
          .groupby("month", as_index=False)["amount"].sum()
    )

def monthly_variance(df, column="amount"):
    """Sample variance of `column` per calendar month."""
    return (
        df.assign(month=df["date"].dt.to_period("M").astype(str))
          .groupby("month", as_index=False)[column].var()
    )