import os

import pandas as pd
from dotenv import load_dotenv


def get_api_key():
    """Return OPENAI_API_KEY, loading .env first if present."""
    load_dotenv()
    return os.environ["OPENAI_API_KEY"]


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


if __name__ == "__main__":
    api_key = get_api_key()
    print(f"loaded a key of length {len(api_key)}")
