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


def monthly_totals_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Total `amount` per month-end and category, one column per category.

    Expects `date` to be datetime64. The index spans every month between the
    first and last transaction, so a category with no activity in a month is 0.
    """
    return (
        df.groupby([pd.Grouper(key="date", freq="ME"), "category"])["amount"]
          .sum()
          .unstack("category")
          .fillna(0.0)
          .rename_axis(index="month", columns=None)
    )


if __name__ == "__main__":
    api_key = get_api_key()
    print(f"loaded a key of length {len(api_key)}")
