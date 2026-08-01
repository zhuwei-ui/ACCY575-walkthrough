# %% [markdown]
# # PCard transaction analysis
#
# Source: `data/transactions_wcategory.csv` — 214,698 dated transactions
# built from `PCard_Data File 1.xlsx`.

# %%
import sys
from pathlib import Path

import pandas as pd

# make `src` importable whether run as a script or interactively
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

from src.analysis import compute_monthly_totals, monthly_variance

pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 40)

# %%
df = pd.read_csv(ROOT / "data" / "transactions_wcategory.csv", parse_dates=["date"])
print(f"{len(df):,} rows x {df.shape[1]} columns")
print(df.dtypes)

# %% [markdown]
# ## Overview

# %%
print(f"date range     : {df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}")
print(f"total spend    : ${df['amount'].sum():,.2f}")
print(f"mean / median  : ${df['amount'].mean():,.2f} / ${df['amount'].median():,.2f}")
print(f"distinct vendor: {df['vendor'].nunique():,}")
print(f"distinct catgry: {df['category'].nunique():,}")
print(f"\nmissing values:\n{df.isna().sum().to_string()}")

# %% [markdown]
# ## Monthly spend and volatility
#
# Uses the helpers in `src/analysis.py`.

# %%
totals = compute_monthly_totals(df)
variance = monthly_variance(df)

monthly = totals.merge(variance, on="month", suffixes=("_total", "_variance"))
monthly["std_dev"] = monthly["amount_variance"] ** 0.5
monthly["n"] = df.groupby(df["date"].dt.to_period("M").astype(str)).size().values

print(monthly.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

# %% [markdown]
# ## Where the money goes

# %%
by_category = (
    df.groupby("category")["amount"]
    .agg(total="sum", n="count", mean="mean")
    .sort_values("total", ascending=False)
)
print("--- top 15 categories by spend ---")
print(by_category.head(15).to_string(float_format=lambda v: f"{v:,.2f}"))

print(f"\ntop 15 account for {by_category.head(15)['total'].sum() / by_category['total'].sum():.1%} of spend")

# %%
by_vendor = (
    df.groupby("vendor")["amount"]
    .agg(total="sum", n="count")
    .sort_values("total", ascending=False)
)
print("--- top 15 vendors by spend ---")
print(by_vendor.head(15).to_string(float_format=lambda v: f"{v:,.2f}"))

# %% [markdown]
# ## Refunds and credits
#
# Negative amounts are refunds — kept deliberately, so any "total spend"
# figure above is net of them.

# %%
refunds = df[df["amount"] < 0]
charges = df[df["amount"] > 0]

print(f"charges : {len(charges):>7,} rows   ${charges['amount'].sum():>16,.2f}")
print(f"refunds : {len(refunds):>7,} rows   ${refunds['amount'].sum():>16,.2f}")
print(f"net     : {len(df):>7,} rows   ${df['amount'].sum():>16,.2f}")
print(f"\nrefunds offset {abs(refunds['amount'].sum()) / charges['amount'].sum():.2%} of gross charges")

print("\n--- categories with the most refund activity ---")
print(
    refunds.groupby("category")["amount"]
    .agg(refunded="sum", n="count")
    .sort_values("refunded")
    .head(10)
    .to_string(float_format=lambda v: f"{v:,.2f}")
)

# %% [markdown]
# ## Outliers
#
# Large single transactions dominate the monthly variance figures above.

# %%
print("--- 10 largest single charges ---")
print(df.nlargest(10, "amount").to_string(index=False))

print("\n--- 10 largest single refunds ---")
print(df.nsmallest(10, "amount").to_string(index=False))
