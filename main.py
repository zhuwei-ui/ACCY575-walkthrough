def main():
    name = "ACCY575-walkthrough"
    greeting = f"Hello World! Good morning! Welcome to {name}!"
    print(greeting)


if __name__ == "__main__":
    main()

import pandas as pd

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean and median for each numeric column."""
    return df.select_dtypes(include="number").agg(["mean", "median"]).T
