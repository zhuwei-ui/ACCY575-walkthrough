"""Aggregations over the FAC `general` extract.

The unit of analysis is the auditor firm-year: one row per
(`auditor_firm_name`, `audit_year`) pair.
"""

import pandas as pd

GROUP_COLUMNS = ("auditor_firm_name", "audit_year")

HOW_CHOICES = ("nunique", "count")


def summarise_by_firm_year(
    df: pd.DataFrame,
    *,
    how: str = "nunique",
    group_columns: tuple[str, ...] = GROUP_COLUMNS,
) -> pd.DataFrame:
    """Summarise every column for each auditor firm and audit year.

    Parameters
    ----------
    how
        ``"nunique"`` counts distinct values per column -- so
        ``auditee_uei`` becomes "how many different auditees this firm served
        that year". ``"count"`` counts non-null values instead, which answers
        "how often was this column populated".

    Returns
    -------
    One row per group, with `n_rows` (the number of submissions in the group)
    followed by one column per remaining column of `df`.

    Notes
    -----
    Both aggregations ignore nulls: a column that is entirely null within a
    group yields 0, not the group size. Groups are formed with
    ``dropna=False`` so submissions with a missing firm name are still counted
    rather than silently dropped.
    """
    if how not in HOW_CHOICES:
        raise ValueError(f"how must be one of {HOW_CHOICES}, got {how!r}")

    missing = [c for c in group_columns if c not in df.columns]
    if missing:
        raise KeyError(f"group columns absent from frame: {missing}")

    grouped = df.groupby(list(group_columns), dropna=False)
    summary = grouped.nunique() if how == "nunique" else grouped.count()
    summary.insert(0, "n_rows", grouped.size())
    return summary.reset_index()


def auditees_per_firm_year(
    df: pd.DataFrame,
    *,
    auditee_column: str = "auditee_ein",
    group_columns: tuple[str, ...] = GROUP_COLUMNS,
) -> pd.DataFrame:
    """Number of distinct auditees each firm served in each year.

    `auditee_ein` is the default identifier rather than `auditee_uei`: in the
    2016-2018 extract `auditee_uei` is the ``GSA_MIGRATION`` placeholder for
    99.97% of rows, so a distinct count over it is close to meaningless. Pass
    `auditee_column="auditee_uei"` to use it anyway.
    """
    for col in (*group_columns, auditee_column):
        if col not in df.columns:
            raise KeyError(f"column absent from frame: {col}")

    return (
        df.groupby(list(group_columns), dropna=False)[auditee_column]
        .nunique()
        .rename("n_auditees")
        .reset_index()
        .sort_values(["n_auditees", *group_columns], ascending=False)
        .reset_index(drop=True)
    )
