import pandas as pd
import pytest

from src.aggregate import (
    GROUP_COLUMNS,
    auditees_per_firm_year,
    summarise_by_firm_year,
)


def make_df():
    """Two firms across two years, with a repeated auditee and a null."""
    return pd.DataFrame(
        {
            "auditor_firm_name": ["A LLP", "A LLP", "A LLP", "B PC", "B PC"],
            "audit_year": [2016, 2016, 2017, 2016, 2016],
            "auditee_ein": ["111", "222", "111", "333", "333"],
            "auditee_uei": ["GSA_MIGRATION"] * 5,
            "entity_type": ["local", "state", "local", "tribal", None],
        }
    )


# --- summarise_by_firm_year --------------------------------------------------


def test_one_row_per_firm_year():
    result = summarise_by_firm_year(make_df())

    assert len(result) == 3  # (A,2016), (A,2017), (B,2016)
    assert list(result.columns[:3]) == ["auditor_firm_name", "audit_year", "n_rows"]


def test_n_rows_is_the_group_size():
    result = summarise_by_firm_year(make_df()).set_index(
        ["auditor_firm_name", "audit_year"]
    )

    assert result.loc[("A LLP", 2016), "n_rows"] == 2
    assert result.loc[("A LLP", 2017), "n_rows"] == 1
    assert result.loc[("B PC", 2016), "n_rows"] == 2


def test_nunique_counts_distinct_values_per_column():
    result = summarise_by_firm_year(make_df()).set_index(
        ["auditor_firm_name", "audit_year"]
    )

    # A LLP in 2016 saw two different auditees
    assert result.loc[("A LLP", 2016), "auditee_ein"] == 2
    # B PC saw the same auditee twice
    assert result.loc[("B PC", 2016), "auditee_ein"] == 1


def test_nunique_ignores_nulls():
    """B PC 2016 has entity_type ['tribal', None] -> one distinct value."""
    result = summarise_by_firm_year(make_df()).set_index(
        ["auditor_firm_name", "audit_year"]
    )
    assert result.loc[("B PC", 2016), "entity_type"] == 1


def test_count_counts_non_null_values():
    result = summarise_by_firm_year(make_df(), how="count").set_index(
        ["auditor_firm_name", "audit_year"]
    )

    assert result.loc[("B PC", 2016), "n_rows"] == 2
    assert result.loc[("B PC", 2016), "entity_type"] == 1  # one is null
    assert result.loc[("A LLP", 2016), "entity_type"] == 2


def test_count_and_nunique_differ_on_repeats():
    """The two modes answer different questions: 'how many different' vs 'how often'."""
    df = make_df()
    nunique = summarise_by_firm_year(df).set_index(["auditor_firm_name", "audit_year"])
    count = summarise_by_firm_year(df, how="count").set_index(
        ["auditor_firm_name", "audit_year"]
    )

    assert nunique.loc[("B PC", 2016), "auditee_ein"] == 1
    assert count.loc[("B PC", 2016), "auditee_ein"] == 2


def test_every_non_group_column_is_summarised():
    df = make_df()
    result = summarise_by_firm_year(df)

    expected = set(df.columns) - set(GROUP_COLUMNS) | {"n_rows", *GROUP_COLUMNS}
    assert set(result.columns) == expected


def test_rows_with_a_null_firm_name_are_kept():
    """dropna=False -- a missing firm name must not silently drop submissions."""
    df = make_df()
    df.loc[0, "auditor_firm_name"] = None

    result = summarise_by_firm_year(df)
    assert result["n_rows"].sum() == len(df)


def test_invalid_how_raises():
    with pytest.raises(ValueError, match="how must be one of"):
        summarise_by_firm_year(make_df(), how="average")


def test_missing_group_column_raises():
    df = make_df().drop(columns=["audit_year"])
    with pytest.raises(KeyError, match="audit_year"):
        summarise_by_firm_year(df)


# --- auditees_per_firm_year --------------------------------------------------


def test_auditees_per_firm_year_counts_distinct_auditees():
    result = auditees_per_firm_year(make_df()).set_index(
        ["auditor_firm_name", "audit_year"]
    )

    assert result.loc[("A LLP", 2016), "n_auditees"] == 2
    assert result.loc[("B PC", 2016), "n_auditees"] == 1


def test_auditees_per_firm_year_is_sorted_descending():
    result = auditees_per_firm_year(make_df())
    assert list(result["n_auditees"]) == sorted(result["n_auditees"], reverse=True)


def test_default_identifier_is_ein_not_uei():
    """auditee_uei is the GSA_MIGRATION placeholder for 99.97% of this extract."""
    by_default = auditees_per_firm_year(make_df())
    by_uei = auditees_per_firm_year(make_df(), auditee_column="auditee_uei")

    assert by_default["n_auditees"].max() == 2
    assert by_uei["n_auditees"].max() == 1  # the sentinel collapses every group


def test_unknown_auditee_column_raises():
    with pytest.raises(KeyError, match="nope"):
        auditees_per_firm_year(make_df(), auditee_column="nope")


def test_result_columns_are_named():
    result = auditees_per_firm_year(make_df())
    assert list(result.columns) == ["auditor_firm_name", "audit_year", "n_auditees"]
