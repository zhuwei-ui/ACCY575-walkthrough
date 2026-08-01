import pandas as pd
import pytest

from src.load import (
    DATE_COLUMNS,
    DTYPES,
    SENTINEL,
    YES_NO_COLUMNS,
    clean_general,
    load_general,
    parse_date_columns,
    sentinels_to_na,
    true_false_to_boolean,
    uninformative_columns,
    yes_no_to_boolean,
)


# --- yes/no conversion -------------------------------------------------------


def test_yes_no_maps_to_booleans():
    result = yes_no_to_boolean(pd.Series(["Yes", "No", "Yes"]))
    assert list(result) == [True, False, True]


def test_yes_no_uses_nullable_boolean_dtype():
    """A plain bool dtype would coerce NA to False and lose the distinction."""
    result = yes_no_to_boolean(pd.Series(["Yes", SENTINEL]))
    assert result.dtype == "boolean"


def test_sentinel_becomes_na_not_false():
    result = yes_no_to_boolean(pd.Series(["Yes", SENTINEL, None]))
    assert result[0] is True or result[0] == True  # noqa: E712
    assert pd.isna(result[1])
    assert pd.isna(result[2])


def test_true_false_accepts_both_spellings():
    """The dictionary documents True/False; the file holds t/f."""
    result = true_false_to_boolean(pd.Series(["t", "f", "True", "False"]))
    assert list(result) == [True, False, True, False]


def test_true_false_rejects_unknown_spelling_as_na():
    result = true_false_to_boolean(pd.Series(["yes"]))
    assert pd.isna(result[0])


# --- sentinels ---------------------------------------------------------------


def test_sentinels_to_na_replaces_across_string_columns():
    df = pd.DataFrame(
        {
            "auditee_uei": [SENTINEL, "REALUEI12345"],
            "auditee_email": [SENTINEL, "a@example.org"],
            "amount": [1.0, 2.0],
        }
    )
    result = sentinels_to_na(df)

    assert pd.isna(result.loc[0, "auditee_uei"])
    assert result.loc[1, "auditee_uei"] == "REALUEI12345"
    assert pd.isna(result.loc[0, "auditee_email"])


def test_sentinels_to_na_leaves_input_unchanged():
    df = pd.DataFrame({"auditee_uei": [SENTINEL]})
    sentinels_to_na(df)
    assert df.loc[0, "auditee_uei"] == SENTINEL


# --- dates -------------------------------------------------------------------


def test_parse_date_columns_converts_to_datetime():
    df = pd.DataFrame({"fy_start_date": ["2016-07-01"], "fy_end_date": ["2017-06-30"]})
    result = parse_date_columns(df)

    assert result["fy_start_date"].dtype.kind == "M"
    assert result.loc[0, "fy_start_date"] == pd.Timestamp("2016-07-01")


def test_parse_date_columns_coerces_junk_to_nat():
    df = pd.DataFrame({"fy_start_date": ["not-a-date"]})
    result = parse_date_columns(df)
    assert pd.isna(result.loc[0, "fy_start_date"])


def test_parse_date_columns_ignores_absent_columns():
    df = pd.DataFrame({"fy_start_date": ["2016-07-01"]})
    result = parse_date_columns(df)  # other DATE_COLUMNS are missing
    assert list(result.columns) == ["fy_start_date"]


# --- uninformative columns ---------------------------------------------------


def test_uninformative_columns_finds_constant_and_empty():
    df = pd.DataFrame(
        {
            "constant": ["USA", "USA", "USA"],
            "all_null": [None, None, None],
            "varies": [1, 2, 3],
        }
    )
    assert set(uninformative_columns(df)) == {"constant", "all_null"}


def test_a_column_with_two_values_is_informative():
    df = pd.DataFrame({"two": ["Yes", "No"]})
    assert uninformative_columns(df) == []


# --- clean_general -----------------------------------------------------------


def make_raw():
    return pd.DataFrame(
        {
            "is_low_risk_auditee": ["Yes", "No", SENTINEL],
            "is_public": ["t", "f", "t"],
            # deliberately varied: a constant column would be dropped as
            # uninformative before the dtype assertions could run
            "fy_start_date": ["2016-07-01", "2016-08-01", "2016-09-01"],
            "auditor_country": ["USA", "USA", "USA"],
            "auditee_uei": [SENTINEL, SENTINEL, "REALUEI12345"],
            "total_amount_expended": [1.0, 2.0, 3.0],
        }
    )


def test_clean_general_returns_frame_and_dropped_list():
    result, dropped = clean_general(make_raw())

    assert isinstance(result, pd.DataFrame)
    assert isinstance(dropped, list)
    assert "auditor_country" in dropped
    assert "auditor_country" not in result.columns


def test_clean_general_converts_booleans_and_dates():
    result, _ = clean_general(make_raw())

    assert result["is_low_risk_auditee"].dtype == "boolean"
    assert result["is_public"].dtype == "boolean"
    assert result["fy_start_date"].dtype.kind == "M"


def test_clean_general_steps_are_opt_out():
    result, dropped = clean_general(
        make_raw(),
        parse_dates=False,
        convert_booleans=False,
        sentinels_as_na=False,
        drop_uninformative=False,
    )

    assert dropped == []
    assert "auditor_country" in result.columns
    assert result.loc[2, "is_low_risk_auditee"] == SENTINEL
    assert result["fy_start_date"].dtype != "datetime64[ns]"


def test_dropping_is_reported_not_silent():
    """A caller must be able to record which columns left the dataset."""
    _, dropped = clean_general(make_raw())
    assert dropped, "expected at least one uninformative column to be reported"


def test_drop_uninformative_is_sample_dependent():
    """A meaningful column that happens to be constant in a slice gets dropped.

    `uninformative_columns` judges by distinct count, so on a subset it can
    discard a column that varies in the full extract. Pass
    `drop_uninformative=False` when working with a sample.
    """
    raw = make_raw()
    raw["fy_start_date"] = "2016-07-01"  # constant within this slice only

    _, dropped = clean_general(raw)
    assert "fy_start_date" in dropped

    kept, _ = clean_general(raw, drop_uninformative=False)
    assert "fy_start_date" in kept.columns


def test_sentinel_column_becomes_droppable_once_nulled():
    """auditee_uei is all-sentinel here, so nulling it makes it uninformative."""
    raw = make_raw()
    raw["auditee_uei"] = SENTINEL
    _, dropped = clean_general(raw)
    assert "auditee_uei" in dropped


# --- constants ---------------------------------------------------------------


def test_dtypes_cover_the_identifier_columns():
    """These are TEXT per the dictionary; without the map pandas infers ints."""
    for col in ("auditee_zip", "auditor_zip", "auditee_ein", "cognizant_agency"):
        assert DTYPES[col] is str


def test_date_and_boolean_column_lists_are_disjoint():
    assert not set(DATE_COLUMNS) & set(YES_NO_COLUMNS)


def test_load_general_requires_an_existing_file():
    with pytest.raises(FileNotFoundError):
        load_general("data/does_not_exist.csv")
