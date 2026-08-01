"""Schema tests.

The frames here are synthetic. Building them in code rather than reading a
fixture keeps the tests independent of `data/` (which is gitignored, so absent
on a fresh clone) and keeps real auditee contact details out of the repo.
"""

import pandas as pd
import pandera.pandas as pa
import pytest

from src.validate import GeneralSchema, describe_failures, validate


def make_valid_frame(n: int = 3) -> pd.DataFrame:
    """A minimal frame satisfying every constraint in GeneralSchema."""
    return pd.DataFrame(
        {
            "report_id": [f"2017-06-GSAFAC-{i:010d}" for i in range(1, n + 1)],
            "auditee_uei": ["GSA_MIGRATION"] * n,
            "audit_year": [2017] * n,
            "entity_type": ["non-profit"] * n,
            "audit_type": ["single-audit"] * n,
            "audit_period_covered": ["annual"] * n,
            "data_source": ["GSAFAC"] * n,
            "type_audit_code": ["UG"] * n,
            "is_public": ["t"] * n,
            "is_low_risk_auditee": ["Yes"] * n,
            "is_going_concern_included": ["No"] * n,
            "is_internal_control_deficiency_disclosed": ["No"] * n,
            "is_internal_control_material_weakness_disclosed": ["No"] * n,
            "is_material_noncompliance_disclosed": ["No"] * n,
            "is_aicpa_audit_guide_included": ["No"] * n,
            "is_multiple_eins": ["No"] * n,
            "is_secondary_auditors": ["No"] * n,
            "gaap_results": ["unmodified_opinion"] * n,
            "sp_framework_opinions": [None] * n,
            "total_amount_expended": [1_000_000.0] * n,
            "dollar_threshold": [750_000.0] * n,
            "auditee_zip": ["06268"] * n,
            "auditor_zip": ["781234567"] * n,
            "auditee_ein": ["123456789"] * n,
            "auditee_state": ["CT"] * n,
            "auditor_state": ["TX"] * n,
            "cognizant_agency": [None] * n,
            "oversight_agency": ["93"] * n,
            "sp_framework_basis": [None] * n,
            "is_sp_framework_required": [None] * n,
            "number_months": [None] * n,
            "auditor_foreign_address": [None] * n,
            "fy_start_date": ["2016-07-01"] * n,
            "fy_end_date": ["2017-06-30"] * n,
        }
    )


def test_a_valid_frame_passes():
    validate(make_valid_frame())


def test_validation_returns_a_frame():
    result = validate(make_valid_frame())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_documented_sentinel_is_accepted():
    """GSA_MIGRATION is a documented placeholder, not a defect -- it must pass."""
    df = make_valid_frame()
    df.loc[0, "is_low_risk_auditee"] = "GSA_MIGRATION"
    df.loc[1, "audit_type"] = "GSA_MIGRATION"
    validate(df)


def test_unmodelled_columns_pass_through():
    """strict=False, so extra columns are tolerated."""
    df = make_valid_frame()
    df["auditee_name"] = "SOMEBODY"
    result = validate(df)
    assert "auditee_name" in result.columns


def test_leading_zero_zip_is_valid():
    df = make_valid_frame()
    df["auditee_zip"] = "06268"
    validate(df)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("report_id", "NOT-A-REPORT-ID"),
        ("audit_year", 2021),
        ("entity_type", "corporation"),
        ("audit_type", "biennial-audit"),
        ("audit_period_covered", "quarterly"),
        ("data_source", "MYSTERY"),
        ("type_audit_code", "XX"),
        ("is_public", "True"),  # dictionary says True/False; the file uses t/f
        ("is_going_concern_included", "MAYBE"),
        ("is_multiple_eins", "GSA_MIGRATION"),  # not permitted on this column
        ("total_amount_expended", -5.0),
        ("dollar_threshold", 0.0),
        ("auditee_zip", "1234"),
        ("auditor_zip", "ABCDE"),
        ("auditee_state", "TEXAS"),
        ("sp_framework_basis", "vibes_basis"),
        ("gaap_results", "unmodified_opinion, made_up_opinion"),
        ("sp_framework_opinions", "not_a_real_opinion"),
        ("number_months", -3.0),
    ],
)
def test_invalid_value_is_rejected(column, bad_value):
    df = make_valid_frame()
    df.loc[1, column] = bad_value
    with pytest.raises(pa.errors.SchemaErrors):
        validate(df)


def test_duplicate_report_id_is_rejected():
    df = make_valid_frame()
    df.loc[1, "report_id"] = df.loc[0, "report_id"]
    with pytest.raises(pa.errors.SchemaErrors):
        validate(df)


def test_backwards_fiscal_year_is_rejected():
    df = make_valid_frame()
    df.loc[1, "fy_start_date"], df.loc[1, "fy_end_date"] = (
        df.loc[1, "fy_end_date"],
        df.loc[1, "fy_start_date"],
    )
    with pytest.raises(pa.errors.SchemaErrors):
        validate(df)


def test_lazy_validation_collects_every_failure():
    """lazy=True must report all violations at once, not just the first."""
    df = make_valid_frame()
    df.loc[0, "entity_type"] = "corporation"
    df.loc[1, "data_source"] = "MYSTERY"
    df.loc[2, "total_amount_expended"] = -1.0

    with pytest.raises(pa.errors.SchemaErrors) as excinfo:
        validate(df)

    failed = set(excinfo.value.failure_cases["column"])
    assert {"entity_type", "data_source", "total_amount_expended"} <= failed


def test_describe_failures_summarises_by_column_and_check():
    df = make_valid_frame()
    df.loc[0, "entity_type"] = "corporation"

    with pytest.raises(pa.errors.SchemaErrors) as excinfo:
        validate(df)

    summary = describe_failures(excinfo.value)
    assert "entity_type" in summary


def test_schema_can_be_built():
    """Guards against SchemaInitError from a check bound to an undeclared field."""
    assert GeneralSchema.to_schema() is not None
