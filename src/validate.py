"""Pandera schema for the FAC `general` extract (audit years 2016-2018).

Constraints come from two sources:

* the data dictionary at https://www.fac.gov/data/download/current-dictionary/
* what `notebooks/01-explore.ipynb` actually found in the file

Where the two disagree, the code follows the data and says so in a comment.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

ZIP_PATTERN = r"^[0-9]{5}(?:[0-9]{4})?$"

# `GSA_MIGRATION` is a documented sentinel, not corruption: it marks records
# migrated from the legacy Census system where a value could not be classified.
# It is accepted here; filter it at the point of analysis instead.
YES_NO = ["Yes", "No", "GSA_MIGRATION"]

OPINIONS = {
    "not_gaap",
    "unmodified_opinion",
    "qualified_opinion",
    "adverse_opinion",
    "disclaimer_of_opinion",
    "GSA_MIGRATION",
}


def _opinions_are_known(series: pd.Series) -> pd.Series:
    """`gaap_results` holds a comma-separated list, e.g. 'unmodified_opinion, adverse_opinion'."""
    def ok(value):
        if pd.isna(value) or str(value).strip() == "":
            return True
        return all(token.strip() in OPINIONS for token in str(value).split(","))

    return series.map(ok)


class GeneralSchema(pa.DataFrameModel):
    """Columns carrying signal. Unmodelled columns pass through (strict=False)."""

    # --- identity ---
    report_id: Series[str] = pa.Field(
        unique=True, str_matches=r"^\d{4}-\d{2}-(CENSUS|GSAFAC)-\d{10}$"
    )
    # 99.97% of rows are the GSA_MIGRATION sentinel, so this column carries
    # almost no information for 2016-2018. Kept unconstrained deliberately.
    auditee_uei: Series[str]

    # bounded by the extract, not by the dictionary
    audit_year: Series[int] = pa.Field(ge=2016, le=2018)

    # --- documented vocabularies ---
    entity_type: Series[str] = pa.Field(
        isin=["non-profit", "state", "local", "higher-ed", "tribal", "unknown"]
    )
    # GSA_MIGRATION is documented but absent from 2016-2018; kept for forward
    # compatibility when later years are added.
    audit_type: Series[str] = pa.Field(
        isin=["single-audit", "program-specific", "GSA_MIGRATION"]
    )
    audit_period_covered: Series[str] = pa.Field(isin=["annual", "biennial", "other"])
    data_source: Series[str] = pa.Field(isin=["CENSUS", "GSAFAC"])
    type_audit_code: Series[str] = pa.Field(isin=["UG"])

    # DISCREPANCY: the dictionary documents True/False, the file contains t/f.
    # Encoding what the data actually holds.
    is_public: Series[str] = pa.Field(isin=["t", "f"])

    # --- boolean-style indicators ---
    is_low_risk_auditee: Series[str] = pa.Field(isin=YES_NO)
    is_going_concern_included: Series[str] = pa.Field(isin=YES_NO)
    is_internal_control_deficiency_disclosed: Series[str] = pa.Field(isin=YES_NO)
    is_internal_control_material_weakness_disclosed: Series[str] = pa.Field(isin=YES_NO)
    is_material_noncompliance_disclosed: Series[str] = pa.Field(isin=YES_NO)
    is_aicpa_audit_guide_included: Series[str] = pa.Field(isin=YES_NO)
    is_multiple_eins: Series[str] = pa.Field(isin=["Yes", "No"])
    is_secondary_auditors: Series[str] = pa.Field(isin=["Yes", "No"])

    # --- opinions: comma-separated composites, validated token-wise below ---
    gaap_results: Series[str] = pa.Field(nullable=True)
    sp_framework_opinions: Series[str] = pa.Field(nullable=True)

    # --- amounts ---
    total_amount_expended: Series[float] = pa.Field(gt=0)
    dollar_threshold: Series[float] = pa.Field(gt=0)

    # --- identifiers kept as text (see DTYPES) ---
    auditee_zip: Series[str] = pa.Field(str_matches=ZIP_PATTERN)
    auditor_zip: Series[str] = pa.Field(str_matches=ZIP_PATTERN)
    auditee_ein: Series[str]
    auditee_state: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2})
    auditor_state: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2})

    # --- sparse columns: nullable is the finding ---
    # 96.6% and 3.4% null respectively
    cognizant_agency: Series[str] = pa.Field(nullable=True)
    oversight_agency: Series[str] = pa.Field(nullable=True)
    # 94.6% null -- only applies to special-framework audits
    sp_framework_basis: Series[str] = pa.Field(
        nullable=True,
        isin=["other_basis", "cash_basis", "tax_basis", "regulatory_basis", "contractual_basis"],
    )
    is_sp_framework_required: Series[str] = pa.Field(nullable=True, isin=YES_NO)
    # 99.997% null (3 non-null rows in 110,948)
    number_months: Series[float] = pa.Field(nullable=True, gt=0)
    # 100% null across the whole extract -- structurally empty
    auditor_foreign_address: Series[float] = pa.Field(nullable=True)

    # --- dates (kept as strings; parsed in the checks below) ---
    fy_start_date: Series[str]
    fy_end_date: Series[str]

    class Config:
        strict = False
        coerce = True

    @pa.check("gaap_results", name="gaap_results_tokens_are_documented")
    def _gaap_results_known(cls, series: Series[str]) -> Series[bool]:
        return _opinions_are_known(series)

    @pa.check("sp_framework_opinions", name="sp_framework_opinion_tokens_are_documented")
    def _sp_opinions_known(cls, series: Series[str]) -> Series[bool]:
        return _opinions_are_known(series)

    @pa.dataframe_check(name="fiscal_year_does_not_run_backwards")
    def _fy_forward(cls, df: pd.DataFrame) -> Series[bool]:
        start = pd.to_datetime(df["fy_start_date"], errors="coerce")
        end = pd.to_datetime(df["fy_end_date"], errors="coerce")
        return end >= start


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate `df`, collecting every failure rather than stopping at the first."""
    return GeneralSchema.validate(df, lazy=True)


def describe_failures(exc: pa.errors.SchemaErrors) -> str:
    """Group failure cases by column and check for a readable summary."""
    return exc.failure_cases.groupby(["column", "check"]).size().to_string()
