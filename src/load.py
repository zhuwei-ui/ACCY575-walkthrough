"""Load and clean the FAC `general` extract.

The pipeline is read -> validate -> clean. Validation runs on the frame exactly
as read, before any type conversion, so `GeneralSchema` describes the file on
disk rather than this module's output.

Constraints come from the data dictionary at
https://www.fac.gov/data/download/current-dictionary/ and from what
`notebooks/01-explore.ipynb` found in the extract.
"""

from pathlib import Path

import pandas as pd

from src.validate import validate

# The dictionary declares these TEXT, but pandas infers int64/float64 and drops
# leading zeros ("Presented as text, because some numbers begin with 0").
DTYPES = {
    "auditee_zip": str,
    "auditor_zip": str,
    "auditee_ein": str,
    "auditor_ein": str,
    "auditee_phone": str,
    "auditor_phone": str,
    "cognizant_agency": str,
    "oversight_agency": str,
}

DATE_COLUMNS = (
    "date_created",
    "ready_for_certification_date",
    "auditor_certified_date",
    "auditee_certified_date",
    "submitted_date",
    "fac_accepted_date",
    "fy_start_date",
    "fy_end_date",
)

YES_NO_COLUMNS = (
    "is_sp_framework_required",
    "is_going_concern_included",
    "is_internal_control_deficiency_disclosed",
    "is_internal_control_material_weakness_disclosed",
    "is_material_noncompliance_disclosed",
    "is_low_risk_auditee",
    "is_aicpa_audit_guide_included",
    "is_additional_ueis",
    "is_multiple_eins",
    "is_secondary_auditors",
)

# Documented placeholder for legacy records migrated from the Census system
# where a value could not be definitively classified. Not corruption.
SENTINEL = "GSA_MIGRATION"


def load_general(path: str | Path) -> pd.DataFrame:
    """Read the extract with identifier columns kept as text.

    Blank CSV fields become NaN, which is pandas' default and what the schema
    expects; no separate blank-normalisation step is needed.
    """
    return pd.read_csv(path, dtype=DTYPES, low_memory=False)


def parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the ISO date columns to datetime64. Unparseable values become NaT."""
    out = df.copy()
    for col in DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def yes_no_to_boolean(series: pd.Series) -> pd.Series:
    """Map Yes/No to a nullable boolean; sentinel and blanks become NA.

    Returns pandas' "boolean" dtype so the three-way distinction (True, False,
    unknown) survives -- a plain bool column would silently coerce NA to False.
    """
    return series.map({"Yes": True, "No": False}).astype("boolean")


def true_false_to_boolean(series: pd.Series) -> pd.Series:
    """Map the t/f encoding used by `is_public` to a nullable boolean.

    The dictionary documents this column as True/False; the file holds t/f.
    Both spellings are accepted here so the loader survives either.
    """
    return series.map(
        {"t": True, "f": False, "True": True, "False": False}
    ).astype("boolean")


def uninformative_columns(df: pd.DataFrame) -> list[str]:
    """Columns holding at most one distinct value, so carrying no signal."""
    return [col for col in df.columns if df[col].nunique(dropna=True) <= 1]


def sentinels_to_na(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the GSA_MIGRATION sentinel with NA across all string columns."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype) == "str":
            out[col] = out[col].replace(SENTINEL, pd.NA)
    return out


def clean_general(
    df: pd.DataFrame,
    *,
    parse_dates: bool = True,
    convert_booleans: bool = True,
    sentinels_as_na: bool = True,
    drop_uninformative: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Return an analysis-ready frame plus the list of columns dropped.

    Every step is opt-out. Dropped columns are returned rather than discarded
    silently, so a caller can record what left the dataset.
    """
    out = df

    if sentinels_as_na:
        out = sentinels_to_na(out)

    if convert_booleans:
        out = out.copy()
        for col in YES_NO_COLUMNS:
            if col in out.columns:
                out[col] = yes_no_to_boolean(out[col])
        if "is_public" in out.columns:
            out["is_public"] = true_false_to_boolean(out["is_public"])

    if parse_dates:
        out = parse_date_columns(out)

    dropped: list[str] = []
    if drop_uninformative:
        dropped = uninformative_columns(out)
        out = out.drop(columns=dropped)

    return out, dropped


def load_clean(
    path: str | Path, *, validate_first: bool = True, **clean_kwargs
) -> tuple[pd.DataFrame, list[str]]:
    """read -> validate -> clean.

    Validation is the first step and raises `SchemaErrors` on failure, so no
    downstream work runs against a bad extract. Pass `validate_first=False`
    only when deliberately inspecting a file you already know is broken.
    """
    raw = load_general(path)
    if validate_first:
        raw = validate(raw)
    return clean_general(raw, **clean_kwargs)


if __name__ == "__main__":
    frame, dropped_columns = load_clean("data/general_2016_2018.csv")
    print(f"OK -- {len(frame):,} rows x {frame.shape[1]} columns after cleaning")
    print(f"dropped {len(dropped_columns)} uninformative columns: {dropped_columns}")
    print("\ndtypes of converted columns:")
    for name in ("fy_start_date", "is_low_risk_auditee", "is_public", "auditee_zip"):
        if name in frame.columns:
            print(f"  {name:26} {frame[name].dtype}")
