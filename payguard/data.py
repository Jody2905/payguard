"""Loading and checking the three input tables: payments, vendors and employees."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PAYMENT_COLUMNS = {
    "payment_id": "Unique id of the payment",
    "payment_date": "When the payment was entered (date and time)",
    "vendor_id": "Who was paid (matches vendors.vendor_id)",
    "invoice_no": "The vendor's invoice number",
    "invoice_date": "Date on the invoice",
    "amount": "Amount paid (EC$)",
    "bank_account": "Account the money went to",
    "created_by": "Employee who entered the payment",
    "approved_by": "Employee who approved it",
    "department": "Department paying",
    "description": "What it was for",
}
VENDOR_COLUMNS = {
    "vendor_id": "Unique id",
    "name": "Vendor name",
    "address": "Address",
    "phone": "Phone number",
    "bank_account": "Current bank account on file",
    "created_date": "When the vendor was set up",
    "created_by": "Employee who set the vendor up",
    "bank_changed_date": "When the bank account was last changed (blank if never)",
}
EMPLOYEE_COLUMNS = {
    "employee_id": "Unique id",
    "name": "Name",
    "address": "Home address",
    "phone": "Phone number",
    "bank_account": "Salary account",
    "department": "Department",
    "approval_limit": "Most they can approve in one payment (EC$; 0 = cannot approve)",
}
OPTIONAL = {"description", "department", "invoice_date", "bank_changed_date", "created_by"}


class DataError(ValueError):
    """The input files are missing something PayGuard needs."""


@dataclass
class Dataset:
    payments: pd.DataFrame
    vendors: pd.DataFrame
    employees: pd.DataFrame

    @property
    def period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.payments["payment_date"].min(), self.payments["payment_date"].max()


def _check(df: pd.DataFrame, columns: dict, table: str) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    missing = [c for c in columns if c not in df.columns and c not in OPTIONAL]
    if missing:
        raise DataError(f"{table} is missing column(s): {', '.join(missing)}")
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def prepare(payments: pd.DataFrame, vendors: pd.DataFrame, employees: pd.DataFrame) -> Dataset:
    """Check columns and types so every rule can rely on them."""
    p = _check(payments.copy(), PAYMENT_COLUMNS, "payments")
    v = _check(vendors.copy(), VENDOR_COLUMNS, "vendors")
    e = _check(employees.copy(), EMPLOYEE_COLUMNS, "employees")

    for df, cols in ((p, ["payment_id", "vendor_id", "invoice_no", "bank_account", "created_by", "approved_by", "department", "description"]),
                     (v, ["vendor_id", "name", "address", "phone", "bank_account", "created_by"]),
                     (e, ["employee_id", "name", "address", "phone", "bank_account", "department"])):
        for c in cols:
            df[c] = _text(df[c])

    p["payment_date"] = pd.to_datetime(p["payment_date"], errors="coerce")
    p["invoice_date"] = pd.to_datetime(p["invoice_date"], errors="coerce")
    p["amount"] = pd.to_numeric(p["amount"], errors="coerce")
    bad = p["payment_date"].isna() | p["amount"].isna()
    if bad.any():
        rows = ", ".join(p.loc[bad, "payment_id"].astype(str).head(5))
        raise DataError(f"{int(bad.sum())} payment(s) have no readable date or amount (e.g. {rows})")
    if p["payment_id"].duplicated().any():
        raise DataError("payment_id must be unique")

    v["created_date"] = pd.to_datetime(v["created_date"], errors="coerce")
    v["bank_changed_date"] = pd.to_datetime(v["bank_changed_date"], errors="coerce")
    e["approval_limit"] = pd.to_numeric(e["approval_limit"], errors="coerce").fillna(0)

    unknown = set(p["vendor_id"]) - set(v["vendor_id"])
    if unknown:
        raise DataError(f"{len(unknown)} payment(s) go to vendors not in vendors.csv (e.g. {sorted(unknown)[:3]})")

    p = p.sort_values("payment_date").reset_index(drop=True)
    return Dataset(p, v.reset_index(drop=True), e.reset_index(drop=True))


def load_folder(folder: str | Path) -> Dataset:
    """Read payments.csv, vendors.csv and employees.csv from a folder."""
    folder = Path(folder)
    files = {n: folder / f"{n}.csv" for n in ("payments", "vendors", "employees")}
    missing = [str(f.name) for f in files.values() if not f.exists()]
    if missing:
        raise DataError(f"{folder} is missing {', '.join(missing)}")
    read = lambda f: pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])
    return prepare(read(files["payments"]), read(files["vendors"]), read(files["employees"]))
