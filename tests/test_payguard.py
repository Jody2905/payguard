"""Tests: each rule on small hand-made data, then the whole pipeline on generated data with known fraud."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from payguard import DataError, analyse, prepare
from payguard.cli import main
from payguard.evaluate import evaluate
from payguard.generate import generate
from payguard.holidays import holidays, is_holiday
from payguard.rules import (Settings, bank_change, employee_vendor_link, exact_duplicates, just_below_limit, near_duplicates,
                            norm_invoice, over_limit, self_approval, split_payments)

EMPLOYEES = pd.DataFrame([
    dict(employee_id="E1", name="Clerk One", address="1 Bay Street, Kingstown", phone="784-456-1111", bank_account="WB-111", department="Finance", approval_limit=0),
    dict(employee_id="E2", name="Approver Two", address="2 Lodge Road, Villa", phone="784-456-2222", bank_account="WB-222", department="Finance", approval_limit=5000),
    dict(employee_id="E3", name="Approver Three", address="3 Murray Road, Layou", phone="784-456-3333", bank_account="WB-333", department="Works", approval_limit=25000),
])


def vendor(vid, **kw):
    v = dict(vendor_id=vid, name=f"Vendor {vid}", address=f"{vid[-1]}0 Main Road, Calliaqua", phone=f"784-500-{vid[-1]}000",
             bank_account=f"GB-{vid}", created_date="2020-01-01", created_by="E1", bank_changed_date="")
    v.update(kw)
    return v


def pay(pid, vid, when, amount, inv, **kw):
    p = dict(payment_id=pid, payment_date=when, vendor_id=vid, invoice_no=inv, invoice_date="2025-01-01", amount=amount,
             bank_account=f"GB-{vid}", created_by="E1", approved_by="E3", department="Works", description="test")
    p.update(kw)
    return p


def ds(payments, vendors=None, employees=EMPLOYEES):
    vendors = vendors or [vendor(v) for v in sorted({p["vendor_id"] for p in payments})]
    return prepare(pd.DataFrame(payments), pd.DataFrame(vendors), employees)


def ids(findings) -> set[str]:
    return set(findings.payment_id)


# ---------------------------------------------------------------- input checks

def test_missing_column_is_explained():
    with pytest.raises(DataError, match="amount"):
        prepare(pd.DataFrame([{"payment_id": "P1"}]), pd.DataFrame([vendor("V1")]), EMPLOYEES)


def test_payment_to_unknown_vendor():
    with pytest.raises(DataError, match="not in vendors"):
        ds([pay("P1", "V9", "2025-03-03 10:00", 100, "1")], vendors=[vendor("V1")])


def test_unreadable_amount():
    with pytest.raises(DataError, match="no readable date or amount"):
        ds([pay("P1", "V1", "2025-03-03 10:00", "abc", "1")])


# ---------------------------------------------------------------- rules

def test_exact_duplicate_ignores_punctuation_and_case():
    d = ds([pay("P1", "V1", "2025-03-03 10:00", 1200, "INV-100"), pay("P2", "V1", "2025-03-10 10:00", 1200, "inv 100"),
            pay("P3", "V2", "2025-03-10 10:00", 1200, "INV-100")])  # another vendor's invoice 100 is fine
    f = exact_duplicates(d, Settings())
    assert ids(f) == {"P2"} and f.related.iloc[0] == ["P1"]
    assert norm_invoice(" inv-100/a ") == "INV100A"


def test_near_duplicate_changed_invoice_number():
    d = ds([pay("P1", "V1", "2025-03-03 10:00", 2450.5, "INV-4471"), pay("P2", "V1", "2025-03-07 10:00", 2450.5, "INV-4471A"),
            pay("P3", "V1", "2025-03-05 10:00", 2450.5, "INV-9902"),     # same amount, unrelated invoice: fine
            pay("P4", "V1", "2025-05-30 10:00", 2450.5, "INV-4471B")])   # too long after: fine
    assert ids(near_duplicates(d, Settings())) == {"P2"}


def test_monthly_contract_is_not_a_duplicate():
    d = ds([pay(f"P{m}", "V1", f"2025-{m:02d}-02 10:00", 10000, f"RENT-{m}") for m in range(1, 13)])
    assert not len(near_duplicates(d, Settings())) and not len(exact_duplicates(d, Settings()))


def test_split_payment_under_limit():
    rows = [pay("P1", "V1", "2025-03-03 10:00", 4800, "A1", approved_by="E2"), pay("P2", "V1", "2025-03-04 10:00", 4700, "A2", approved_by="E2")]
    rows += [pay(f"Q{i}", "V1", f"2025-{m:02d}-15 10:00", 900, f"B{i}", approved_by="E2") for i, m in enumerate(range(5, 12))]
    f = split_payments(ds(rows), Settings())
    assert ids(f) == {"P1", "P2"}
    assert "over E2's EC$5,000.00 limit" in f.detail.iloc[0]


def test_one_payment_over_the_limit_is_not_a_split():
    rows = [pay("P1", "V1", "2025-03-03 10:00", 6000, "A1", approved_by="E3"), pay("P2", "V1", "2025-03-04 10:00", 200, "A2", approved_by="E2")]
    assert not len(split_payments(ds(rows), Settings()))


def test_just_below_limit_pattern():
    rows = [pay(f"P{i}", "V1", f"2025-{1 + i % 12:02d}-{10 + i % 9} 10:00", 4900 + i, f"X{i}", approved_by="E2") for i in range(8)]
    rows += [pay(f"N{i}", f"V{2 + i % 5}", f"2025-{1 + i % 12:02d}-05 11:00", 300 + 37 * i, f"Y{i}", approved_by="E2") for i in range(60)]
    assert ids(just_below_limit(ds(rows), Settings())) == {f"P{i}" for i in range(8)}


def test_self_approval_and_over_limit():
    d = ds([pay("P1", "V1", "2025-03-03 10:00", 900, "1", created_by="E2", approved_by="E2"),
            pay("P2", "V1", "2025-03-03 11:00", 9000, "2", approved_by="E2"),
            pay("P3", "V1", "2025-03-03 12:00", 900, "3", approved_by="E9")])
    assert ids(self_approval(d, Settings())) == {"P1"}
    f = over_limit(d, Settings())
    assert set(zip(f.payment_id, f.rule)) == {("P2", "over_limit"), ("P3", "unknown_approver")}


def test_vendor_with_employee_bank_account_or_address():
    vendors = [vendor("V1", bank_account="WB-222"),                              # Approver Two's salary account
               vendor("V2", address="3 Murray Rd., Layou"),                     # Approver Three's address, written differently
               vendor("V3", address="4 Murray Road, Layou")]                    # a neighbour: fine
    d = ds([pay("P1", "V1", "2025-03-03 10:00", 100, "1"), pay("P2", "V2", "2025-03-03 10:00", 100, "1"),
            pay("P3", "V3", "2025-03-03 10:00", 100, "1")], vendors=vendors)
    f = employee_vendor_link(d, Settings())
    assert ids(f) == {"P1", "P2"}
    assert "bank account" in f.set_index("payment_id").detail["P1"]


def test_bank_change_then_big_payment():
    v = [vendor("V1", bank_changed_date="2025-06-01")]
    rows = [pay(f"P{i}", "V1", f"2025-{1 + i:02d}-10 10:00", 1000, f"I{i}") for i in range(5)]
    rows.append(pay("BIG", "V1", "2025-06-05 10:00", 9000, "I99"))
    rows.append(pay("LATE", "V1", "2025-09-05 10:00", 9000, "I100"))
    f = bank_change(ds(rows, vendors=v), Settings())
    assert ids(f) == {"BIG"} and f.weight.iloc[0] >= 0.8


# ---------------------------------------------------------------- holidays

def test_svg_holidays():
    h = holidays(2025)
    assert h[date(2025, 10, 27)] == "Independence Day"
    assert h[date(2025, 4, 18)] == "Good Friday"
    assert h[date(2025, 7, 7)] == "Carnival Monday"
    assert h[date(2025, 8, 4)] == "Emancipation Day"
    assert is_holiday(pd.Timestamp("2025-12-25 09:00")) == "Christmas Day"
    assert is_holiday(date(2025, 12, 24)) is None


# ---------------------------------------------------------------- whole pipeline

@pytest.fixture(scope="module")
def sample():
    t = generate(n_payments=3000, seed=3)
    return t, analyse(prepare(t["payments"], t["vendors"], t["employees"]))


def test_catches_planted_fraud(sample):
    t, r = sample
    e = evaluate(r, t["truth"])
    assert e["recall"] >= 0.9, e["per_scheme"]
    assert e["precision"] >= 0.5
    caught = dict(zip(e["per_scheme"].scheme, e["per_scheme"].recall))
    for scheme in ("duplicate", "ghost_vendor", "self_approval"):
        assert caught[scheme] == 1.0, scheme


def test_every_flag_is_explained(sample):
    _, r = sample
    assert (r.findings.detail.str.len() > 20).all()
    assert r.cases.risk.between(35, 100).all()
    assert list(r.cases.risk) == sorted(r.cases.risk, reverse=True)


def test_cli_end_to_end(tmp_path, capsys):
    data, out = tmp_path / "data", tmp_path / "out"
    main(["generate", "--out", str(data), "--payments", "1500", "--seed", "5"])
    main(["scan", str(data), "--out", str(out)])
    assert (out / "payguard_cases.xlsx").stat().st_size > 5000
    page = (out / "payguard_report.html").read_text()
    assert "Payments to review" in page and "<script" not in page
    main(["evaluate", str(data)])
    assert "recall" in capsys.readouterr().out
    xl = pd.read_excel(out / "payguard_cases.xlsx", sheet_name=None)
    assert {"Summary", "Cases to review", "By rule", "By vendor", "By employee", "All findings"} <= set(xl)
