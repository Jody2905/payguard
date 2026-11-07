"""Reports an auditor can hand on: an Excel workbook and a single-file HTML report."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .rules import LABELS
from .scoring import Result

LEVEL_COLORS = {"Critical": "C0392B", "High": "E67E22", "Medium": "F1C40F", "Low": "95A5A6"}


def case_table(result: Result) -> pd.DataFrame:
    c = result.cases
    detail = result.findings.groupby("payment_id").apply(
        lambda g: " | ".join(g.sort_values("weight", ascending=False).detail), include_groups=False)
    related = result.findings.groupby("payment_id").related.apply(lambda r: ", ".join(sorted({x for xs in r for x in xs})))
    out = c[["payment_id", "risk", "level", "payment_date", "vendor_id", "vendor", "invoice_no", "amount",
             "created_by", "approved_by", "department", "reasons"]].copy()
    out["why"] = out.payment_id.map(detail)
    out["related_payments"] = out.payment_id.map(related)
    out["review_status"] = "Open"
    out["reviewer_notes"] = ""
    return out


def summary(result: Result) -> dict:
    p = result.dataset.payments
    start, end = result.dataset.period
    c = result.cases
    return {
        "Period": f"{start:%d %b %Y} to {end:%d %b %Y}",
        "Payments checked": f"{len(p):,}",
        "Total paid": f"EC${p.amount.sum():,.2f}",
        "Payments to review": f"{len(c):,}",
        "Amount in payments to review": f"EC${c.amount.sum():,.2f}",
        "Critical": f"{(c.level == 'Critical').sum():,}",
        "High": f"{(c.level == 'High').sum():,}",
        "Medium": f"{(c.level == 'Medium').sum():,}",
    }


# ---------------------------------------------------------------- Excel

def to_excel(result: Result, path: str | Path) -> Path:
    path = Path(path)
    cases = case_table(result)
    rules = result.by_rule()[["label", "payments", "amount"]].rename(columns={"label": "rule", "amount": "amount_ec$"})
    findings = result.findings.assign(rule=result.findings.rule.map(LABELS).fillna(result.findings.rule),
                                      related=result.findings.related.map(", ".join))
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(list(summary(result).items()), columns=["", "Value"]).to_excel(xw, sheet_name="Summary", index=False)
        cases.to_excel(xw, sheet_name="Cases to review", index=False)
        rules.to_excel(xw, sheet_name="By rule", index=False)
        result.by_vendor().to_excel(xw, sheet_name="By vendor", index=False)
        result.by_employee().to_excel(xw, sheet_name="By employee", index=False)
        findings.to_excel(xw, sheet_name="All findings", index=False)

        wb = xw.book
        head = PatternFill("solid", fgColor="1F3A5F")
        for ws in wb.worksheets:
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = head
            ws.freeze_panes = "A2"
            for i, col in enumerate(ws.columns, 1):
                width = max(len(str(c.value or "")) for c in list(col)[:200])
                ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 70)
        ws = wb["Cases to review"]
        hdr = [c.value for c in ws[1]]
        lvl, amt, why = hdr.index("level") + 1, hdr.index("amount") + 1, hdr.index("why") + 1
        for row in ws.iter_rows(min_row=2):
            color = LEVEL_COLORS.get(row[lvl - 1].value)
            if color:
                row[lvl - 1].fill = PatternFill("solid", fgColor=color)
                row[lvl - 1].font = Font(bold=True, color="FFFFFF")
            row[amt - 1].number_format = '#,##0.00'
            row[why - 1].alignment = Alignment(wrap_text=True, vertical="top")
        ws.column_dimensions[get_column_letter(why)].width = 90
        ws.auto_filter.ref = ws.dimensions
        for name in ("By rule", "By vendor", "By employee"):
            s = wb[name]
            h = [c.value for c in s[1]]
            for col in ("amount", "amount_ec$"):
                if col in h:
                    for row in s.iter_rows(min_row=2):
                        row[h.index(col)].number_format = '#,##0.00'
    return path


# ---------------------------------------------------------------- HTML

CSS = """
body{font-family:Segoe UI,system-ui,sans-serif;margin:0;background:#f4f6f9;color:#1d2733}
header{background:#1f3a5f;color:#fff;padding:28px 40px}header h1{margin:0;font-size:26px}header p{margin:6px 0 0;opacity:.8}
main{max-width:1200px;margin:0 auto;padding:28px 40px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:28px}
.tile{background:#fff;border-radius:10px;padding:16px 18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.tile b{display:block;font-size:22px;margin-top:4px}.tile span{color:#5b6b7c;font-size:13px}
h2{font-size:18px;margin:30px 0 12px}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:13px}
th{background:#eef2f7;text-align:left;padding:9px 10px;font-weight:600}td{padding:9px 10px;border-top:1px solid #edf0f4;vertical-align:top}
.num{text-align:right;white-space:nowrap}.why{color:#3c4a59;max-width:520px}
.lvl{display:inline-block;padding:2px 9px;border-radius:12px;color:#fff;font-weight:600;font-size:12px}
.bar{background:#dfe6ee;border-radius:4px;height:10px;min-width:120px}.bar i{display:block;height:10px;border-radius:4px;background:#1f3a5f}
footer{color:#5b6b7c;font-size:12px;margin:30px 0}
"""


def _e(x) -> str:
    return html.escape(str(x))


def to_html(result: Result, path: str | Path, title: str = "Payment review", top: int = 100) -> Path:
    path = Path(path)
    s = summary(result)
    tiles = "".join(f'<div class="tile"><span>{_e(k)}</span><b>{_e(v)}</b></div>' for k, v in s.items() if k not in ("Critical", "High", "Medium"))
    lv = "".join(f'<div class="tile"><span>{k}</span><b style="color:#{LEVEL_COLORS[k]}">{_e(s[k])}</b></div>' for k in ("Critical", "High", "Medium"))
    rules = result.by_rule()
    mx = max(rules.payments.max(), 1) if len(rules) else 1
    rule_rows = "".join(
        f'<tr><td>{_e(r.label)}</td><td class="num">{r.payments:,}</td><td><div class="bar"><i style="width:{r.payments / mx * 100:.0f}%"></i></div></td>'
        f'<td class="num">EC${r.amount:,.2f}</td></tr>' for r in rules.itertuples())
    cases = case_table(result).head(top)
    case_rows = "".join(
        f'<tr><td>{_e(r.payment_id)}</td><td><span class="lvl" style="background:#{LEVEL_COLORS[r.level]}">{_e(r.level)} {r.risk:.0f}</span></td>'
        f'<td>{r.payment_date:%d %b %Y}</td><td>{_e(r.vendor)}</td><td class="num">EC${r.amount:,.2f}</td>'
        f'<td class="why">{"<br>".join(_e(x) for x in str(r.why).split(" | "))}</td></tr>' for r in cases.itertuples())
    vend = result.by_vendor().head(15)
    vend_rows = "".join(f'<tr><td>{_e(r.vendor)}</td><td class="num">{r.cases}</td><td class="num">EC${r.amount:,.2f}</td>'
                        f'<td class="num">{r.top_risk:.0f}</td></tr>' for r in vend.itertuples())
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>{CSS}</style></head><body>
<header><h1>{_e(title)}</h1><p>PayGuard report · {_e(s['Period'])} · generated {datetime.now():%d %b %Y %H:%M}</p></header>
<main><div class="tiles">{tiles}{lv}</div>
<h2>What was found</h2><table><tr><th>Check</th><th class="num">Payments</th><th></th><th class="num">Amount</th></tr>{rule_rows}</table>
<h2>Payments to review (top {min(top, len(result.cases))} of {len(result.cases):,})</h2>
<table><tr><th>Payment</th><th>Risk</th><th>Date</th><th>Vendor</th><th class="num">Amount</th><th>Why</th></tr>{case_rows}</table>
<h2>Vendors with the most to review</h2>
<table><tr><th>Vendor</th><th class="num">Payments</th><th class="num">Amount</th><th class="num">Highest risk</th></tr>{vend_rows}</table>
<footer>A flag is a reason to look, not proof of fraud. Each payment's risk score combines its findings: 80+ critical, 60+ high, 35+ medium.</footer>
</main></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path
