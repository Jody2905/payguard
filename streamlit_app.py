"""PayGuard dashboard. Run with:  streamlit run streamlit_app.py"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from payguard.data import DataError, prepare
from payguard.evaluate import evaluate
from payguard.generate import generate
from payguard.report import LEVEL_COLORS, case_table, to_excel, to_html
from payguard.rules import LABELS, RULES, Settings
from payguard.scoring import analyse

st.set_page_config(page_title="PayGuard", page_icon="🛡️", layout="wide")
COLORS = {k: "#" + v for k, v in LEVEL_COLORS.items()}
STATUSES = ["Open", "Checked: fine", "Checked: error, recover money", "Escalated to audit"]


@st.cache_data(show_spinner="Making sample data…")
def sample(seed: int) -> dict[str, pd.DataFrame]:
    return generate(seed=seed)


@st.cache_data(show_spinner="Checking payments…")
def run(payments: pd.DataFrame, vendors: pd.DataFrame, employees: pd.DataFrame, enabled: tuple, outlier_share: float):
    ds = prepare(payments, vendors, employees)
    s = Settings()
    s.enabled = set(enabled)
    r = analyse(ds, s, outlier_share=outlier_share)
    return r


def money(x: float) -> str:
    return f"EC${x:,.2f}"


def short(x: float) -> str:
    """EC$46.5M / EC$388K, for the big number tiles."""
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= size:
            return f"EC${x / size:,.1f}{unit}"
    return f"EC${x:,.0f}"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛡️ PayGuard")
    st.caption("Finds duplicate, split and suspicious payments, and explains why.")
    source = st.radio("Data", ["Sample treasury data", "Upload my own CSV files"])
    truth = None
    if source == "Sample treasury data":
        seed = st.number_input("Sample number", 1, 9999, 7, help="Each number makes a different made-up year of payments.")
        t = sample(int(seed))
        payments, vendors, employees, truth = t["payments"], t["vendors"], t["employees"], t["truth"]
        st.info("Made-up data with fraud planted in it, so you can see how much PayGuard catches.")
    else:
        up = {n: st.file_uploader(f"{n}.csv", type="csv", key=n) for n in ("payments", "vendors", "employees")}
        with st.expander("What columns do I need?"):
            from payguard.data import EMPLOYEE_COLUMNS, PAYMENT_COLUMNS, VENDOR_COLUMNS
            for name, cols in (("payments", PAYMENT_COLUMNS), ("vendors", VENDOR_COLUMNS), ("employees", EMPLOYEE_COLUMNS)):
                st.markdown(f"**{name}.csv**: " + ", ".join(f"`{c}`" for c in cols))
        if not all(up.values()):
            st.stop()
        read = lambda f: pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])
        payments, vendors, employees = (read(up[n]) for n in ("payments", "vendors", "employees"))

    st.subheader("Checks")
    enabled = tuple(name for name in RULES if st.checkbox(LABELS[name], value=True, key="rule_" + name))
    outlier_share = st.slider("Statistical outliers to flag", 0.0, 0.05, 0.01, 0.005, format="%.3f",
                              help="Share of payments the Isolation Forest model marks as unusual. 0 turns it off.")
    threshold = st.slider("Review from risk score", 10, 90, 35, 5)

try:
    result = run(payments, vendors, employees, enabled, outlier_share)
except DataError as e:
    st.error(f"Problem with the files: {e}")
    st.stop()

ds = result.dataset
scored = result.scored
cases = scored[scored.risk >= threshold].sort_values(["risk", "amount"], ascending=False)
if "review" not in st.session_state:
    st.session_state.review = {}

# ---------------------------------------------------------------- header
start, end = ds.period
st.header("Payment review")
st.caption(f"{start:%d %b %Y} to {end:%d %b %Y} · {len(ds.payments):,} payments · {len(ds.vendors)} vendors · {len(ds.employees)} staff")
m = st.columns(5)
m[0].metric("Total paid", short(ds.payments.amount.sum()), help=money(ds.payments.amount.sum()))
m[1].metric("To review", f"{len(cases):,}")
m[2].metric("Amount to review", short(cases.amount.sum()), help=money(cases.amount.sum()))
m[3].metric("Critical", int((cases.level == "Critical").sum()))
m[4].metric("Reviewed", sum(1 for pid in cases.payment_id if st.session_state.review.get(pid, {}).get("status", "Open") != "Open"))

tabs = st.tabs(["Overview", "Cases", "Vendors", "Staff"] + (["How accurate is it?"] if truth is not None else []))

# ---------------------------------------------------------------- overview
with tabs[0]:
    c1, c2 = st.columns(2)
    br = result.by_rule()
    fig = px.bar(br.sort_values("payments"), x="payments", y="label", orientation="h", labels={"label": "", "payments": "Payments flagged"},
                 title="What each check found", text="payments")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    c1.plotly_chart(fig, use_container_width=True)

    lv = cases.level.value_counts().reindex(["Critical", "High", "Medium", "Low"]).dropna().reset_index()
    lv.columns = ["level", "payments"]
    fig = px.pie(lv, names="level", values="payments", color="level", color_discrete_map=COLORS, hole=0.55, title="Payments to review by risk")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    c2.plotly_chart(fig, use_container_width=True)

    monthly = scored.assign(month=scored.payment_date.dt.to_period("M").dt.to_timestamp(),
                            flagged=scored.risk >= threshold).groupby(["month", "flagged"]).amount.sum().reset_index()
    monthly["flagged"] = monthly.flagged.map({True: "To review", False: "Not flagged"})
    fig = px.bar(monthly, x="month", y="amount", color="flagged", title="Money paid each month",
                 color_discrete_map={"To review": "#C0392B", "Not flagged": "#9FB3C8"}, labels={"amount": "EC$", "month": ""})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    d1, d2 = st.columns(2)
    xbuf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        to_excel(result, Path(tmp) / "cases.xlsx")
        xbuf.write((Path(tmp) / "cases.xlsx").read_bytes())
        to_html(result, Path(tmp) / "report.html")
        html_bytes = (Path(tmp) / "report.html").read_bytes()
    d1.download_button("⬇️ Excel workbook of cases", xbuf.getvalue(), "payguard_cases.xlsx", use_container_width=True)
    d2.download_button("⬇️ HTML report", html_bytes, "payguard_report.html", "text/html", use_container_width=True)

# ---------------------------------------------------------------- cases
with tabs[1]:
    f1, f2, f3 = st.columns([1, 2, 2])
    levels = f1.multiselect("Risk", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])
    rule_pick = f2.multiselect("Found by", sorted({LABELS.get(r, r) for r in result.findings.rule}))
    search = f3.text_input("Search vendor, payment or invoice")
    view = cases[cases.level.isin(levels)]
    if rule_pick:
        view = view[view.reasons.map(lambda s: any(r in s for r in rule_pick))]
    if search:
        q = search.lower()
        view = view[view.vendor.str.lower().str.contains(q, na=False) | view.payment_id.str.lower().str.contains(q)
                    | view.invoice_no.str.lower().str.contains(q, na=False)]
    view = view.assign(status=view.payment_id.map(lambda p: st.session_state.review.get(p, {}).get("status", "Open")))
    st.caption(f"{len(view):,} payments · {money(view.amount.sum())}. Click a row to see why it was flagged.")
    pick = st.dataframe(
        view[["payment_id", "risk", "level", "payment_date", "vendor", "invoice_no", "amount", "approved_by", "reasons", "status"]],
        hide_index=True, use_container_width=True, height=380, on_select="rerun", selection_mode="single-row",
        column_config={"risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
                       "amount": st.column_config.NumberColumn("Amount (EC$)", format="accounting"),
                       "payment_date": st.column_config.DatetimeColumn("Date", format="DD MMM YYYY HH:mm"),
                       "reasons": "Found by", "payment_id": "Payment", "level": "Level", "invoice_no": "Invoice",
                       "approved_by": "Approved by", "vendor": "Vendor", "status": "Review"})
    rows = pick.selection.rows if pick and pick.selection else []
    if rows:
        r = view.iloc[rows[0]]
        st.divider()
        a, b = st.columns([3, 2])
        with a:
            st.subheader(f"{r.payment_id} · {money(r.amount)}")
            st.markdown(f"<span style='background:{COLORS[r.level]};color:white;padding:3px 10px;border-radius:10px;font-weight:600'>"
                        f"{r.level} risk {r.risk:.0f}</span>", unsafe_allow_html=True)
            st.markdown(f"**{r.vendor}** ({r.vendor_id}) · invoice **{r.invoice_no}** · paid {r.payment_date:%d %b %Y %H:%M} "
                        f"to account `{r.bank_account}`  \nEntered by **{r.created_by}**, approved by **{r.approved_by}** · {r.department}")
            st.markdown("#### Why it was flagged")
            for f in result.findings_for(r.payment_id).itertuples():
                st.markdown(f"- **{LABELS.get(f.rule, f.rule)}** (weight {f.weight:.2f}): {f.detail}")
            related = sorted({x for xs in result.findings_for(r.payment_id).related for x in xs})
            if related:
                st.markdown("#### Related payments")
                st.dataframe(scored[scored.payment_id.isin(related)][["payment_id", "payment_date", "invoice_no", "amount", "approved_by", "risk"]],
                             hide_index=True, use_container_width=True)
        with b:
            st.markdown("#### Your review")
            cur = st.session_state.review.get(r.payment_id, {"status": "Open", "notes": ""})
            status = st.selectbox("Outcome", STATUSES, index=STATUSES.index(cur["status"]), key="st_" + r.payment_id)
            notes = st.text_area("Notes", cur["notes"], key="nt_" + r.payment_id, placeholder="What you checked and found")
            if st.button("Save review", type="primary"):
                st.session_state.review[r.payment_id] = {"status": status, "notes": notes}
                st.toast("Review saved for this session. Download the review log to keep it.", icon="✅")
                st.rerun()
            vh = scored[scored.vendor_id == r.vendor_id]
            st.markdown("#### This vendor")
            st.caption(f"{len(vh)} payments, {money(vh.amount.sum())} in total, usually {money(vh.amount.median())}")
            fig = px.scatter(vh, x="payment_date", y="amount", color="level", color_discrete_map=COLORS, height=240,
                             labels={"payment_date": "", "amount": "EC$"})
            fig.add_scatter(x=[r.payment_date], y=[r.amount], mode="markers", marker=dict(size=16, symbol="circle-open", color="black"),
                            name="This payment")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    if st.session_state.review:
        log = pd.DataFrame([{"payment_id": k, **v} for k, v in st.session_state.review.items()])
        st.download_button("⬇️ Download review log (CSV)", log.to_csv(index=False), "payguard_review_log.csv")

# ---------------------------------------------------------------- vendors / staff
with tabs[2]:
    bv = result.by_vendor()
    st.caption("Vendors with payments to review, riskiest first.")
    st.dataframe(bv, hide_index=True, use_container_width=True,
                 column_config={"amount": st.column_config.NumberColumn("Amount (EC$)", format="accounting"),
                                "top_risk": st.column_config.ProgressColumn("Highest risk", min_value=0, max_value=100, format="%.0f")})
with tabs[3]:
    st.caption("Staff who entered or approved payments to review. Many flags can mean a problem, or just a busy job: check the reasons.")
    st.dataframe(result.by_employee(), hide_index=True, use_container_width=True,
                 column_config={"amount": st.column_config.NumberColumn("Amount (EC$)", format="accounting")})

# ---------------------------------------------------------------- accuracy
if truth is not None:
    with tabs[4]:
        e = evaluate(result, truth, threshold=threshold)
        st.markdown("The sample data has fraud planted in it. This is how much PayGuard found at the current settings.")
        k = st.columns(3)
        k[0].metric("Planted fraud caught (recall)", f"{e['recall']:.0%}")
        k[0].caption(f"{e['caught']} of {e['planted']} planted payments")
        k[1].metric("Flags that were fraud (precision)", f"{e['precision']:.0%}")
        k[1].caption(f"{e['caught']} of {e['flagged']} flagged payments")
        k[2].metric("F1 score", f"{e['f1']:.2f}")
        ps = e["per_scheme"]
        fig = px.bar(ps, x="recall", y="scheme_label", orientation="h", range_x=[0, 1], text=ps.apply(lambda r: f"{r.caught}/{r.planted}", axis=1),
                     labels={"scheme_label": "", "recall": "Share caught"}, title="Caught, by type of fraud")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if len(e["missed"]):
            st.markdown("**Missed**")
            st.dataframe(e["missed"], hide_index=True, use_container_width=True)
        st.caption("Try turning checks off, or moving the review threshold, to see the trade-off between catching more and flagging less.")
