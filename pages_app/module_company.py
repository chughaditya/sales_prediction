"""
module_company.py
--------------------
Company Sales Prediction module (one of three tabs on the Prediction page).
"""

import streamlit as st
import pandas as pd

from prediction.specialized_predictor import get_specialized_predictor
from prediction.module_pdf import generate_module_pdf
from charts.premium_charts import gauge_chart, confidence_meter, donut_breakdown, trend_line, feature_contribution_bar, shap_waterfall
from components.glass_ui import module_hero, glass_card_start, glass_card_end, metric_tile, risk_badge, insight_card, prediction_loading, result_headline
from components.batch_upload import render_batch_upload
from utils.helpers import dataframe_to_csv_bytes
from utils.database import get_db

BUNDLE_FILE = "company_sales_bundle.pkl"
MODULE_KEY = "company_sales"


def _revenue_category(value: float) -> str:
    if value < 2000:
        return "Emerging"
    if value < 8000:
        return "Growth Stage"
    if value < 25000:
        return "Established"
    return "Enterprise Scale"


def _risk_level(growth_pct: float, confidence: float) -> str:
    if confidence is None:
        confidence = 60
    if growth_pct < 0 or confidence < 55:
        return "High"
    if growth_pct < 15 or confidence < 75:
        return "Medium"
    return "Low"


def _build_insights(inputs: dict, prediction: float, growth_pct: float, category: str, risk: str, explain: dict):
    top_driver = explain["all_sorted_by_abs"][0][0].split("=")[0] if explain["all_sorted_by_abs"] else "Previous Year Revenue"
    points = [
        f"The model's single biggest influence on this forecast was <b>{top_driver}</b>.",
        f"Projected growth of <b>{growth_pct:+.1f}%</b> places this company in the <b>{category}</b> revenue band.",
    ]
    if inputs["Marketing Budget"] < inputs["Previous Year Revenue"] * 0.03:
        points.append("Marketing spend looks low relative to prior revenue &mdash; increasing it could unlock further upside.")
    else:
        points.append("Marketing investment is healthy relative to revenue, supporting sustained demand generation.")
    if inputs["Online Presence Score"] < 40:
        points.append("Online presence score is below average; improving digital visibility is a quick lever for growth.")
    if inputs["Employee Growth Rate"] > 20:
        points.append("Rapid headcount growth detected &mdash; ensure operational costs scale efficiently alongside it.")
    points.append(f"Overall business risk for this forecast is assessed as <b>{risk}</b>.")
    return points


def render():
    predictor = get_specialized_predictor(BUNDLE_FILE)
    db = get_db()

    module_hero("\U0001F3E2", "Company Sales Prediction", "Forecast annual sales from company fundamentals, spend, and growth signals.")

    if not predictor.is_ready:
        st.warning("Company Sales model not found. Run `training/build_specialized_models.py` once to generate it.")
        return

    bundle = predictor.bundle
    ranges = bundle["numeric_ranges"]
    cats = bundle["categories"]

    mode = st.radio("Prediction Mode", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True, key=f"{MODULE_KEY}_mode")

    if mode == "Batch Upload (CSV)":
        render_batch_upload(predictor, MODULE_KEY, "Company Sales", db)
        return

    glass_card_start()
    st.markdown("**Company Profile**")
    company_name = st.text_input(
        "Company Name", value="", placeholder="e.g. Nova Retail Pvt Ltd",
        key=f"{MODULE_KEY}_name",
        help="Optional label so you can find this forecast again later on the Prediction History page.",
    )
    c1, c2 = st.columns(2)
    with c1:
        industry = st.selectbox("Industry", cats["Industry"], key=f"{MODULE_KEY}_industry")
        country = st.selectbox("Country", cats["Country"], key=f"{MODULE_KEY}_country")
        company_age = st.number_input("Company Age (years)", min_value=0.0, max_value=150.0, value=float(round(ranges["Company Age"][1])), key=f"{MODULE_KEY}_age")
        employees = st.number_input("Number of Employees", min_value=1.0, value=float(round(ranges["Number of Employees"][1])), key=f"{MODULE_KEY}_emp")
        market_size = st.number_input("Market Size ($M)", min_value=0.0, value=round(ranges["Market Size"][1], 1), key=f"{MODULE_KEY}_market")
        customer_count = st.number_input("Customer Count", min_value=0.0, value=float(round(ranges["Customer Count"][1])), key=f"{MODULE_KEY}_cust")
        annual_expenses = st.number_input("Annual Expenses ($K)", min_value=0.0, value=round(ranges["Annual Expenses"][1], 1), key=f"{MODULE_KEY}_exp")
    with c2:
        funding = st.number_input("Funding Raised ($K)", min_value=0.0, value=round(ranges["Funding Raised"][1], 1), key=f"{MODULE_KEY}_funding")
        prev_revenue = st.number_input("Previous Year Revenue ($K)", min_value=0.0, value=round(ranges["Previous Year Revenue"][1], 1), key=f"{MODULE_KEY}_prevrev")
        marketing_budget = st.number_input("Marketing Budget ($K)", min_value=0.0, value=round(ranges["Marketing Budget"][1], 1), key=f"{MODULE_KEY}_mkt")
        rd_spend = st.number_input("R&D Spending ($K)", min_value=0.0, value=round(ranges["R&D Spending"][1], 1), key=f"{MODULE_KEY}_rd")
        op_cost = st.number_input("Operational Cost ($K)", min_value=0.0, value=round(ranges["Operational Cost"][1], 1), key=f"{MODULE_KEY}_op")
        online_score = st.slider("Online Presence Score", 0, 100, int(ranges["Online Presence Score"][1]), key=f"{MODULE_KEY}_online")
        growth_rate = st.slider("Employee Growth Rate (%)", -20, 100, int(ranges["Employee Growth Rate"][1]), key=f"{MODULE_KEY}_egr")
    glass_card_end()

    bcol1, bcol2 = st.columns([1, 1])
    predict_clicked = bcol1.button("\U0001F52E Predict Sales", type="primary", width='stretch', key=f"{MODULE_KEY}_predict")
    reset_clicked = bcol2.button("\u21BB Reset Form", width='stretch', key=f"{MODULE_KEY}_reset")

    if reset_clicked:
        for k in list(st.session_state.keys()):
            if k.startswith(MODULE_KEY):
                del st.session_state[k]
        st.rerun()

    if not predict_clicked:
        return

    inputs = {
        "Industry": industry, "Country": country, "Company Age": company_age,
        "Number of Employees": employees, "Market Size": market_size, "Customer Count": customer_count,
        "Annual Expenses": annual_expenses, "Funding Raised": funding, "Previous Year Revenue": prev_revenue,
        "Marketing Budget": marketing_budget, "R&D Spending": rd_spend, "Operational Cost": op_cost,
        "Online Presence Score": online_score, "Employee Growth Rate": growth_rate,
    }

    prediction_loading(["Analyzing company fundamentals...", "Running ensemble model...", "Preparing insights..."])

    prediction, confidence = predictor.predict(inputs)
    growth_pct = ((prediction - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
    category = _revenue_category(prediction)
    risk = _risk_level(growth_pct, confidence)
    explain = predictor.explain(inputs)

    display_name = company_name.strip() or "Unnamed Company"
    record_inputs = {"Company Name": display_name, **inputs}

    db.log_module_prediction(
        MODULE_KEY, record_inputs,
        {"prediction": prediction, "category": category, "growth_pct": growth_pct, "risk": risk},
        prediction, confidence,
    )

    st.markdown("---")
    result_headline("Estimated Annual Sales", f"${prediction:,.0f}K", f"{display_name} \u2022 Revenue Category: {category}")

    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_tile("\U0001F4C8", "Growth vs Prior Year", f"{growth_pct:+.1f}%", "vs previous year revenue", "violet")
    with m2: metric_tile("\U0001F3AF", "Confidence", f"{confidence:.1f}%" if confidence else "\u2014", "model certainty", "blue")
    with m3: metric_tile("\U0001F3F7\uFE0F", "Category", category, "revenue tier", "teal")
    with m4: metric_tile("\u26A0\uFE0F", "Risk Level", risk, "forecast risk", "amber")

    st.write("")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(gauge_chart(prediction, 0, max(prediction * 1.6, 1000), "Revenue Gauge ($K)"), width='stretch')
    with ch2:
        st.plotly_chart(confidence_meter(confidence or 0), width='stretch')

    ch3, ch4 = st.columns(2)
    with ch3:
        years = ["Y-2", "Y-1", "This Year", "Forecast"]
        trend_vals = [prev_revenue * 0.85, prev_revenue, prev_revenue, prediction]
        st.plotly_chart(trend_line(years, trend_vals, "Sales Trend", "Revenue ($K)"), width='stretch')
    with ch4:
        st.plotly_chart(donut_breakdown(
            ["Operational Cost", "Marketing", "R&D", "Other Expenses", "Net Margin (est.)"],
            [op_cost, marketing_budget, rd_spend, max(annual_expenses - op_cost, 0),
             max(prediction - op_cost - marketing_budget - rd_spend, 1)],
            "Revenue Breakdown"
        ), width='stretch')

    st.write("")
    insight_card("AI Insights", _build_insights(inputs, prediction, growth_pct, category, risk, explain))
    st.markdown(risk_badge(risk), unsafe_allow_html=True)

    st.write("")
    st.markdown("**Explainable AI**")
    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(feature_contribution_bar(explain["positives"], explain["negatives"], "Top Positive & Negative Features"), width='stretch')
    with e2:
        st.plotly_chart(shap_waterfall(explain["base_value"], explain["all_sorted_by_abs"], prediction, "Prediction Waterfall"), width='stretch')

    st.write("")
    d1, d2 = st.columns(2)
    with d1:
        pdf_bytes = generate_module_pdf(
            "Company Sales Prediction Report", "Estimated Annual Sales", f"${prediction:,.0f}K",
            record_inputs, confidence,
            insights=[p.replace("<b>", "").replace("</b>", "") for p in _build_insights(inputs, prediction, growth_pct, category, risk, explain)],
            extra_metrics={"Revenue Category": category, "Growth %": f"{growth_pct:+.1f}%", "Risk Level": risk},
        )
        st.download_button("\U0001F4C4 Download Prediction Report (PDF)", data=pdf_bytes,
                            file_name="company_sales_prediction.pdf", mime="application/pdf", width='stretch')
    with d2:
        csv_df = pd.DataFrame([{**record_inputs, "Predicted Sales ($K)": prediction, "Growth %": growth_pct, "Category": category}])
        st.download_button("\u2B07\uFE0F Export CSV", data=dataframe_to_csv_bytes(csv_df),
                            file_name="company_sales_prediction.csv", mime="text/csv", width='stretch')
