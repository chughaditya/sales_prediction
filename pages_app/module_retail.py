"""
module_retail.py
--------------------
Retail Sales Prediction module (one of three tabs on the Prediction page).
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

BUNDLE_FILE = "retail_sales_bundle.pkl"
MODULE_KEY = "retail_sales"


def _demand_category(value: float) -> str:
    if value < 1000:
        return "Low Demand"
    if value < 5000:
        return "Moderate Demand"
    if value < 15000:
        return "High Demand"
    return "Peak Demand"


def _risk_level(growth_pct: float, confidence: float) -> str:
    if confidence is None:
        confidence = 60
    if growth_pct < -10 or confidence < 55:
        return "High"
    if growth_pct < 5 or confidence < 75:
        return "Medium"
    return "Low"


def _build_insights(inputs: dict, prediction: float, growth_pct: float, category: str, risk: str, explain: dict):
    top_driver = explain["all_sorted_by_abs"][0][0].split("=")[0] if explain["all_sorted_by_abs"] else "Previous Month Sales"
    points = [
        f"The model's single biggest influence on this forecast was <b>{top_driver}</b>.",
        f"Projected change of <b>{growth_pct:+.1f}%</b> vs last month places this SKU/outlet in the <b>{category}</b> band.",
    ]
    if inputs["Promotion Running"] == "Yes":
        points.append("An active promotion is boosting expected sales &mdash; monitor stock so you don't run out mid-campaign.")
    else:
        points.append("No promotion is currently running; a limited-time discount could lift sales further.")
    if inputs["Discount %"] < 5:
        points.append("Discount level is low relative to typical promotions &mdash; there may be room to test a deeper discount.")
    if inputs["Festival Season"] == "Yes" or inputs["Holiday"] == "Yes":
        points.append("This forecast falls on a festival/holiday period, which typically drives a temporary demand spike.")
    points.append(f"Overall business risk for this forecast is assessed as <b>{risk}</b>.")
    return points


def render():
    predictor = get_specialized_predictor(BUNDLE_FILE)
    db = get_db()

    module_hero("\U0001F6D2", "Retail Sales Prediction", "Forecast product/outlet sales from store, product, and promotional signals.",
                gradient="linear-gradient(120deg, #17C3B2 0%, #0EA5A0 100%)")

    if not predictor.is_ready:
        st.warning("Retail Sales model not found. Run `training/build_specialized_models.py` once to generate it.")
        return

    bundle = predictor.bundle
    ranges = bundle["numeric_ranges"]
    cats = bundle["categories"]

    mode = st.radio("Prediction Mode", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True, key=f"{MODULE_KEY}_mode")

    if mode == "Batch Upload (CSV)":
        render_batch_upload(predictor, MODULE_KEY, "Retail Sales", db)
        return

    glass_card_start()
    st.markdown("**Store & Product Details**")
    store_name = st.text_input(
        "Store / Product Name", value="", placeholder="e.g. Downtown Outlet - Dairy Section",
        key=f"{MODULE_KEY}_name",
        help="Optional label so you can find this forecast again later on the Prediction History page.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        store_type = st.selectbox("Store Type", cats["Store Type"], key=f"{MODULE_KEY}_storetype")
        outlet_size = st.selectbox("Outlet Size", cats["Outlet Size"], key=f"{MODULE_KEY}_outletsize")
        outlet_loc = st.selectbox("Outlet Location Type", cats["Outlet Location Type"], key=f"{MODULE_KEY}_outletloc")
        product_cat = st.selectbox("Product Category", cats["Product Category"], key=f"{MODULE_KEY}_prodcat")
        store_id = st.number_input("Store ID", min_value=1.0, value=float(round(ranges["Store ID"][1])), key=f"{MODULE_KEY}_storeid")
    with c2:
        fat_content = st.selectbox("Product Fat Content", cats["Product Fat Content"], key=f"{MODULE_KEY}_fat")
        promotion = st.selectbox("Promotion Running", cats["Promotion Running"], key=f"{MODULE_KEY}_promo")
        holiday = st.selectbox("Holiday", cats["Holiday"], key=f"{MODULE_KEY}_holiday")
        festival = st.selectbox("Festival Season", cats["Festival Season"], key=f"{MODULE_KEY}_festival")
        weekday = st.selectbox("Weekday", cats["Weekday"], key=f"{MODULE_KEY}_weekday")
        month = st.selectbox("Month", cats["Month"], key=f"{MODULE_KEY}_month")
    with c3:
        product_weight = st.number_input("Product Weight (kg)", min_value=0.0, value=round(ranges["Product Weight"][1], 2), key=f"{MODULE_KEY}_weight")
        product_visibility = st.slider("Product Visibility", 0.0, 0.3, round(ranges["Product Visibility"][1], 3), key=f"{MODULE_KEY}_visibility")
        product_mrp = st.number_input("Product MRP ($)", min_value=0.0, value=round(ranges["Product MRP"][1], 2), key=f"{MODULE_KEY}_mrp")
        stock_qty = st.number_input("Stock Quantity", min_value=0.0, value=float(round(ranges["Stock Quantity"][1])), key=f"{MODULE_KEY}_stock")
        discount_pct = st.slider("Discount %", 0, 50, int(ranges["Discount %"][1]), key=f"{MODULE_KEY}_discount")
        marketing_spend = st.number_input("Marketing Spend ($)", min_value=0.0, value=round(ranges["Marketing Spend"][1], 2), key=f"{MODULE_KEY}_mktspend")
        prev_month_sales = st.number_input("Previous Month Sales ($)", min_value=0.0, value=round(ranges["Previous Month Sales"][1], 2), key=f"{MODULE_KEY}_prevsales")
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
        "Store Type": store_type, "Outlet Size": outlet_size, "Outlet Location Type": outlet_loc,
        "Product Category": product_cat, "Product Fat Content": fat_content, "Promotion Running": promotion,
        "Holiday": holiday, "Festival Season": festival, "Weekday": weekday, "Month": month,
        "Store ID": store_id, "Product Weight": product_weight, "Product Visibility": product_visibility,
        "Product MRP": product_mrp, "Stock Quantity": stock_qty, "Discount %": discount_pct,
        "Marketing Spend": marketing_spend, "Previous Month Sales": prev_month_sales,
    }

    prediction_loading(["Reading store & product signals...", "Running ensemble model...", "Preparing insights..."])

    prediction, confidence = predictor.predict(inputs)
    growth_pct = ((prediction - prev_month_sales) / prev_month_sales * 100) if prev_month_sales > 0 else 0
    category = _demand_category(prediction)
    risk = _risk_level(growth_pct, confidence)
    explain = predictor.explain(inputs)

    display_name = store_name.strip() or "Unnamed Store/Product"
    record_inputs = {"Store/Product Name": display_name, **inputs}

    db.log_module_prediction(
        MODULE_KEY, record_inputs,
        {"prediction": prediction, "category": category, "growth_pct": growth_pct, "risk": risk},
        prediction, confidence,
    )

    st.markdown("---")
    result_headline("Expected Sales", f"${prediction:,.0f}", f"{display_name} \u2022 {category}")

    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_tile("\U0001F4C8", "Change vs Last Month", f"{growth_pct:+.1f}%", "vs previous month sales", "violet")
    with m2: metric_tile("\U0001F3AF", "Confidence", f"{confidence:.1f}%" if confidence else "\u2014", "model certainty", "blue")
    with m3: metric_tile("\U0001F3F7\uFE0F", "Category", category, "demand tier", "teal")
    with m4: metric_tile("\u26A0\uFE0F", "Risk Level", risk, "forecast risk", "amber")

    st.write("")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(gauge_chart(prediction, 0, max(prediction * 1.6, 2000), "Sales Gauge ($)"), width='stretch')
    with ch2:
        st.plotly_chart(confidence_meter(confidence or 0), width='stretch')

    ch3, ch4 = st.columns(2)
    with ch3:
        periods = ["M-2", "M-1", "This Month", "Forecast"]
        trend_vals = [prev_month_sales * 0.9, prev_month_sales, prev_month_sales, prediction]
        st.plotly_chart(trend_line(periods, trend_vals, "Sales Trend", "Sales ($)"), width='stretch')
    with ch4:
        st.plotly_chart(donut_breakdown(
            ["Marketing Spend", "Discount Value (est.)", "Stock Value (est.)", "Net Contribution (est.)"],
            [marketing_spend, product_mrp * stock_qty * (discount_pct / 100),
             product_mrp * stock_qty * 0.02, max(prediction - marketing_spend, 1)],
            "Sales Breakdown"
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
            "Retail Sales Prediction Report", "Expected Sales", f"${prediction:,.0f}",
            record_inputs, confidence,
            insights=[p.replace("<b>", "").replace("</b>", "") for p in _build_insights(inputs, prediction, growth_pct, category, risk, explain)],
            extra_metrics={"Demand Category": category, "Change %": f"{growth_pct:+.1f}%", "Risk Level": risk},
        )
        st.download_button("\U0001F4C4 Download Prediction Report (PDF)", data=pdf_bytes,
                            file_name="retail_sales_prediction.pdf", mime="application/pdf", width='stretch')
    with d2:
        csv_df = pd.DataFrame([{**record_inputs, "Predicted Sales ($)": prediction, "Category": category}])
        st.download_button("\u2B07\uFE0F Export CSV", data=dataframe_to_csv_bytes(csv_df),
                            file_name="retail_sales_prediction.csv", mime="text/csv", width='stretch')
