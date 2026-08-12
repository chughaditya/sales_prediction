"""
module_house.py
------------------
House Price Prediction module (one of three tabs on the Prediction page).
"""

import streamlit as st
import pandas as pd

from prediction.specialized_predictor import get_specialized_predictor
from prediction.module_pdf import generate_module_pdf
from charts.premium_charts import gauge_chart, confidence_meter, donut_breakdown, bar_comparison, feature_contribution_bar, shap_waterfall
from components.glass_ui import module_hero, glass_card_start, glass_card_end, metric_tile, risk_badge, insight_card, prediction_loading, result_headline
from components.batch_upload import render_batch_upload
from utils.helpers import dataframe_to_csv_bytes
from utils.database import get_db

BUNDLE_FILE = "house_price_bundle.pkl"
MODULE_KEY = "house_price"


def _investment_score(inputs: dict, prediction: float) -> int:
    score = 50
    score += 10 if inputs["Condition"] in ("Excellent", "Good") else -5
    score += 8 if inputs["Public Transport Score"] > 6 else 0
    score += 6 if inputs["Distance to City Center"] < 10 else -6
    score += 6 if inputs["Crime Rate"] < 4 else -8
    score += 5 if inputs["Nearby Schools"] > 5 else 0
    score += 5 if inputs["Swimming Pool"] == "Yes" else 0
    score += 5 if inputs["Smart Home Features"] == "Yes" else 0
    return max(1, min(99, score))


def _build_insights(inputs: dict, prediction: float, inv_score: int, explain: dict):
    top_driver = explain["all_sorted_by_abs"][0][0].split("=")[0] if explain["all_sorted_by_abs"] else "Area (sq ft)"
    points = [
        f"<b>{top_driver}</b> had the strongest influence on this valuation.",
        f"Investment Score of <b>{inv_score}/100</b> reflects location quality, condition, and amenities combined.",
    ]
    if inputs["Distance to City Center"] > 20:
        points.append("Property is far from the city center, which typically caps long-term appreciation potential.")
    else:
        points.append("Proximity to the city center supports stronger future appreciation.")
    if inputs["Crime Rate"] > 6:
        points.append("Crime rate in the area is relatively high &mdash; factor this into long-term risk assessment.")
    if inputs["Condition"] in ("Fair", "Poor"):
        points.append("Property condition is below average; renovation could meaningfully lift market value.")
    return points


def render():
    predictor = get_specialized_predictor(BUNDLE_FILE)
    db = get_db()

    module_hero("\U0001F3E0", "House Price Prediction", "Estimate market value from property specs, location, and amenities.",
                gradient="linear-gradient(120deg, #F59E0B 0%, #E08E00 100%)")

    if not predictor.is_ready:
        st.warning("House Price model not found. Run `training/build_specialized_models.py` once to generate it.")
        return

    bundle = predictor.bundle
    ranges = bundle["numeric_ranges"]
    cats = bundle["categories"]

    mode = st.radio("Prediction Mode", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True, key=f"{MODULE_KEY}_mode")

    if mode == "Batch Upload (CSV)":
        render_batch_upload(predictor, MODULE_KEY, "House Price", db)
        return

    glass_card_start()
    st.markdown("**Property Details**")
    property_name = st.text_input(
        "Property Name / Reference", value="", placeholder="e.g. Green Valley Villa #12",
        key=f"{MODULE_KEY}_name",
        help="Optional label so you can find this valuation again later on the Prediction History page.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        area = st.number_input("Area (sq ft)", min_value=100.0, value=round(ranges["Area (sq ft)"][1], 0), key=f"{MODULE_KEY}_area")
        bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=15.0, value=float(round(ranges["Bedrooms"][1])), key=f"{MODULE_KEY}_bed")
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=15.0, value=float(round(ranges["Bathrooms"][1])), key=f"{MODULE_KEY}_bath")
        floors = st.number_input("Floors", min_value=1.0, max_value=10.0, value=float(round(ranges["Floors"][1])), key=f"{MODULE_KEY}_floors")
        garage = st.number_input("Garage Capacity", min_value=0.0, max_value=10.0, value=float(round(ranges["Garage Capacity"][1])), key=f"{MODULE_KEY}_garage")
        year_built = st.number_input("Year Built", min_value=1900.0, max_value=2026.0, value=round(ranges["Year Built"][1], 0), key=f"{MODULE_KEY}_year")
        property_age = st.number_input("Property Age (years)", min_value=0.0, value=round(2026 - ranges["Year Built"][1], 0), key=f"{MODULE_KEY}_age")
    with c2:
        location = st.selectbox("Location", cats["Location"], key=f"{MODULE_KEY}_loc")
        city = st.selectbox("City", cats["City"], key=f"{MODULE_KEY}_city")
        distance = st.slider("Distance to City Center (km)", 0.0, 50.0, round(ranges["Distance to City Center"][1], 1), key=f"{MODULE_KEY}_dist")
        schools = st.number_input("Nearby Schools", min_value=0.0, max_value=20.0, value=float(round(ranges["Nearby Schools"][1])), key=f"{MODULE_KEY}_sch")
        hospitals = st.number_input("Nearby Hospitals", min_value=0.0, max_value=20.0, value=float(round(ranges["Nearby Hospitals"][1])), key=f"{MODULE_KEY}_hosp")
        crime_rate = st.slider("Crime Rate (0-10)", 0.0, 10.0, round(ranges["Crime Rate"][1], 1), key=f"{MODULE_KEY}_crime")
        transport = st.slider("Public Transport Score (0-10)", 0.0, 10.0, round(ranges["Public Transport Score"][1], 1), key=f"{MODULE_KEY}_transport")
    with c3:
        lot_size = st.number_input("Lot Size (sq ft)", min_value=0.0, value=round(ranges["Lot Size"][1], 0), key=f"{MODULE_KEY}_lot")
        condition = st.selectbox("Condition", cats["Condition"], key=f"{MODULE_KEY}_cond")
        furnished = st.selectbox("Furnished", cats["Furnished"], key=f"{MODULE_KEY}_furn")
        pool = st.selectbox("Swimming Pool", cats["Swimming Pool"], key=f"{MODULE_KEY}_pool")
        garden = st.selectbox("Garden", cats["Garden"], key=f"{MODULE_KEY}_garden")
        basement = st.selectbox("Basement", cats["Basement"], key=f"{MODULE_KEY}_basement")
        smart_home = st.selectbox("Smart Home Features", cats["Smart Home Features"], key=f"{MODULE_KEY}_smart")
    glass_card_end()

    bcol1, bcol2 = st.columns([1, 1])
    predict_clicked = bcol1.button("\U0001F52E Predict Price", type="primary", width='stretch', key=f"{MODULE_KEY}_predict")
    reset_clicked = bcol2.button("\u21BB Reset Form", width='stretch', key=f"{MODULE_KEY}_reset")

    if reset_clicked:
        for k in list(st.session_state.keys()):
            if k.startswith(MODULE_KEY):
                del st.session_state[k]
        st.rerun()

    if not predict_clicked:
        return

    inputs = {
        "Area (sq ft)": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms, "Floors": floors,
        "Garage Capacity": garage, "Year Built": year_built, "Property Age": property_age,
        "Location": location, "City": city, "Distance to City Center": distance,
        "Nearby Schools": schools, "Nearby Hospitals": hospitals, "Crime Rate": crime_rate,
        "Public Transport Score": transport, "Lot Size": lot_size, "Condition": condition,
        "Furnished": furnished, "Swimming Pool": pool, "Garden": garden, "Basement": basement,
        "Smart Home Features": smart_home,
    }

    prediction_loading(["Assessing property specs...", "Scoring the neighborhood...", "Valuing the property..."])

    prediction, confidence = predictor.predict(inputs)
    inv_score = _investment_score(inputs, prediction)
    explain = predictor.explain(inputs)
    risk = "Low" if inv_score >= 70 else ("Medium" if inv_score >= 45 else "High")
    appreciation = "Strong" if inv_score >= 70 else ("Moderate" if inv_score >= 45 else "Limited")

    display_name = property_name.strip() or "Unnamed Property"
    record_inputs = {"Property Name": display_name, **inputs}

    db.log_module_prediction(
        MODULE_KEY, record_inputs, {"prediction": prediction, "investment_score": inv_score, "risk": risk}, prediction, confidence,
    )

    st.markdown("---")
    result_headline("Estimated House Price", f"${prediction:,.0f}", f"{display_name} \u2022 {city} \u2022 {location}")

    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_tile("\U0001F4B0", "Price / sq ft", f"${prediction / max(area,1):,.0f}", "value density", "amber")
    with m2: metric_tile("\U0001F3AF", "Confidence", f"{confidence:.1f}%" if confidence else "\u2014", "model certainty", "blue")
    with m3: metric_tile("\U0001F4CA", "Investment Score", f"{inv_score}/100", "location + amenities", "teal")
    with m4: metric_tile("\U0001F4C8", "Appreciation Outlook", appreciation, "next 3-5 years", "violet")

    st.write("")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(gauge_chart(prediction, 0, max(prediction * 1.6, 200000), "Price Gauge ($)"), width='stretch')
    with ch2:
        st.plotly_chart(gauge_chart(inv_score, 0, 100, "Property / Investment Score"), width='stretch')

    ch3, ch4 = st.columns(2)
    with ch3:
        comp_labels = ["This Property", "Neighborhood Avg (est.)", "City Avg (est.)"]
        comp_vals = [prediction, prediction * 0.92, prediction * 0.8]
        st.plotly_chart(bar_comparison(comp_labels, comp_vals, "Price Comparison", "Price ($)"), width='stretch')
    with ch4:
        st.plotly_chart(donut_breakdown(
            ["Location Score", "Condition Score", "Amenity Score", "Transport Score"],
            [max(10 - distance / 5, 1), {"Excellent": 10, "Good": 8, "Fair": 5, "Poor": 2}[condition],
             (pool == "Yes") * 3 + (garden == "Yes") * 3 + (smart_home == "Yes") * 4 + 1, transport + 1],
            "Feature Contribution"
        ), width='stretch')

    st.write("")
    insight_card("AI Insights", _build_insights(inputs, prediction, inv_score, explain))
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
            "House Price Prediction Report", "Estimated House Price", f"${prediction:,.0f}",
            record_inputs, confidence,
            insights=[p.replace("<b>", "").replace("</b>", "") for p in _build_insights(inputs, prediction, inv_score, explain)],
            extra_metrics={"Investment Score": f"{inv_score}/100", "Risk Level": risk, "Appreciation Outlook": appreciation},
        )
        st.download_button("\U0001F4C4 Download Prediction Report (PDF)", data=pdf_bytes,
                            file_name="house_price_prediction.pdf", mime="application/pdf", width='stretch')
    with d2:
        csv_df = pd.DataFrame([{**record_inputs, "Predicted Price ($)": prediction, "Investment Score": inv_score}])
        st.download_button("\u2B07\uFE0F Export CSV", data=dataframe_to_csv_bytes(csv_df),
                            file_name="house_price_prediction.csv", mime="text/csv", width='stretch')
