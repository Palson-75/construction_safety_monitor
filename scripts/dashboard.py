
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs" / "integrated_monitor"


# ============================================================
# 2. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Construction Site Safety Dashboard",
    page_icon="🏗️",
    layout="wide"
)


# ============================================================
# 3. HEADER
# ============================================================

st.title("🏗️ Construction Site Safety Monitoring System")

st.write(
    "AI-based monitoring of construction workers, "
    "personal protective equipment (PPE), and safety status."
)

st.divider()


# ============================================================
# 4. FIND AVAILABLE REPORTS
# ============================================================

report_files = sorted(
    OUTPUT_DIR.glob("*_risk_report.csv")
)

if not report_files:
    st.error(
        "No risk reports found. "
        "Please run integrated_safety_monitor.py first."
    )
    st.stop()


# ============================================================
# 5. SELECT PROCESSED VIDEO REPORT
# ============================================================

st.subheader("Select Processed Report")

selected_report = st.selectbox(
    "Choose a processed video report",
    report_files,
    format_func=lambda path: path.name.replace(
        "_risk_report.csv", ""
    )
)

video_name = selected_report.name.replace(
    "_risk_report.csv", ""
)


# ============================================================
# 6. LOAD WORKER RISK REPORT
# ============================================================

try:
    df = pd.read_csv(selected_report)

except Exception as error:
    st.error(f"Unable to read the risk report: {error}")
    st.stop()


if df.empty:
    st.warning("The selected report contains no worker records.")
    st.stop()


if "risk_level" not in df.columns:
    st.error("The risk_level column is missing from the report.")
    st.stop()


# ============================================================
# 7. WORKER RISK SUMMARY
# ============================================================

st.subheader("Worker Risk Summary")

risk_counts = (
    df["risk_level"]
    .astype(str)
    .str.strip()
    .str.title()
    .value_counts()
    .reindex(["Low", "High"], fill_value=0)
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Low Risk Workers",
        value=int(risk_counts["Low"])
    )

with col2:
    st.metric(
        label="High Risk Workers",
        value=int(risk_counts["High"])
    )


# ============================================================
# 8. WORKER RISK DISTRIBUTION CHART
# ============================================================

st.subheader("Worker Risk Distribution")

risk_chart_df = pd.DataFrame({
    "Risk Level": ["Low Risk", "High Risk"],
    "Workers": [
        int(risk_counts["Low"]),
        int(risk_counts["High"])
    ]
})

fig = px.bar(
    risk_chart_df,
    x="Risk Level",
    y="Workers",
    color="Risk Level",
    text="Workers",
    title="Low Risk vs High Risk Workers"
)

fig.update_traces(
    textposition="outside"
)

fig.update_layout(
    showlegend=False,
    xaxis_title="Risk Category",
    yaxis_title="Number of Workers"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 9. PPE DETECTION SUMMARY
# ============================================================

st.divider()

st.subheader("PPE Detection Summary")

ppe_categories = [
    "helmet",
    "vest",
    "boots",
    "gloves",
    "goggles"
]

available_categories = [
    category
    for category in ppe_categories
    if category in df.columns
]

if available_categories:

    ppe_summary = []

    for category in available_categories:

        counts = (
            df[category]
            .astype(str)
            .str.upper()
            .value_counts()
        )

        ppe_summary.append({
            "PPE Category": category.replace(
                "_", " "
            ).title(),

            "Detected": int(
                counts.get("DETECTED", 0)
            ),

            "Negative Detection": int(
                counts.get("NEGATIVE", 0)
            ),

            "Conflicting Evidence": int(
                counts.get("CONFLICT", 0)
            )
        })

    ppe_df = pd.DataFrame(ppe_summary)

    st.dataframe(
        ppe_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "No PPE category columns were found in the report."
    )


# ============================================================
# 10. DOWNLOAD WORKER RISK REPORT
# ============================================================

st.divider()

st.subheader("Download Worker Risk Report")

csv_data = df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download CSV Report",
    data=csv_data,
    file_name=f"{video_name}_risk_report.csv",
    mime="text/csv"
)


# ============================================================
# 11. DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Note: Risk categories are derived from PPE detection rules. "
    "They are not independently validated predictions of accidents "
    "or real-world injury risk. UNKNOWN indicates insufficient "
    "detection evidence."
)
