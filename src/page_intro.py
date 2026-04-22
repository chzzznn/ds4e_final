import streamlit as st
import numpy as np
import plotly.graph_objects as go

from data_loader import load_data


def render():
    df = load_data()

    st.markdown("""
    <div class='hero-section'>
        <div class='tag'>📍 Business Intelligence · Education</div>
        <div class='hero-title'>Can Education<br><span class='accent'>Predict Unemployment?</span></div>
        <div class='hero-subtitle'>
            We analyze global education indicators across 202 countries to understand
            how literacy, school completion, and enrollment rates drive unemployment —
            and build a model to predict it.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    valid_unemp = df["Unemployment_Rate"].replace(0, np.nan).dropna()
    with col1:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>202</div>
            <div class='metric-label'>Countries</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>27</div>
            <div class='metric-label'>Features</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{valid_unemp.mean():.1f}%</div>
            <div class='metric-label'>Avg Unemployment</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{valid_unemp.max():.1f}%</div>
            <div class='metric-label'>Max Unemployment</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("<div class='section-title'>🎯 The Business Problem</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#A0A0C0; line-height:1.85; font-size:0.95rem;'>
        Unemployment is one of the most pressing socioeconomic challenges globally. While many factors drive it,
        <strong style='color:#C0C0E0;'>education is widely considered the most controllable lever</strong> for governments and NGOs.
        <br><br>
        This app addresses a key question for policymakers: <em>which education interventions are most predictive
        of unemployment reduction?</em> By building regression models on 202 countries, we can surface
        actionable insights on where to invest in education systems.
        <br><br>
        The model predicts <strong style='color:#7C9EFF;'>Unemployment Rate (%)</strong> using indicators
        like school completion rates, proficiency scores, literacy rates, tertiary enrollment, and birth rate.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-title'>📋 Key Features</div>", unsafe_allow_html=True)
        features_info = {
            "Completion Rates": "Primary, Lower & Upper Secondary (M/F)",
            "Literacy Rates": "Youth 15–24 (Male & Female)",
            "Proficiency Scores": "Reading & Math at grade levels",
            "Enrollment Rates": "Primary & Tertiary (gross %)",
            "Birth Rate": "Births per 1,000 population",
        }
        for k, v in features_info.items():
            st.markdown(f"""
            <div style='background:#13162B; border:1px solid #22264A; border-radius:10px;
                        padding:10px 16px; margin-bottom:8px;'>
                <span style='color:#7C9EFF; font-weight:600; font-size:0.85rem;'>{k}</span><br>
                <span style='color:#6868A0; font-size:0.8rem;'>{v}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🗂️ Dataset Preview</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>First 10 rows of the raw dataset</div>", unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, height=320)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>📐 Dataset Shape</div>", unsafe_allow_html=True)
        st.code(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}", language=None)
        st.markdown("<div class='section-title' style='font-size:1.1rem; margin-top:16px;'>📊 Statistical Summary</div>", unsafe_allow_html=True)
        st.dataframe(df.describe().round(2), use_container_width=True)

    with col_b:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>🔍 Missing Values (zeros as proxy)</div>", unsafe_allow_html=True)
        zero_counts = (df.select_dtypes(include=np.number) == 0).sum().sort_values(ascending=False).head(15)
        fig_miss = go.Figure(go.Bar(
            x=zero_counts.values, y=zero_counts.index,
            orientation="h",
            marker=dict(color="#7C9EFF", opacity=0.8)
        ))
        fig_miss.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9090B0"),
            xaxis=dict(gridcolor="#1E2140"),
            yaxis=dict(gridcolor="#1E2140"),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_miss, use_container_width=True)
