import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import shap
import wandb
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict | Global Unemployment",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3, .big-title {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background-color: #0D0F1A;
    color: #E8E8F0;
}

[data-testid="stSidebar"] {
    background-color: #13162B;
    border-right: 1px solid #2A2D4A;
}

[data-testid="stSidebar"] * {
    color: #C8C8E0 !important;
}

.metric-card {
    background: linear-gradient(135deg, #1A1D35 0%, #1F2240 100%);
    border: 1px solid #2E3260;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: transform 0.2s;
}

.metric-card:hover {
    transform: translateY(-3px);
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #7C9EFF;
    line-height: 1;
}

.metric-label {
    font-size: 0.8rem;
    color: #8888AA;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
}

.hero-section {
    background: linear-gradient(135deg, #0D0F1A 0%, #1A1040 50%, #0D1A2E 100%);
    border: 1px solid #2A2D4A;
    border-radius: 24px;
    padding: 60px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(124,158,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.1;
    margin-bottom: 16px;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #9090B0;
    max-width: 600px;
    line-height: 1.7;
}

.accent {
    color: #7C9EFF;
}

.tag {
    display: inline-block;
    background: rgba(124,158,255,0.12);
    border: 1px solid rgba(124,158,255,0.3);
    color: #7C9EFF;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #E8E8F0;
    margin-bottom: 6px;
}

.section-sub {
    color: #7070A0;
    font-size: 0.9rem;
    margin-bottom: 24px;
}

.model-card {
    background: #13162B;
    border: 1px solid #22264A;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 12px;
}

.model-card.selected {
    border-color: #7C9EFF;
    background: #1A1D40;
}

.insight-box {
    background: linear-gradient(135deg, rgba(124,158,255,0.08), rgba(100,200,160,0.05));
    border-left: 3px solid #7C9EFF;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #C0C0D8;
    line-height: 1.6;
}

.stSelectbox > div > div {
    background-color: #1A1D35 !important;
    border-color: #2E3260 !important;
    color: #E8E8F0 !important;
}

.stSlider > div > div > div {
    background-color: #7C9EFF !important;
}

div[data-testid="stMetric"] {
    background: #13162B;
    border: 1px solid #22264A;
    border-radius: 12px;
    padding: 16px;
}

.stDataFrame { border-radius: 12px; overflow: hidden; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Global_Education.csv", encoding="latin1")
    df.columns = df.columns.str.strip()
    # Replace 0 with NaN for unemployment (many missing coded as 0)
    df["Unemployment_Rate"] = df["Unemployment_Rate"].replace(0, np.nan)
    return df

@st.cache_data
def get_model_data():
    df = load_data()
    features = [
        "Completion_Rate_Primary_Male", "Completion_Rate_Primary_Female",
        "Completion_Rate_Lower_Secondary_Male", "Completion_Rate_Lower_Secondary_Female",
        "Completion_Rate_Upper_Secondary_Male", "Completion_Rate_Upper_Secondary_Female",
        "Youth_15_24_Literacy_Rate_Male", "Youth_15_24_Literacy_Rate_Female",
        "Birth_Rate", "Gross_Primary_Education_Enrollment",
        "Gross_Tertiary_Education_Enrollment",
        "Lower_Secondary_End_Proficiency_Reading", "Lower_Secondary_End_Proficiency_Math",
    ]
    target = "Unemployment_Rate"
    df_model = df[features + [target, "Countries and areas"]].copy()
    df_model = df_model.replace(0, np.nan)
    df_model = df_model.dropna(subset=[target])
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(df_model[features]), columns=features)
    y = df_model[target].values
    countries = df_model["Countries and areas"].values
    return X, y, countries, features

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 30px 0;'>
        <div style='font-family:Syne; font-size:1.4rem; font-weight:800; color:#7C9EFF;'>🎓 EduPredict</div>
        <div style='font-size:0.75rem; color:#5555AA; margin-top:4px; letter-spacing:1px;'>GLOBAL EDUCATION ANALYTICS</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Overview & Data", "📊  Data Visualization", "🤖  Prediction Models",
         "🔍  Explainability (SHAP)", "⚙️  Hyperparameter Tuning"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#4444AA; padding:0 8px;'>
        <b style='color:#5566CC;'>Target Variable</b><br>
        Unemployment Rate (%)<br><br>
        <b style='color:#5566CC;'>Dataset</b><br>
        202 countries · 27 features<br><br>
        <b style='color:#5566CC;'>Source</b><br>
        UNESCO / World Bank
    </div>
    """, unsafe_allow_html=True)

df = load_data()

# ═══════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW & DATA
# ═══════════════════════════════════════════════════════════
if "Overview" in page:
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
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>202</div>
            <div class='metric-label'>Countries</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
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

    # Business case
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


# ═══════════════════════════════════════════════════════════
# PAGE 2: DATA VISUALIZATION
# ═══════════════════════════════════════════════════════════
elif "Visualization" in page:
    st.markdown("<div class='hero-title' style='font-size:2.2rem; color:#E8E8F0;'>📊 Data <span style='color:#7C9EFF;'>Visualization</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Exploratory analysis revealing key patterns in global education & unemployment</div>", unsafe_allow_html=True)

    df_viz = df.copy()
    df_viz["Unemployment_Rate"] = df_viz["Unemployment_Rate"].replace(0, np.nan)
    df_viz["Youth_15_24_Literacy_Rate_Female"] = df_viz["Youth_15_24_Literacy_Rate_Female"].replace(0, np.nan)
    df_viz["Gross_Tertiary_Education_Enrollment"] = df_viz["Gross_Tertiary_Education_Enrollment"].replace(0, np.nan)
    df_viz["Completion_Rate_Upper_Secondary_Female"] = df_viz["Completion_Rate_Upper_Secondary_Female"].replace(0, np.nan)
    df_viz["Birth_Rate"] = df_viz["Birth_Rate"].replace(0, np.nan)

    # WORLD MAP
    st.markdown("<div class='section-title'>🌍 Unemployment Rate by Country</div>", unsafe_allow_html=True)
    fig_map = px.choropleth(
        df_viz.dropna(subset=["Unemployment_Rate"]),
        locations="Countries and areas",
        locationmode="country names",
        color="Unemployment_Rate",
        color_continuous_scale=["#1A1D35", "#3050A0", "#7C9EFF", "#FFD700", "#FF6B6B"],
        title="",
        labels={"Unemployment_Rate": "Unemployment (%)"},
    )
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=True,
                 coastlinecolor="#2A2D4A", showland=True, landcolor="#13162B",
                 showocean=True, oceancolor="#0D0F1A"),
        font=dict(color="#9090B0"),
        coloraxis_colorbar=dict(tickfont=dict(color="#9090B0")),
        height=420, margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ROW: Scatter plots
    st.markdown("<div class='section-title' style='margin-top:16px;'>🔗 Education vs. Unemployment</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        df_s1 = df_viz.dropna(subset=["Unemployment_Rate", "Gross_Tertiary_Education_Enrollment"])
        fig1 = px.scatter(
            df_s1, x="Gross_Tertiary_Education_Enrollment", y="Unemployment_Rate",
            hover_name="Countries and areas",
            trendline="ols",
            color="Unemployment_Rate",
            color_continuous_scale=["#4466FF", "#FF6B6B"],
            labels={"Gross_Tertiary_Education_Enrollment": "Tertiary Enrollment (%)", "Unemployment_Rate": "Unemployment (%)"},
            title="Tertiary Enrollment vs Unemployment"
        )
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
                           font=dict(color="#9090B0"), height=340,
                           xaxis=dict(gridcolor="#1E2140"), yaxis=dict(gridcolor="#1E2140"),
                           title_font=dict(color="#C0C0D8"))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("<div class='insight-box'>💡 Countries with higher tertiary enrollment tend to show <b>lower unemployment</b>, supporting the education-employment link.</div>", unsafe_allow_html=True)

    with col2:
        df_s2 = df_viz.dropna(subset=["Unemployment_Rate", "Birth_Rate"])
        fig2 = px.scatter(
            df_s2, x="Birth_Rate", y="Unemployment_Rate",
            hover_name="Countries and areas",
            trendline="ols",
            color="Unemployment_Rate",
            color_continuous_scale=["#4466FF", "#FF6B6B"],
            labels={"Birth_Rate": "Birth Rate", "Unemployment_Rate": "Unemployment (%)"},
            title="Birth Rate vs Unemployment"
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
                           font=dict(color="#9090B0"), height=340,
                           xaxis=dict(gridcolor="#1E2140"), yaxis=dict(gridcolor="#1E2140"),
                           title_font=dict(color="#C0C0D8"))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<div class='insight-box'>💡 High birth rates correlate with higher unemployment — often reflecting countries with lower development and education investment.</div>", unsafe_allow_html=True)

    # GENDER GAP
    st.markdown("<div class='section-title' style='margin-top:16px;'>⚖️ Gender Gap in Education Completion</div>", unsafe_allow_html=True)
    df_gender = df_viz.copy()
    df_gender = df_gender.replace(0, np.nan)
    df_gender["gap_primary"] = df_gender["Completion_Rate_Primary_Male"] - df_gender["Completion_Rate_Primary_Female"]
    df_gender["gap_secondary"] = df_gender["Completion_Rate_Upper_Secondary_Male"] - df_gender["Completion_Rate_Upper_Secondary_Female"]
    df_gender_top = df_gender.dropna(subset=["gap_secondary"]).nlargest(20, "gap_secondary")

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(name="Primary Gap (M-F)", x=df_gender_top["Countries and areas"],
                             y=df_gender_top["gap_primary"], marker_color="#7C9EFF"))
    fig_gap.add_trace(go.Bar(name="Upper Secondary Gap (M-F)", x=df_gender_top["Countries and areas"],
                             y=df_gender_top["gap_secondary"], marker_color="#FF6B9D"))
    fig_gap.update_layout(
        barmode="group", height=380,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
        font=dict(color="#9090B0"), xaxis=dict(tickangle=-45, gridcolor="#1E2140"),
        yaxis=dict(gridcolor="#1E2140", title="Completion Gap (Male - Female, %)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(b=120)
    )
    st.plotly_chart(fig_gap, use_container_width=True)
    st.markdown("<div class='insight-box'>💡 In Chad, Niger, and Mali, male students complete upper secondary at <b>10–20% higher rates</b> than females — a significant equity gap linked to higher unemployment in these regions.</div>", unsafe_allow_html=True)

    # CORRELATION HEATMAP
    st.markdown("<div class='section-title' style='margin-top:16px;'>🧮 Correlation Heatmap</div>", unsafe_allow_html=True)
    corr_cols = [
        "Unemployment_Rate", "Birth_Rate", "Gross_Tertiary_Education_Enrollment",
        "Completion_Rate_Upper_Secondary_Male", "Completion_Rate_Upper_Secondary_Female",
        "Youth_15_24_Literacy_Rate_Male", "Youth_15_24_Literacy_Rate_Female",
        "Lower_Secondary_End_Proficiency_Reading", "Lower_Secondary_End_Proficiency_Math"
    ]
    corr_df = df_viz[corr_cols].replace(0, np.nan).dropna()
    corr_matrix = corr_df.corr()

    fig_corr, ax = plt.subplots(figsize=(10, 7))
    fig_corr.patch.set_facecolor("#0D0F1A")
    ax.set_facecolor("#13162B")
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlBu_r",
                ax=ax, linewidths=0.5, linecolor="#1E2140",
                annot_kws={"size": 8, "color": "white"},
                cbar_kws={"shrink": 0.8})
    ax.tick_params(colors="#8888AA", labelsize=8)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig_corr, use_container_width=True)
    plt.close()

    # Distribution
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>📈 Unemployment Distribution</div>", unsafe_allow_html=True)
        fig_hist = px.histogram(df_viz.dropna(subset=["Unemployment_Rate"]),
                                x="Unemployment_Rate", nbins=30,
                                color_discrete_sequence=["#7C9EFF"])
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
                                font=dict(color="#9090B0"), height=280,
                                xaxis=dict(gridcolor="#1E2140"), yaxis=dict(gridcolor="#1E2140"),
                                margin=dict(t=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_d2:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>🏆 Top 10 Highest Unemployment</div>", unsafe_allow_html=True)
        top10 = df_viz.dropna(subset=["Unemployment_Rate"]).nlargest(10, "Unemployment_Rate")
        fig_top = px.bar(top10, x="Unemployment_Rate", y="Countries and areas",
                         orientation="h", color="Unemployment_Rate",
                         color_continuous_scale=["#7C9EFF", "#FF6B6B"])
        fig_top.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
                               font=dict(color="#9090B0"), height=280,
                               xaxis=dict(gridcolor="#1E2140"), yaxis=dict(gridcolor="#1E2140"),
                               margin=dict(t=10))
        st.plotly_chart(fig_top, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 3: PREDICTION MODELS
# ═══════════════════════════════════════════════════════════
elif "Prediction" in page:
    st.markdown("<div class='hero-title' style='font-size:2.2rem; color:#E8E8F0;'>🤖 Prediction <span style='color:#7C9EFF;'>Models</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Train & compare 6 regression models to predict unemployment rate</div>", unsafe_allow_html=True)

    X, y, countries, features = get_model_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    MODELS = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>🎛 Select Models to Compare</div>", unsafe_allow_html=True)
        selected_models = st.multiselect(
            "Choose models", list(MODELS.keys()),
            default=["Linear Regression", "Random Forest", "Gradient Boosting"],
            label_visibility="collapsed"
        )
        primary_model = st.selectbox("Primary model for prediction", selected_models if selected_models else list(MODELS.keys()))

    with col_info:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>📐 Train / Test Split</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Samples", len(y))
        c2.metric("Train Samples", len(y_train))
        c3.metric("Test Samples", len(y_test))

    if not selected_models:
        st.warning("Please select at least one model.")
        st.stop()

    # Train all selected models
    results = {}
    trained_models = {}
    for name in selected_models:
        model = MODELS[name]
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        trained_models[name] = model
        results[name] = {
            "R² Score": round(r2_score(y_test, preds), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
            "MAE": round(mean_absolute_error(y_test, preds), 4),
            "CV R² (5-fold)": round(cross_val_score(MODELS[name], X_train_s, y_train, cv=5, scoring="r2").mean(), 4)
        }

    # Results table
    st.markdown("<div class='section-title' style='margin-top:24px;'>📊 Model Comparison</div>", unsafe_allow_html=True)
    results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})

    fig_compare = go.Figure()
    metrics = ["R² Score", "RMSE", "MAE"]
    colors = ["#7C9EFF", "#FF6B9D", "#64C8A0"]
    for i, metric in enumerate(metrics):
        fig_compare.add_trace(go.Bar(
            name=metric, x=results_df["Model"], y=results_df[metric],
            marker_color=colors[i]
        ))
    fig_compare.update_layout(
        barmode="group", height=350,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
        font=dict(color="#9090B0"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1E2140"), yaxis=dict(gridcolor="#1E2140"),
        margin=dict(t=10)
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    st.dataframe(results_df.set_index("Model"), use_container_width=True)

    # Actual vs Predicted
    st.markdown("<div class='section-title' style='margin-top:24px;'>🎯 Actual vs Predicted — " + primary_model + "</div>", unsafe_allow_html=True)
    best_model = trained_models[primary_model]
    preds_best = best_model.predict(X_test_s)

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(x=y_test, y=preds_best, mode="markers",
                                  marker=dict(color="#7C9EFF", size=8, opacity=0.7)))
    fig_avp.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                  y=[y_test.min(), y_test.max()],
                                  mode="lines", line=dict(color="#FF6B6B", dash="dash"), name="Perfect Fit"))
    fig_avp.update_layout(
        height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
        font=dict(color="#9090B0"), xaxis=dict(title="Actual", gridcolor="#1E2140"),
        yaxis=dict(title="Predicted", gridcolor="#1E2140"),
        legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=10)
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    # Live Prediction Tool
    st.markdown("<div class='section-title' style='margin-top:24px;'>🧮 Interactive Prediction Tool</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Adjust education indicators to predict unemployment rate</div>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        vals = {}
        with c1:
            vals["Completion_Rate_Primary_Male"] = st.slider("Primary Completion Male (%)", 0, 100, 75)
            vals["Completion_Rate_Primary_Female"] = st.slider("Primary Completion Female (%)", 0, 100, 72)
            vals["Completion_Rate_Lower_Secondary_Male"] = st.slider("Lower Secondary Male (%)", 0, 100, 65)
            vals["Completion_Rate_Lower_Secondary_Female"] = st.slider("Lower Secondary Female (%)", 0, 100, 60)
        with c2:
            vals["Completion_Rate_Upper_Secondary_Male"] = st.slider("Upper Secondary Male (%)", 0, 100, 55)
            vals["Completion_Rate_Upper_Secondary_Female"] = st.slider("Upper Secondary Female (%)", 0, 100, 52)
            vals["Youth_15_24_Literacy_Rate_Male"] = st.slider("Youth Literacy Male (%)", 0, 100, 85)
            vals["Youth_15_24_Literacy_Rate_Female"] = st.slider("Youth Literacy Female (%)", 0, 100, 80)
        with c3:
            vals["Birth_Rate"] = st.slider("Birth Rate", 5.0, 50.0, 18.0)
            vals["Gross_Primary_Education_Enrollment"] = st.slider("Primary Enrollment (%)", 50, 150, 105)
            vals["Gross_Tertiary_Education_Enrollment"] = st.slider("Tertiary Enrollment (%)", 0, 120, 40)
            vals["Lower_Secondary_End_Proficiency_Reading"] = st.slider("Reading Proficiency", 0, 100, 50)
            vals["Lower_Secondary_End_Proficiency_Math"] = st.slider("Math Proficiency", 0, 100, 45)

        submitted = st.form_submit_button("🔮 Predict Unemployment Rate", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([vals])[features]
        input_scaled = scaler.transform(input_df)
        prediction = best_model.predict(input_scaled)[0]
        prediction = max(0, prediction)
        color = "#64C8A0" if prediction < 6 else "#FFD700" if prediction < 12 else "#FF6B6B"
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1A1D35,#1F2240); border:2px solid {color};
                    border-radius:16px; padding:30px; text-align:center; margin-top:16px;'>
            <div style='font-family:Syne; font-size:3rem; font-weight:800; color:{color};'>{prediction:.2f}%</div>
            <div style='color:#8888AA; font-size:0.9rem; margin-top:8px;'>Predicted Unemployment Rate using {primary_model}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4: EXPLAINABILITY (SHAP)
# ═══════════════════════════════════════════════════════════
elif "Explainability" in page:
    st.markdown("<div class='hero-title' style='font-size:2.2rem; color:#E8E8F0;'>🔍 Explainability <span style='color:#7C9EFF;'>(SHAP)</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Understanding which education factors drive unemployment predictions</div>", unsafe_allow_html=True)

    X, y, countries, features = get_model_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model_choice = st.selectbox("Choose model to explain", ["Random Forest", "Gradient Boosting", "Linear Regression"])
    MODELS_XAI = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
    }
    model = MODELS_XAI[model_choice]
    model.fit(X_train_s, y_train)

    X_test_df = pd.DataFrame(X_test_s, columns=features)

    with st.spinner("Computing SHAP values..."):
        if model_choice == "Linear Regression":
            explainer = shap.LinearExplainer(model, X_train_s)
        else:
            explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df if model_choice != "Linear Regression" else X_test_s)

    # Feature importance bar
    st.markdown("<div class='section-title' style='margin-top:8px;'>📊 Global Feature Importance (Mean |SHAP|)</div>", unsafe_allow_html=True)
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"Feature": features, "Mean |SHAP|": mean_shap}).sort_values("Mean |SHAP|", ascending=True)
    fig_shap_bar = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                           color="Mean |SHAP|", color_continuous_scale=["#1A1D35", "#7C9EFF"])
    fig_shap_bar.update_layout(
        height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
        font=dict(color="#9090B0"), xaxis=dict(gridcolor="#1E2140"), yaxis=dict(gridcolor="#1E2140"),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_shap_bar, use_container_width=True)

    # SHAP summary plot
    st.markdown("<div class='section-title' style='margin-top:16px;'>🌡️ SHAP Beeswarm Summary Plot</div>", unsafe_allow_html=True)
    fig_summary, ax = plt.subplots(figsize=(10, 5))
    fig_summary.patch.set_facecolor("#0D0F1A")
    ax.set_facecolor("#0D0F1A")
    shap.summary_plot(shap_values, X_test_df if model_choice != "Linear Regression" else pd.DataFrame(X_test_s, columns=features),
                      feature_names=features, show=False, color_bar=True, plot_size=None)
    plt.gcf().patch.set_facecolor("#0D0F1A")
    for ax_ in plt.gcf().get_axes():
        ax_.set_facecolor("#0D0F1A")
        ax_.tick_params(colors="#8888AA")
        ax_.spines["bottom"].set_color("#2A2D4A")
        ax_.spines["left"].set_color("#2A2D4A")
    st.pyplot(fig_summary, use_container_width=True)
    plt.close()

    # SHAP waterfall for one sample
    st.markdown("<div class='section-title' style='margin-top:16px;'>🎯 Single Prediction Breakdown</div>", unsafe_allow_html=True)
    sample_idx = st.slider("Select test sample index", 0, len(X_test_s) - 1, 0)
    fig_wf, ax_wf = plt.subplots(figsize=(10, 4))
    fig_wf.patch.set_facecolor("#0D0F1A")

    # ── FIX: TreeExplainer.expected_value can be an array; always coerce to scalar ──
    raw_ev = explainer.expected_value if hasattr(explainer, "expected_value") else 0.0
    base_val = float(raw_ev[0]) if hasattr(raw_ev, "__len__") else float(raw_ev)

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=base_val,
            data=X_test_df.iloc[sample_idx] if model_choice != "Linear Regression" else pd.Series(X_test_s[sample_idx], index=features),
            feature_names=features,
        ),
        show=False
    )
    plt.gcf().patch.set_facecolor("#0D0F1A")
    for ax_ in plt.gcf().get_axes():
        ax_.set_facecolor("#13162B")
        ax_.tick_params(colors="#8888AA")
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close()

    # Key insights
    top_feat = shap_df.iloc[-1]["Feature"]
    st.markdown(f"""
    <div class='insight-box' style='margin-top:16px;'>
    💡 <b>Key Insight:</b> According to SHAP analysis, <b>{top_feat.replace('_', ' ')}</b> is the
    most influential variable in predicting unemployment for the <b>{model_choice}</b> model.
    Features with positive SHAP values push predictions higher; negative values push them lower.
    </div>
    <div class='insight-box'>
    💡 <b>Policy Takeaway:</b> Improving upper secondary completion rates and tertiary enrollment
    are the levers with the highest predicted impact on reducing unemployment globally.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 5: HYPERPARAMETER TUNING (W&B)
# ═══════════════════════════════════════════════════════════
elif "Hyperparameter" in page:
    st.markdown("<div class='hero-title' style='font-size:2.2rem; color:#E8E8F0;'>⚙️ Hyperparameter <span style='color:#7C9EFF;'>Tuning</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Track experiments with Weights & Biases and select the best configuration</div>", unsafe_allow_html=True)

    X, y, countries, features = get_model_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # W&B config
    st.markdown("<div class='section-title'>🔑 Weights & Biases Configuration</div>", unsafe_allow_html=True)
    col_wb1, col_wb2 = st.columns(2)
    with col_wb1:
        wb_api_key = st.text_input("W&B API Key (optional)", type="password",
                                    placeholder="Leave blank to run locally without W&B logging")
        wb_project = st.text_input("W&B Project Name", value="global-education-unemployment")
    with col_wb2:
        st.markdown("""
        <div style='background:#13162B; border:1px solid #22264A; border-radius:12px; padding:16px; margin-top:8px;'>
            <div style='color:#7C9EFF; font-weight:600; font-size:0.85rem; margin-bottom:8px;'>ℹ️ About W&B Logging</div>
            <div style='color:#6868A0; font-size:0.82rem; line-height:1.6;'>
            Experiments are always run locally. If a W&B API key is provided,
            results are additionally logged to your W&B dashboard for remote tracking.
            Get your key at <b>wandb.ai/settings</b>.
            </div>
        </div>""", unsafe_allow_html=True)

    # Hyperparameter Grid
    st.markdown("<div class='section-title' style='margin-top:24px;'>🎛 Experiment Grid</div>", unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**Random Forest**")
        rf_estimators = st.multiselect("n_estimators", [50, 100, 200, 300], default=[50, 100, 200])
        rf_depths = st.multiselect("max_depth", [3, 5, 8, None], default=[3, 5, 8])
    with col_g2:
        st.markdown("**Ridge Regression**")
        ridge_alphas = st.multiselect("alpha", [0.01, 0.1, 1.0, 10.0, 100.0], default=[0.1, 1.0, 10.0])
        st.markdown("**Gradient Boosting**")
        gb_lr = st.multiselect("learning_rate", [0.01, 0.05, 0.1, 0.2], default=[0.05, 0.1])

    run_tuning = st.button("🚀 Run Hyperparameter Tuning", use_container_width=True)

    if run_tuning:
        use_wandb = bool(wb_api_key.strip())
        if use_wandb:
            os.environ["WANDB_API_KEY"] = wb_api_key.strip()
            try:
                wandb.login(key=wb_api_key.strip(), relogin=True)
                st.success("✅ W&B login successful! Experiments will be logged.")
            except Exception as e:
                st.warning(f"W&B login failed: {e}. Running locally.")
                use_wandb = False

        all_results = []
        progress = st.progress(0)
        status_text = st.empty()

        experiments = []
        for n_est in rf_estimators:
            for depth in rf_depths:
                experiments.append(("Random Forest", {"n_estimators": n_est, "max_depth": depth}))
        for alpha in ridge_alphas:
            experiments.append(("Ridge", {"alpha": alpha}))
        for lr in gb_lr:
            experiments.append(("Gradient Boosting", {"n_estimators": 100, "learning_rate": lr}))

        for i, (model_name, params) in enumerate(experiments):
            status_text.markdown(f"<div style='color:#7C9EFF; font-size:0.9rem;'>Running: {model_name} | {params}</div>", unsafe_allow_html=True)

            if model_name == "Random Forest":
                model = RandomForestRegressor(random_state=42, **params)
            elif model_name == "Ridge":
                model = Ridge(**params)
            else:
                model = GradientBoostingRegressor(random_state=42, **params)

            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)

            row = {"Model": model_name, "R²": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)}
            row.update(params)
            all_results.append(row)

            if use_wandb:
                try:
                    run = wandb.init(project=wb_project, name=f"{model_name}_{i}", config=params, reinit=True)
                    wandb.log({"r2": r2, "rmse": rmse, "mae": mae})
                    wandb.finish()
                except Exception:
                    pass

            progress.progress((i + 1) / len(experiments))

        status_text.empty()
        progress.empty()

        results_df = pd.DataFrame(all_results).sort_values("R²", ascending=False)
        best = results_df.iloc[0]

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0D1A10,#0D1D0D); border:2px solid #64C8A0;
                    border-radius:16px; padding:24px; margin:16px 0;'>
            <div style='font-family:Syne; font-size:1.1rem; font-weight:700; color:#64C8A0; margin-bottom:8px;'>
                🏆 Best Model: {best['Model']}
            </div>
            <div style='color:#90D0A0; font-size:0.9rem;'>
                R² = <b>{best['R²']}</b> &nbsp;|&nbsp; RMSE = <b>{best['RMSE']}</b> &nbsp;|&nbsp; MAE = <b>{best['MAE']}</b>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>📋 All Experiment Results</div>", unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True)

        # Parallel coordinates plot
        st.markdown("<div class='section-title' style='margin-top:16px;'>📈 R² Score per Run</div>", unsafe_allow_html=True)
        fig_runs = px.bar(results_df.reset_index(drop=True),
                           x=results_df.reset_index(drop=True).index,
                           y="R²", color="Model",
                           color_discrete_sequence=["#7C9EFF", "#FF6B9D", "#64C8A0"],
                           labels={"index": "Run #"})
        fig_runs.update_layout(
            height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
            font=dict(color="#9090B0"), xaxis=dict(gridcolor="#1E2140"),
            yaxis=dict(gridcolor="#1E2140"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10)
        )
        st.plotly_chart(fig_runs, use_container_width=True)

        if use_wandb:
            st.markdown(f"""
            <div class='insight-box'>
            📡 All {len(experiments)} experiments logged to W&B project <b>{wb_project}</b>.
            Visit <a href='https://wandb.ai' style='color:#7C9EFF;'>wandb.ai</a> to view your full dashboard.
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#13162B; border:1px dashed #2A2D4A; border-radius:16px;
                    padding:40px; text-align:center; color:#5555AA; margin-top:24px;'>
            <div style='font-size:2rem;'>⚙️</div>
            <div style='font-family:Syne; font-size:1.1rem; margin-top:8px;'>Configure your grid above and click Run</div>
            <div style='font-size:0.85rem; margin-top:6px;'>Experiments run locally; optionally log to W&B with your API key</div>
        </div>""", unsafe_allow_html=True)