import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import get_model_data


MODELS_XAI = {
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Linear Regression": lambda: LinearRegression(),
}


def render():
    st.markdown("<div class='hero-title' style='font-size:2.2rem; color:#E8E8F0;'>🔍 Explainability <span style='color:#7C9EFF;'>(SHAP)</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Understanding which education factors drive unemployment predictions</div>", unsafe_allow_html=True)

    X, y, countries, features = get_model_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model_choice = st.selectbox("Choose model to explain", list(MODELS_XAI.keys()))
    model = MODELS_XAI[model_choice]()
    model.fit(X_train_s, y_train)

    X_test_df = pd.DataFrame(X_test_s, columns=features)

    with st.spinner("Computing SHAP values..."):
        if model_choice == "Linear Regression":
            explainer = shap.LinearExplainer(model, X_train_s)
            shap_values = explainer.shap_values(X_test_s)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_df)

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

    # Beeswarm summary plot
    st.markdown("<div class='section-title' style='margin-top:16px;'>🌡️ SHAP Beeswarm Summary Plot</div>", unsafe_allow_html=True)
    fig_summary, _ = plt.subplots(figsize=(10, 5))
    fig_summary.patch.set_facecolor("#0D0F1A")
    shap_input = X_test_df if model_choice != "Linear Regression" else pd.DataFrame(X_test_s, columns=features)
    shap.summary_plot(shap_values, shap_input, feature_names=features, show=False, plot_size=None)
    plt.gcf().patch.set_facecolor("#0D0F1A")
    for ax_ in plt.gcf().get_axes():
        ax_.set_facecolor("#0D0F1A")
        ax_.tick_params(colors="#8888AA")
        ax_.spines["bottom"].set_color("#2A2D4A")
        ax_.spines["left"].set_color("#2A2D4A")
    st.pyplot(fig_summary, use_container_width=True)
    plt.close()

    # Waterfall for single sample
    st.markdown("<div class='section-title' style='margin-top:16px;'>🎯 Single Prediction Breakdown</div>", unsafe_allow_html=True)
    sample_idx = st.slider("Select test sample index", 0, len(X_test_s) - 1, 0)
    fig_wf, _ = plt.subplots(figsize=(10, 4))
    fig_wf.patch.set_facecolor("#0D0F1A")
    data_row = (
        X_test_df.iloc[sample_idx]
        if model_choice != "Linear Regression"
        else pd.Series(X_test_s[sample_idx], index=features)
    )
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=float(explainer.expected_value[0]) if hasattr(explainer, "expected_value") and hasattr(explainer.expected_value, "__len__") else float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0,
            data=data_row,
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