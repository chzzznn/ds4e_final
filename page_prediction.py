import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from data_loader import get_model_data


MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge Regression": lambda: Ridge(alpha=1.0),
    "Lasso Regression": lambda: Lasso(alpha=0.1),
    "Decision Tree": lambda: DecisionTreeRegressor(max_depth=4, random_state=42),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
}


def render():
    st.markdown("<div class='hero-title' style='font-size:2.2rem; color:#E8E8F0;'>🤖 Prediction <span style='color:#7C9EFF;'>Models</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Train & compare 6 regression models to predict unemployment rate</div>", unsafe_allow_html=True)

    X, y, countries, features = get_model_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>🎛 Select Models to Compare</div>", unsafe_allow_html=True)
        selected_models = st.multiselect(
            "Choose models", list(MODELS.keys()),
            default=["Linear Regression", "Random Forest", "Gradient Boosting"],
            label_visibility="collapsed"
        )
        primary_model = st.selectbox(
            "Primary model for prediction",
            selected_models if selected_models else list(MODELS.keys())
        )

    with col_info:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>📐 Train / Test Split</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Samples", len(y))
        c2.metric("Train Samples", len(y_train))
        c3.metric("Test Samples", len(y_test))

    if not selected_models:
        st.warning("Please select at least one model.")
        st.stop()

    # Train & evaluate
    results = {}
    trained_models = {}
    for name in selected_models:
        model = MODELS[name]()
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        trained_models[name] = model
        cv_model = MODELS[name]()
        results[name] = {
            "R² Score": round(r2_score(y_test, preds), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
            "MAE": round(mean_absolute_error(y_test, preds), 4),
            "CV R² (5-fold)": round(
                cross_val_score(cv_model, X_train_s, y_train, cv=5, scoring="r2").mean(), 4
            ),
        }

    # Comparison chart
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
    st.markdown(f"<div class='section-title' style='margin-top:24px;'>🎯 Actual vs Predicted — {primary_model}</div>", unsafe_allow_html=True)
    best_model = trained_models[primary_model]
    preds_best = best_model.predict(X_test_s)

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(x=y_test, y=preds_best, mode="markers",
                                  marker=dict(color="#7C9EFF", size=8, opacity=0.7)))
    fig_avp.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
        mode="lines", line=dict(color="#FF6B6B", dash="dash"), name="Perfect Fit"
    ))
    fig_avp.update_layout(
        height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
        font=dict(color="#9090B0"), xaxis=dict(title="Actual", gridcolor="#1E2140"),
        yaxis=dict(title="Predicted", gridcolor="#1E2140"),
        legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=10)
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    # Interactive prediction tool
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
        prediction = max(0, best_model.predict(input_scaled)[0])
        color = "#64C8A0" if prediction < 6 else "#FFD700" if prediction < 12 else "#FF6B6B"
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1A1D35,#1F2240); border:2px solid {color};
                    border-radius:16px; padding:30px; text-align:center; margin-top:16px;'>
            <div style='font-family:Syne; font-size:3rem; font-weight:800; color:{color};'>{prediction:.2f}%</div>
            <div style='color:#8888AA; font-size:0.9rem; margin-top:8px;'>Predicted Unemployment Rate using {primary_model}</div>
        </div>""", unsafe_allow_html=True)
