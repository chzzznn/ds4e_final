import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from data_loader import get_model_data
import wandb_tracker as wb


def render():
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

    # Hyperparameter grid
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
            logged_in = wb.login(wb_api_key)
            if logged_in:
                st.success("✅ W&B login successful! Experiments will be logged.")
            else:
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
            status_text.markdown(
                f"<div style='color:#7C9EFF; font-size:0.9rem;'>Running: {model_name} | {params}</div>",
                unsafe_allow_html=True
            )

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
                wb.log_run(
                    project=wb_project,
                    run_name=f"{model_name}_{i}",
                    config=params,
                    metrics={"r2": r2, "rmse": rmse, "mae": mae}
                )

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

        st.markdown("<div class='section-title' style='margin-top:16px;'>📈 R² Score per Run</div>", unsafe_allow_html=True)
        fig_runs = px.bar(
            results_df.reset_index(drop=True),
            x=results_df.reset_index(drop=True).index,
            y="R²", color="Model",
            color_discrete_sequence=["#7C9EFF", "#FF6B9D", "#64C8A0"],
            labels={"index": "Run #"}
        )
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
