import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from data_loader import load_data


def render():
    st.markdown("<div class='hero-title' style='font-size:2.2rem; color:#E8E8F0;'>📊 Data <span style='color:#7C9EFF;'>Visualization</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Exploratory analysis revealing key patterns in global education & unemployment</div>", unsafe_allow_html=True)

    df = load_data()
    df_viz = df.copy()
    df_viz["Unemployment_Rate"] = df_viz["Unemployment_Rate"].replace(0, np.nan)
    df_viz["Youth_15_24_Literacy_Rate_Female"] = df_viz["Youth_15_24_Literacy_Rate_Female"].replace(0, np.nan)
    df_viz["Gross_Tertiary_Education_Enrollment"] = df_viz["Gross_Tertiary_Education_Enrollment"].replace(0, np.nan)
    df_viz["Completion_Rate_Upper_Secondary_Female"] = df_viz["Completion_Rate_Upper_Secondary_Female"].replace(0, np.nan)
    df_viz["Birth_Rate"] = df_viz["Birth_Rate"].replace(0, np.nan)

    # World Map
    st.markdown("<div class='section-title'>🌍 Unemployment Rate by Country</div>", unsafe_allow_html=True)
    fig_map = px.choropleth(
        df_viz.dropna(subset=["Unemployment_Rate"]),
        locations="Countries and areas",
        locationmode="country names",
        color="Unemployment_Rate",
        color_continuous_scale=["#1A1D35", "#3050A0", "#7C9EFF", "#FFD700", "#FF6B6B"],
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

    # Scatter plots
    st.markdown("<div class='section-title' style='margin-top:16px;'>🔗 Education vs. Unemployment</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        df_s1 = df_viz.dropna(subset=["Unemployment_Rate", "Gross_Tertiary_Education_Enrollment"])
        fig1 = px.scatter(
            df_s1, x="Gross_Tertiary_Education_Enrollment", y="Unemployment_Rate",
            hover_name="Countries and areas", trendline="ols",
            color="Unemployment_Rate",
            color_continuous_scale=["#4466FF", "#FF6B6B"],
            labels={"Gross_Tertiary_Education_Enrollment": "Tertiary Enrollment (%)",
                    "Unemployment_Rate": "Unemployment (%)"},
            title="Tertiary Enrollment vs Unemployment"
        )
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
            font=dict(color="#9090B0"), height=340,
            xaxis=dict(gridcolor="#1E2140"), yaxis=dict(gridcolor="#1E2140"),
            title_font=dict(color="#C0C0D8")
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("<div class='insight-box'>💡 Countries with higher tertiary enrollment tend to show <b>lower unemployment</b>, supporting the education-employment link.</div>", unsafe_allow_html=True)

    with col2:
        df_s2 = df_viz.dropna(subset=["Unemployment_Rate", "Birth_Rate"])
        fig2 = px.scatter(
            df_s2, x="Birth_Rate", y="Unemployment_Rate",
            hover_name="Countries and areas", trendline="ols",
            color="Unemployment_Rate",
            color_continuous_scale=["#4466FF", "#FF6B6B"],
            labels={"Birth_Rate": "Birth Rate", "Unemployment_Rate": "Unemployment (%)"},
            title="Birth Rate vs Unemployment"
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
            font=dict(color="#9090B0"), height=340,
            xaxis=dict(gridcolor="#1E2140"), yaxis=dict(gridcolor="#1E2140"),
            title_font=dict(color="#C0C0D8")
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<div class='insight-box'>💡 High birth rates correlate with higher unemployment — often reflecting countries with lower development and education investment.</div>", unsafe_allow_html=True)

    # Gender gap
    st.markdown("<div class='section-title' style='margin-top:16px;'>⚖️ Gender Gap in Education Completion</div>", unsafe_allow_html=True)
    df_gender = df_viz.copy().replace(0, np.nan)
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
        legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(b=120)
    )
    st.plotly_chart(fig_gap, use_container_width=True)
    st.markdown("<div class='insight-box'>💡 In Chad, Niger, and Mali, male students complete upper secondary at <b>10–20% higher rates</b> than females — a significant equity gap linked to higher unemployment in these regions.</div>", unsafe_allow_html=True)

    # Correlation heatmap
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

    # Distribution & Top 10
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>📈 Unemployment Distribution</div>", unsafe_allow_html=True)
        fig_hist = px.histogram(df_viz.dropna(subset=["Unemployment_Rate"]),
                                x="Unemployment_Rate", nbins=30,
                                color_discrete_sequence=["#7C9EFF"])
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
            font=dict(color="#9090B0"), height=280,
            xaxis=dict(gridcolor="#1E2140"), yaxis=dict(gridcolor="#1E2140"),
            margin=dict(t=10)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_d2:
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>🏆 Top 10 Highest Unemployment</div>", unsafe_allow_html=True)
        top10 = df_viz.dropna(subset=["Unemployment_Rate"]).nlargest(10, "Unemployment_Rate")
        fig_top = px.bar(top10, x="Unemployment_Rate", y="Countries and areas",
                         orientation="h", color="Unemployment_Rate",
                         color_continuous_scale=["#7C9EFF", "#FF6B6B"])
        fig_top.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,15,26,0.8)",
            font=dict(color="#9090B0"), height=280,
            xaxis=dict(gridcolor="#1E2140"), yaxis=dict(gridcolor="#1E2140"),
            margin=dict(t=10)
        )
        st.plotly_chart(fig_top, use_container_width=True)
