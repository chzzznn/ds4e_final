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
