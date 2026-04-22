import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

DATASETS = {
    "Global Education": "Global_Education.csv",
    # Add more datasets here if needed
}

DATASET_DESCRIPTIONS = {
    "Global Education": {
        "title": "Global Education Indicators",
        "problem": "How do education indicators predict unemployment rates across different countries?",
        "target": "Unemployment_Rate",
        "target_desc": "The unemployment rate (%) for each country, which we aim to predict based on education indicators.",
        "source": "UNESCO Institute for Statistics (UIS) and World Bank",
        "rows": 202,
        "features_desc": {
            "Completion_Rate_Primary_Male": "The percentage of male students who complete primary education in each country", 
            "Completion_Rate_Primary_Female": "The percentage of female students who complete primary education in each country",
            "Completion_Rate_Lower_Secondary_Male": "The percentage of male students who complete lower secondary education in each country",
            "Completion_Rate_Lower_Secondary_Female": "The percentage of female students who complete lower secondary education in each country",
            "Completion_Rate_Upper_Secondary_Male": "The percentage of male students who complete upper secondary education in each country",
            "Completion_Rate_Upper_Secondary_Female": "The percentage of female students who complete upper secondary education in each country",
            "Youth_15_24_Literacy_Rate_Male": "The percentage of male youth (15-24) who are literate in each country",
            "Youth_15_24_Literacy_Rate_Female": "The percentage of female youth (15-24) who are literate in each country",
            "Birth_Rate": "The number of births per 1,000 people in each country",
            "Gross_Primary_Education_Enrollment": "The gross enrollment ratio for primary education in each country",
            "Gross_Tertiary_Education_Enrollment": "The gross enrollment ratio for tertiary education in each country",
            "Lower_Secondary_End_Proficiency_Reading": "The proficiency level in reading at the end of lower secondary education in each country",
            "Lower_Secondary_End_Proficiency_Math": "The proficiency level in math at the end of lower secondary education in each country"
        }
    }
}   # Add descriptions for new datasets here

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
