# 🎓 EduPredict — Global Education & Unemployment Analytics

A Streamlit application that analyzes global education indicators across 202 countries to understand how literacy, school completion, and enrollment rates drive unemployment — and builds a machine learning model to predict it.

---

## 📁 Project Structure

```
edupredict/
├── app.py                    # Main entry point & navigation
├── data_loader.py            # Data loading & preprocessing logic
├── precompute_importance.py  # Precomputes SHAP feature importance
├── wandb_tracker.py          # Weights & Biases experiment logging
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
└── pages/
    ├── __init__.py
    ├── page_intro.py         # Overview & Data page
    ├── page_visualization.py # Data Visualization page
    ├── page_prediction.py    # Prediction Models page
    ├── page_explainability.py# SHAP Explainability page
    └── page_tuning.py        # Hyperparameter Tuning page
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/edupredict.git
cd edupredict
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your W&B API key (optional)
```

### 4. Add the dataset

Place `Global_Education.csv` in the project root. The dataset is sourced from UNESCO / World Bank and contains 202 countries × 27 features.

### 5. (Optional) Precompute SHAP importance

```bash
python precompute_importance.py
```

### 6. Run the app

```bash
streamlit run app.py
```

---

## 📊 Features

| Page | Description |
|------|-------------|
| 🏠 Overview & Data | Dataset preview, KPIs, business context |
| 📊 Data Visualization | World map, scatter plots, gender gap, correlation heatmap |
| 🤖 Prediction Models | Train & compare 6 regression models, interactive prediction tool |
| 🔍 Explainability (SHAP) | Global feature importance, beeswarm plot, waterfall per sample |
| ⚙️ Hyperparameter Tuning | Grid search with optional W&B experiment tracking |

---

## 🧠 Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting

---

## 🎯 Target Variable

**Unemployment Rate (%)** — predicted from 13 education indicators including completion rates, literacy rates, proficiency scores, enrollment rates, and birth rate.

---

## 🔧 Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn, SHAP
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Experiment Tracking**: Weights & Biases (wandb)

---

## 📄 Data Source

UNESCO Institute for Statistics / World Bank — Global Education dataset covering 202 countries.
