# Loan Default Prediction App

A machine learning web application that predicts whether a loan applicant will be approved or rejected, built with Python, scikit-learn, and Streamlit.

**Live App:** https://loan-default-predictor-pravin.streamlit.app/
**GitHub:** https://github.com/Pravin-Singh-Charan-gh/loan-default-predictor

---

## Model Performance

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 78.86%   |
| Random Forest       | 76.42%   |
| XGBoost             | 73.17%   |

Best model: **Logistic Regression (78.86%)**

---

## Tech Stack

- **Language:** Python 3.11
- **ML Libraries:** scikit-learn, XGBoost, pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Web App:** Streamlit
- **Dataset:** Loan Prediction Dataset (Kaggle) — 614 records, 12 features
- **Deployment:** Streamlit Cloud

---

## Features

- Predict loan approval with confidence percentage from applicant details
- Compare accuracy of 3 ML models side by side
- Interactive EDA dashboard with 4 charts: approval distribution, income analysis, credit history impact, and feature importance
- Real-time prediction form with applicant summary table

---

## Project Structure

```
loan_project/
├── data/loan.csv           # Kaggle dataset
├── plots/                  # EDA charts (auto-generated)
├── train_model.py          # Data cleaning, EDA, model training, saving
├── app.py                  # Streamlit web application
├── model.pkl               # Saved best model
├── features.pkl            # Saved feature names
└── requirements.txt        # Dependencies
```

---

## How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (generates plots/ and model.pkl)
python train_model.py

# Launch the web app
streamlit run app.py
```

---

## Key ML Concepts Applied

- **Data Cleaning:** Handled missing values using mode/mean imputation
- **Feature Encoding:** Label encoding for categorical variables
- **Train-Test Split:** 80/20 split with random seed for reproducibility
- **Model Comparison:** Evaluated Logistic Regression, Random Forest, XGBoost
- **Feature Importance:** Identified Credit History and Loan Amount as top predictors
- **Model Persistence:** Saved trained model using joblib for production use

---
## Author
Pravin Singh Charan
 | Linkedin: https://www.linkedin.com/in/pravin07/
