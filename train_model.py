import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ── Load data ──────────────────────────────────────────────
os.makedirs('plots', exist_ok=True)
df = pd.read_csv('data/loan.csv')

# ── Clean missing values ───────────────────────────────────
df['Gender']           = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']          = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents']       = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed']    = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount']       = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History']   = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# ── EDA charts ────────────────────────────────────────────
plt.figure(figsize=(6, 4))
df['Loan_Status'].value_counts().plot(kind='bar', color=['#378ADD', '#E24B4A'])
plt.title('Loan Approval: Approved vs Rejected')
plt.xlabel('Loan Status (Y=Approved, N=Rejected)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/target.png')
plt.close()

plt.figure(figsize=(8, 4))
sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df,
            hue='Loan_Status', palette=['#378ADD', '#E24B4A'], legend=False)
plt.title('Applicant Income vs Loan Status')
plt.tight_layout()
plt.savefig('plots/income_vs_status.png')
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(x='Credit_History', hue='Loan_Status', data=df,
              palette=['#378ADD', '#E24B4A'])
plt.title('Credit History vs Loan Approval')
plt.tight_layout()
plt.savefig('plots/credit_history.png')
plt.close()

# ── Encode and split ──────────────────────────────────────
le = LabelEncoder()
text_columns = ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Property_Area', 'Loan_Status']
for col in text_columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train models ──────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds  = model.predict(X_test)
    acc    = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name:25s} → {acc*100:.2f}%")

best_name  = max(results, key=results.get)
best_model = models[best_name]
print(f"\nBest model : {best_name} ({results[best_name]*100:.2f}%)")
print(classification_report(y_test, best_model.predict(X_test)))

# ── Feature importance chart ──────────────────────────────
rf_model    = models['Random Forest']
importances = rf_model.feature_importances_
feat_names  = X.columns.tolist()
sorted_idx  = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[sorted_idx], color='#378ADD')
plt.xticks(range(len(importances)),
           [feat_names[i] for i in sorted_idx],
           rotation=45, ha='right')
plt.title('Feature Importance — Which factors matter most?')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()

# ── Save model ────────────────────────────────────────────
joblib.dump(best_model, 'model.pkl')
joblib.dump(feat_names, 'features.pkl')