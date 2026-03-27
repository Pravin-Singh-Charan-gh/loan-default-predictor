# 🏦 Loan Approval Predictor

A machine learning-powered web application that predicts loan approval status based on applicant details. Built with Streamlit, this project uses a trained classification model to assess loan eligibility with high accuracy.

## 🌐 Live Demo

**Try the application now**: [https://loan-default-predictor-pravin.streamlit.app/](https://loan-default-predictor-pravin.streamlit.app/)

The app is deployed and accessible online. No installation required – just visit the link and start making predictions!

---

## 📋 Table of Contents

- [Live Demo](#-live-demo)
- [Quick Start](#quick-start)
- [Overview](#overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Key Features Explained](#-key-features-explained)
- [Data Analysis Insights](#-data-analysis-insights)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)

---

## Quick Start

### Want to use it right now?
👉 **Visit the live app**: [https://loan-default-predictor-pravin.streamlit.app/](https://loan-default-predictor-pravin.streamlit.app/)

### Want to run it locally?
```bash
git clone <repository-url>
cd loan_project
pip install -r requirements.txt
streamlit run app.py
```

Then open: `http://localhost:8501`

---

## Overview

This project is an end-to-end machine learning application that predicts whether a loan application will be approved or rejected. The application consists of two main components:

1. **Model Training Pipeline** (`train_model.py`) - Trains and serializes the ML model
2. **Web Interface** (`app.py`) - Interactive Streamlit dashboard for predictions

The model achieves **78.86% accuracy** using Logistic Regression and provides real-time predictions with confidence scores.

---

## ✨ Features

### 🎯 Core Functionality
- **Real-time Loan Predictions**: Instant approval/rejection predictions with confidence scores
- **Interactive Dashboard**: User-friendly Streamlit interface for easy navigation
- **Applicant Summary**: Detailed table showing all input parameters and submitted values
- **Confidence Metrics**: Clear percentage-based confidence levels for predictions

### 📊 Data Analysis Dashboard
- **Loan Approval Distribution**: Visual breakdown of approved vs. rejected loans
- **Income vs Approval Analysis**: Correlation between applicant income and approval status
- **Credit History Impact**: Analysis of credit history as a predictor
- **Feature Importance**: Visualization of which features drive predictions most

### 🔍 Model Transparency
- **Model Comparison**: Bar chart comparing accuracy across multiple algorithms
- **Confidence Indicators**: Real-time probability scores for each prediction
- **Feature Engineering**: Proper encoding and normalization of categorical variables

---

## 📁 Project Structure

```
LOAN_PROJECT/
├── app.py                          # Main Streamlit application
├── train_model.py                  # Model training and serialization
├── loan.csv                        # Raw dataset
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/                           # Data directory
│   └── loan.csv                    # Loan application data
│
├── plots/                          # EDA visualizations
│   ├── target.png                  # Approval distribution chart
│   ├── income_vs_status.png        # Income vs approval analysis
│   ├── credit_history.png          # Credit history impact visualization
│   └── feature_importance.png      # Model feature importance
│
└── model artifacts/
    ├── model.pkl                   # Trained ML model
    ├── features.pkl                # Feature names list
    ├── target_col.pkl              # Target column label
    └── target_col_dict.pkl         # Target encoding mapping
```

---

## 📊 Dataset

### Overview
- **Total Records**: Loan application data with demographic and financial information
- **Target Variable**: Loan approval status (Approved/Rejected)
- **Features**: 11 input features with demographic, financial, and property information

### Features Description

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| **Gender** | Categorical | Applicant's gender | Male, Female |
| **Married** | Categorical | Marital status | Yes, No |
| **Dependents** | Categorical | Number of dependents | 0, 1, 2, 3+ |
| **Education** | Categorical | Educational qualification | Graduate, Not Graduate |
| **Self_Employed** | Categorical | Self-employment status | Yes, No |
| **Applicant_Income** | Numerical | Annual income (₹) | Continuous |
| **Coapplicant_Income** | Numerical | Co-applicant income (₹) | Continuous |
| **Loan_Amount** | Numerical | Loan amount requested (in thousands) | Continuous |
| **Loan_Term** | Numerical | Loan duration (months) | 12, 36, 60, 84, 120, 180, 360 |
| **Credit_History** | Numerical | Credit history status | 0.0 (Bad), 1.0 (Good) |
| **Property_Area** | Categorical | Property location | Urban, Semiurban, Rural |

---

## 🎯 Model Performance

### Algorithm Comparison

| Model | Accuracy | Use Case |
|-------|----------|----------|
| **Logistic Regression** | 78.86% | ⭐ Selected (Best performance) |
| **Random Forest** | 76.42% | Ensemble baseline |
| **XGBoost** | 73.17% | Gradient boosting alternative |

### Key Metrics
- **Best Performing Model**: Logistic Regression
- **Accuracy**: 78.86%
- **Feature Engineering**: Categorical encoding for categorical variables
- **Model Output**: Binary classification with probability scores

### Top Predictive Features
1. **Credit History** - Strongest predictor of loan approval
2. **Loan Amount** - Second most important feature
3. **Applicant Income** - Significant financial indicator
4. **Co-applicant Income** - Secondary income consideration

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone or Extract the Project**
   ```bash
   cd loan_project
   ```

2. **Create a Virtual Environment** (Optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib matplotlib
   ```

4. **Verify Installation**
   ```bash
   streamlit --version
   python -c "import pandas, numpy, joblib; print('All packages installed!')"
   ```

---

## 💻 Usage

### 🌐 Online Access (Recommended)
Simply visit the live deployment: **[https://loan-default-predictor-pravin.streamlit.app/](https://loan-default-predictor-pravin.streamlit.app/)**

No installation or setup required. Just open the link in your browser and start making predictions!

### 📱 Local Installation

#### Train the Model (Optional)
If you want to retrain the model with updated data:

```bash
python train_model.py
```

This will:
- Load the `loan.csv` dataset
- Preprocess and encode features
- Train the Logistic Regression model
- Save artifacts: `model.pkl`, `features.pkl`, etc.

### Run the Web Application

1. **Start the Streamlit Server**
   ```bash
   streamlit run app.py
   ```

2. **Access the Application**
   - Local URL: `http://localhost:8501`
   - Open in your web browser

3. **Make a Prediction**
   - Fill in applicant details in the left sidebar
   - Adjust all required fields
   - Click the "🔍 Predict Approval" button
   - View results in the main panel

### Example Input
```
Gender: Male
Married: Yes
Dependents: 2
Education: Graduate
Self Employed: No
Applicant Income: ₹50,000
Co-applicant Income: ₹30,000
Loan Amount: ₹150 (thousands)
Loan Term: 360 months
Credit History: Good (1.0)
Property Area: Urban
```

---

## ⚙️ Configuration

### Input Encoding Mapping

The application uses the following encoding scheme:

```python
# Gender
"Male": 1, "Female": 0

# Marital & Employment Status
"Yes": 1, "No": 0

# Education
"Graduate": 0, "Not Graduate": 1

# Property Area
"Urban": 2, "Semiurban": 1, "Rural": 0

# Dependents
"0": 0, "1": 1, "2": 2, "3+": 3

# Credit History
"Good (1.0)": 1.0, "Bad (0.0)": 0.0
```

### Customization Options

**Modify Default Values** (in `app.py`):
```python
income     = st.sidebar.number_input("Applicant Income (₹)", 
                                     min_value=0, 
                                     value=5000,    # Change default value
                                     step=500)
```

**Adjust Loan Term Options**:
```python
loan_term  = st.sidebar.selectbox("Loan Term (months)", 
                                  [360, 180, 120, 84, 60, 36, 12])
```

**Update Color Scheme** (in `app.py`):
```python
colors = ['#378ADD', '#3B6D11', '#854F0B']  # Modify these hex values
```

---

## 🔍 Key Features Explained

### 1. Sidebar Input Form
- **Location**: Left panel of the application
- **Functionality**: Collects all applicant information
- **Features**: Dropdowns for categorical data, number inputs for numerical data
- **Button**: "🔍 Predict Approval" triggers the prediction

### 2. Prediction Result Panel
- **Location**: Bottom-left section
- **Shows**: 
  - ✅ Approval status (Green for approved, Red for rejected)
  - Confidence percentage (Approval or Rejection confidence)
  - Complete applicant summary table

### 3. Model Comparison Chart
- **Location**: Bottom-right section
- **Shows**: Accuracy comparison across three models
- **Purpose**: Demonstrates model selection rationale

### 4. EDA Dashboard
- **Location**: Below main prediction panel
- **Tabs**:
  - Tab 1: Loan approval distribution chart
  - Tab 2: Income vs approval scatter/box plot
  - Tab 3: Credit history impact analysis
  - Tab 4: Feature importance ranking

---

## 📈 Data Analysis Insights

### Key Findings

1. **Class Imbalance**: The dataset shows more approved loans than rejected ones
   - Implication: Model may have slight bias toward approval prediction
   - Mitigation: Can use class weights during model training

2. **Income Correlation**: Higher income applicants tend to get approved
   - Exceptions exist (some high-income applicants are rejected)
   - Income threshold varies by other factors

3. **Credit History Dominance**: Single strongest predictor
   - Applicants with good credit history have much higher approval rate
   - Nearly mandatory for loan approval

4. **Loan Amount Impact**: Larger loan amounts have lower approval rates
   - Combined with income, defines approval likelihood
   - Risk assessment strongly influenced by this factor

---

## ☁️ Deployment

### Current Deployment
The application is **live and accessible** at:
```
https://loan-default-predictor-pravin.streamlit.app/
```

**Features**:
- ✅ Always available and up-to-date
- ✅ No installation required
- ✅ Accessible from any device with internet
- ✅ Automatically updated when code changes

### Deploy Your Own Version

#### Option 1: Deploy to Streamlit Cloud (Recommended)

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Visit Streamlit Cloud**: https://streamlit.io/cloud

3. **Create New App**
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Choose branch and file (`app.py`)
   - Click "Deploy"

4. **Share your URL** with others

#### Option 2: Deploy to Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py" > Procfile

# Create setup.sh
cat > setup.sh << EOF
mkdir -p ~/.streamlit/
echo "[server]" > ~/.streamlit/config.toml
echo "port = $PORT" >> ~/.streamlit/config.toml
echo "enableCORS = false" >> ~/.streamlit/config.toml
echo "headless = true" >> ~/.streamlit/config.toml
EOF

# Deploy
heroku login
heroku create your-app-name
git push heroku main
```

#### Option 3: Deploy to AWS / Google Cloud

- Use EC2/App Engine with Docker
- Container example:
  ```dockerfile
  FROM python:3.9
  COPY . /app
  WORKDIR /app
  RUN pip install -r requirements.txt
  CMD streamlit run app.py --server.port=8080
  ```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Solution**:
```bash
pip install streamlit
```

### Issue: "FileNotFoundError: 'model.pkl' not found"

**Solution**:
1. Ensure you're in the correct directory
2. Run `python train_model.py` to generate the model
3. Verify model.pkl exists in the project root

### Issue: "plots/target.png not found"

**Solution**:
1. Create a `plots/` directory
2. Run `python train_model.py` to generate visualizations
3. Or temporarily comment out the image display lines in app.py

### Issue: Application runs but shows no predictions

**Solution**:
1. Ensure all input fields are filled
2. Click the "🔍 Predict Approval" button after filling inputs
3. Check browser console for JavaScript errors
4. Restart the Streamlit server

### Issue: Accuracy shown as different than documented

**Solution**:
1. Model accuracy depends on training data
2. If you retrained the model with different data, accuracy will change
3. Update the accuracies list in the bar chart section if needed

---

## 🚀 Future Enhancements

### Short-term Improvements
- [ ] Add data validation with clear error messages
- [ ] Implement batch prediction (CSV upload)
- [ ] Add model explainability with SHAP values
- [ ] Create model performance metrics dashboard
- [ ] Add user authentication and application history

### Medium-term Features
- [ ] Integrate with external credit scoring APIs
- [ ] Add real-time data updates from financial databases
- [ ] Implement A/B testing for model versions
- [ ] Create risk stratification scoring
- [ ] Add loan recommendation system

### Long-term Vision
- [ ] Deploy to cloud platform (AWS, GCP, Azure)
- [ ] Build REST API backend
- [ ] Add mobile application
- [ ] Implement continuous model monitoring
- [ ] Create comprehensive admin dashboard
- [ ] Add multivariate analysis and forecasting
- [ ] Implement ensemble methods with auto-weighting

---

## 📝 Dependencies

All required packages are listed in `requirements.txt`:

```
streamlit>=1.18.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
joblib>=1.2.0
matplotlib>=3.6.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 👨‍💻 Contributing

To improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit changes (`git commit -am 'Add improvement'`)
5. Push to branch (`git push origin feature/improvement`)
6. Create a Pull Request

---

## 📧 Support & Questions

For issues or questions:

1. Check the **Troubleshooting** section above
2. Review the code comments in `app.py` and `train_model.py`
3. Verify all dependencies are installed correctly
4. Ensure model artifacts (model.pkl, features.pkl) exist

---

## 🎓 Learning Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **Scikit-learn ML Models**: https://scikit-learn.org
- **Pandas Data Manipulation**: https://pandas.pydata.org
- **Machine Learning Basics**: https://developers.google.com/machine-learning

---

## 📊 Model Training Details

The `train_model.py` script performs the following:

1. **Data Loading**: Loads loan.csv dataset
2. **Data Cleaning**: Handles missing values and data quality issues
3. **Feature Encoding**: Converts categorical variables to numerical
4. **Model Training**: Trains Logistic Regression model
5. **Model Evaluation**: Calculates accuracy and other metrics
6. **Serialization**: Saves model and features using joblib
7. **Visualization**: Generates EDA plots saved to `/plots` directory

---

## 🎯 Next Steps

1. **Install & Run**: Follow the Installation section
2. **Explore**: Try different applicant profiles in the interface
3. **Understand**: Review EDA dashboard insights
4. **Extend**: Consider implementing suggested enhancements
5. **Deploy**: Consider cloud deployment for production use

---

**Last Updated**: March 2026  
**Python Version**: 3.8+  
**Status**: ✅ Production Ready

---

## Quick Start Command

```bash
# Complete setup and run in one go
pip install -r requirements.txt && streamlit run app.py
```

Your Loan Approval Predictor is now ready to use! 🚀
