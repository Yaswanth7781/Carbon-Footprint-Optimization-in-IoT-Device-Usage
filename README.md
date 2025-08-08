# 🌱 Carbon Footprint Optimization in IoT Device Usage using Machine Learning

## 📌 Overview
This project focuses on optimizing **IoT device energy consumption** to reduce **carbon emissions** and promote sustainability.  
Using **XGBoost Regression** and **Support Vector Machine (SVM)** classification, the system predicts future energy consumption, categorizes devices based on power usage, and recommends optimal operational schedules — particularly shifting high-energy tasks to off-peak hours.

---

## 🎯 Objectives
- Predict future energy consumption of IoT devices using **XGBoost**.
- Categorize IoT devices into **High**, **Medium**, and **Low** power consumers using **SVM**.
- Schedule high-power consumption devices during **off-peak hours** to leverage renewable energy sources.
- Use feature engineering to improve prediction accuracy.
- Deploy lightweight models for **real-time, edge-based optimization**.

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries & Frameworks:**
  - `pandas`, `numpy` – Data handling
  - `scikit-learn` – Machine learning models & preprocessing
  - `xgboost` – Regression modeling
  - `matplotlib`, `seaborn`, `plotly` – Data visualization
  - `shap` – Feature importance analysis
- **Tools:** Jupyter Notebook, Git  
- **Data Source:** Kaggle IoT Energy Dataset

---

## 🔍 Methodology
1. **Data Collection**  
   - IoT device usage data from Kaggle, including features like usage duration, temperature, priority, power source, network activity, and idle time.

2. **Data Preprocessing**
   - **One-Hot Encoding** for categorical data (e.g., power source).
   - **Label Encoding** for ordinal features (e.g., priority: Low=0, Medium=1, High=2).
   - **StandardScaler** for numerical normalization.

3. **Exploratory Data Analysis (EDA)**
   - Identified energy consumption trends, peak/off-peak usage, and inefficiencies.
   - Boxplots and correlation matrices for feature impact analysis.

4. **Model Development**
   - **XGBoost Regressor** – Predicts future energy consumption.
   - **SVM Classifier** – Categorizes devices into energy consumption groups.

5. **Evaluation**
   - XGBoost:
     - MAE: 0.0633
     - MSE: 0.0079
     - R² Score: 0.9989
   - SVM:
     - Accuracy: 99%
     - Low misclassification in confusion matrix

6. **Deployment**
   - Models suitable for **edge devices** for real-time decision-making.
   - Recommendations for optimal scheduling of devices.

---

## 📊 Results
| Model      | Accuracy | MAE    | MSE    | R² Score |
|------------|----------|--------|--------|----------|
| XGBoost    | 99.46%   | 0.0633 | 0.0079 | 0.9989   |
| SVM        | 99.03%   | N/A    | N/A    | N/A      |

- **XGBoost** showed almost perfect prediction accuracy.
- **SVM** provided highly accurate device categorization.
- Visualizations confirmed a clear correlation between device features and energy usage.

---

## 📁 Project Structure


