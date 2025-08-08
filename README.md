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

## 📂 Repository Structure

```
/
├── .gitignore              # Specifies files for Git to ignore
├── README.md               # Project documentation
├── app.py                  # Main Flask application with API endpoints
├── requirements.txt        # Python dependencies for installation
├── train_models.py         # Script to train and save the ML models
│
├── data/
│   ├── iot_carbon_footprint_encoded.csv
│   └── iot_carbon_footprint_labeled_svm.csv
│
├── models/
│   ├── scaler.pkl
│   ├── svm_model.pkl
│   └── xgb_model.pkl
│
└── static/
    └── index.html          # Frontend web page
```

---

## ⚙️ Setup and Execution

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

-   Git
-   Python 3.8 or newer

### 2. Clone the Repository

```bash
git clone [https://github.com/vanamakrishnagurusai/Carbon-Footprint-Optimization-in-IoT-Device-Usage.git](https://github.com/vanamakrishnagurusai/Carbon-Footprint-Optimization-in-IoT-Device-Usage.git)
cd Carbon-Footprint-Optimization-in-IoT-Device-Usage
```

### 3. Install Dependencies

It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required libraries
pip install -r requirements.txt
```

### 4. Train the Models (Optional)

The repository already contains pre-trained models in the `/models` directory. However, if you wish to retrain the models using the datasets in the `/data` directory, run the following command:

```bash
python train_models.py
```
This script will process the datasets, train the XGBoost and SVM models, and save the updated `.pkl` files in the `models/` folder.

### 5. Run the Application Server

Start the Flask server to launch the web interface and activate the API.

```bash
python app.py
```
The application will be accessible at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

---

## 🌐 API Endpoints

The application exposes the following API endpoints.

### Health Check

-   **URL**: `/api/status`
-   **Method**: `GET`
-   **Description**: Checks the server status and confirms if the ML models were loaded successfully.

### Prediction

-   **URL**: `/api/predict`
-   **Method**: `POST`
-   **Description**: Takes IoT device data as input, performs feature engineering, and returns energy predictions and scheduling recommendations.
-   **Request Body (JSON)**:

    ```json
    {
        "device_id": 101,
        "energy_consumption": 55.5,
        "usage_duration": 12.0,
        "temperature": 25.5,
        "priority": 1,
        "time_of_operation": 1,
        "idle_time": 2.5
    }
    ```

-   **Success Response (JSON)**:

    ```json
    {
        "success": true,
        "predicted_consumption": 54.8,
        "predicted_category": "Medium",
        "schedule": "Off-Peak Hours",
        "suggestion": "Moderate consumption. Medium priority allows Off-Peak."
    }
    ```


