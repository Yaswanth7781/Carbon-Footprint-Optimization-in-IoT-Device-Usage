# train_and_save_models.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Train and save XGBoost model
def train_xgboost_model(file_path):
    print("Training XGBoost model...")
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Check for missing values
    df.dropna(inplace=True)
    
    # Feature Engineering
    df['energy_efficiency'] = df['energy_consumption'] / df['usage_duration']
    df['idle_energy_waste'] = df['idle_time'] * df['energy_consumption']
    df['normalized_priority'] = df['priority'] / df['priority'].max()
    
    # Define features & target
    X = df.drop(columns=['energy_consumption'])
    y = df['energy_consumption']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost Model
    xgb_regressor = xgb.XGBRegressor(n_estimators=450, learning_rate=0.5, max_depth=5, random_state=42)
    xgb_regressor.fit(X_train, y_train)
    
    # Save the model
    with open('models/xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_regressor, f)
    
    print("XGBoost model trained and saved!")

# Train and save SVM model
def train_svm_model(file_path):
    print("Training SVM model...")
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Check if 'energy_label' exists
    if "energy_label" not in df.columns:
        raise KeyError("Column 'energy_label' not found in dataset. Ensure it's labeled correctly.")
    
    # Handle missing values
    df.dropna(inplace=True)
    
    # Define features and target
    X = df.drop(columns=["energy_label"])
    y = df["energy_label"]  # Target (0 = Low, 1 = Medium, 2 = High)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM model
    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm_model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    with open('models/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("SVM model and scaler trained and saved!")

if __name__ == "__main__":
    # Replace these with your actual file paths
    xgboost_data_path = "data/iot_carbon_footprint_encoded.csv" # Use the new filename
    svm_data_path = "data/iot_carbon_footprint_labeled_svm.csv"
    
    train_xgboost_model(xgboost_data_path)
    train_svm_model(svm_data_path)
    
    print("All models trained and saved successfully!")
