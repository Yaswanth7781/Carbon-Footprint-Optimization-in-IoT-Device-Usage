


from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Define model file paths
MODEL_DIR = "models"
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def load_models():
    """Load machine learning models and scaler"""
    try:
        # Check if model files exist
        if not all(os.path.exists(path) for path in [XGB_MODEL_PATH, SVM_MODEL_PATH, SCALER_PATH]):
            raise FileNotFoundError("Required model files are missing")
            
        # Load models
        with open(XGB_MODEL_PATH, 'rb') as f:
            xgb_model = pickle.load(f)
        with open(SVM_MODEL_PATH, 'rb') as f:
            svm_model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
            
        return xgb_model, svm_model, scaler
        
    except Exception as e:
        app.logger.error(f"Error loading models: {str(e)}")
        raise

# Initialize models at startup
try:
    xgb_model, svm_model, scaler = load_models()
except Exception as e:
    app.logger.error(f"Failed to initialize models: {str(e)}")
    xgb_model = svm_model = scaler = None

@app.route('/')
def serve_index():
    """Serve the frontend HTML"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if xgb_model is None or svm_model is None or scaler is None:
        return jsonify({
            'success': False,
            'error': 'Models not loaded'
        }), 500

    try:
        # Get and validate input data
        data = request.get_json()
        required_fields = [
            'device_id', 'energy_consumption', 'usage_duration', 
            'temperature', 'priority', 'time_of_operation', 'idle_time'
        ]
        
        if not all(field in data for field in required_fields):
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400

        # Convert input to floats
        input_data = {k: float(data[k]) for k in required_fields}
        
        # Feature engineering
        energy_efficiency = input_data['energy_consumption'] / input_data['usage_duration']
        idle_energy_waste = input_data['idle_time'] * input_data['energy_consumption']
        normalized_priority = input_data['priority'] / 2  # Assuming max priority is 2

        # Prepare data for XGBoost
        xgb_features = {
            'device_id': input_data['device_id'],
            'usage_duration': input_data['usage_duration'],
            'temperature': input_data['temperature'],
            'priority': input_data['priority'],
            'network_activity_MB':0,
            'time_of_operation': input_data['time_of_operation'],
            'idle_time': input_data['idle_time'],
            'power_source_Battery': 0,  # Default values
            'power_source_Solar': 0,
            'energy_efficiency': energy_efficiency,
            'idle_energy_waste': idle_energy_waste,
            'normalized_priority': normalized_priority
        }
        xgb_df = pd.DataFrame([xgb_features])

        # Prepare data for SVM
        svm_features = {
            'device_id': input_data['device_id'],
            'energy_consumption': input_data['energy_consumption'],
            'usage_duration': input_data['usage_duration'],
            'temperature': input_data['temperature'],
            'priority': input_data['priority'],
            'network_activity_MB':0,
            'time_of_operation': input_data['time_of_operation'],
            'idle_time': input_data['idle_time'],
            'power_source_Battery': 0,
            'power_source_Solar': 0,
        }
        svm_df = pd.DataFrame([svm_features])

        # Make predictions
        predicted_consumption = xgb_model.predict(xgb_df)[0]
        svm_scaled = scaler.transform(svm_df)
        predicted_category_num = svm_model.predict(svm_scaled)[0]
        
        # Convert numeric prediction to category
        category_mapping = {0: "Low", 1: "Medium", 2: "High"}
        predicted_category = category_mapping.get(predicted_category_num, "Unknown")

        # Generate recommendations
        schedule, suggestion = get_recommendation(
            predicted_category, 
            input_data['priority'], 
            input_data['time_of_operation']
        )

        return jsonify({
            'success': True,
            'predicted_consumption': float(predicted_consumption),
            'predicted_category': predicted_category,
            'schedule': schedule,
            'suggestion': suggestion
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_recommendation(predicted_label, priority, time_of_operation):
    """Generate scheduling recommendations based on predictions"""
    priority = float(priority)
    time_of_operation = float(time_of_operation)
    
    if predicted_label == "High":
        if priority == 0 and time_of_operation == 0:
            return ("Off-Peak Hours", 
                   "High consumption but already in Off-Peak Hours. Optimize workload.")
        elif priority == 0:
            return ("Off-Peak Hours", "High consumption. Schedule in Off-Peak Hours.")
        elif priority == 1:
            return ("Off-Peak Hours", "High consumption. Medium priority suggests Off-Peak.")
        else:
            return ("Peak Hours", "High consumption. High priority requires Peak Hours.")

    elif predicted_label == "Medium":
        if priority == 0 and time_of_operation == 0:
            return ("Off-Peak Hours", 
                   "Medium consumption in Off-Peak. Consider energy-saving measures.")
        elif priority == 0:
            return ("Off-Peak Hours", "Moderate consumption. Prefer Off-Peak Hours.")
        elif priority == 1:
            return ("Off-Peak Hours", "Moderate consumption. Medium priority allows Off-Peak.")
        else:
            return ("Peak Hours", "Moderate consumption. High priority needs Peak Hours.")
    
    else:  # Low
        if time_of_operation == 0:
            return ("Off-Peak Hours", "Low consumption. Already optimal in Off-Peak.")
        else:
            return ("Off-Peak Hours", "Low consumption. Flexible but Off-Peak is optimal.")

@app.route('/api/status', methods=['GET'])
def status():
    """API health check endpoint"""
    return jsonify({
        'status': 'online',
        'models_loaded': all(m is not None for m in [xgb_model, svm_model, scaler]),
        'xgb_model': xgb_model is not None,
        'svm_model': svm_model is not None,
        'scaler': scaler is not None
    })

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
