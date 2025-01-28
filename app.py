from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

# Flask app initialize
app = Flask(__name__)
CORS(app)  # CORS enable karein

# Model load karein
model_path = 'disease_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Make sure it is in the correct directory.")

with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Disease Prediction API is running!"

# Prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Request se data read karein
        data = request.json
        
        # Input validation
        if 'symptoms' not in data:
            return jsonify({'error': 'Missing "symptoms" in request data'}), 400
        
        # Symptoms ko numpy array me convert karein
        symptoms = np.array(data['symptoms']).reshape(1, -1)
        
        # Model se prediction karein
        prediction = model.predict(symptoms)
        return jsonify({'disease': prediction[0]})  # Predicted disease return karein
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)