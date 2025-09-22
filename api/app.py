from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import sys
import os

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from neural_network import NeuralNetwork

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Load the trained model
model = None

def load_model():
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'model.pkl')
        model = NeuralNetwork.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        image_data = data['image']
        
        # Convert to numpy array and normalize
        image_array = np.array(image_data, dtype=np.float32)
        image_array = image_array.reshape(1, 784) / 255.0
        
        # Get prediction and probabilities
        prediction = model.predict(image_array)[0]
        probabilities = model.predict_proba(image_array)[0]
        
        # Convert probabilities to percentages
        confidence_scores = {str(i): float(probabilities[i] * 100) for i in range(10)}
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(probabilities[prediction] * 100),
            'all_scores': confidence_scores
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
