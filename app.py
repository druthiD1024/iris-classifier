import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

# ─────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────
app    = Flask(__name__, static_folder='.')
model  = joblib.load('iris_model.pkl')
scaler = joblib.load('iris_scaler.pkl')

CLASS_NAMES = ['setosa', 'versicolor', 'virginica']
FEATURES    = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

print("✅ Model & scaler loaded successfully", flush=True)

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'iris_svm_v1'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400

    # Validate all 4 features present
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        features = [[
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width']),
        ]]
    except ValueError:
        return jsonify({'error': 'All values must be numbers'}), 400

    # Scale & predict
    features_sc = scaler.transform(features)
    pred        = model.predict(features_sc)[0]
    probs       = model.predict_proba(features_sc)[0]

    return jsonify({
        'prediction':   CLASS_NAMES[pred],
        'class_id':     int(pred),
        'confidence':   round(float(probs[pred]), 4),
        'probabilities': {
            'setosa':     round(float(probs[0]), 4),
            'versicolor': round(float(probs[1]), 4),
            'virginica':  round(float(probs[2]), 4),
        }
    }), 200


if __name__ == '__main__':
    print("🚀 Starting Iris Classifier API...")
    print("   GET  http://localhost:5000/health")
    print("   POST http://localhost:5000/predict")
    app.run(debug=True, port=5000)
