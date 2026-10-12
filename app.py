from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load model (placeholder - replace with actual model path)
model = None
try:
    model = load_model('models/defect_detection_model.h5')
    print("Model loaded successfully")
except:
    print("No model found. Using placeholder inference.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    """Preprocess image for model inference"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_defect(img_array):
    """Predict defect using the model"""
    if model is not None:
        predictions = model.predict(img_array)
        return predictions[0]
    else:
        # Placeholder prediction (random for demo)
        # Returns [defect_probability, normal_probability]
        np.random.seed(hash(img_array.tobytes()) % 2**32)
        defect_prob = np.random.uniform(0.1, 0.9)
        return np.array([defect_prob, 1 - defect_prob])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400
    
    try:
        # Read and process image
        img = Image.open(io.BytesIO(file.read()))
        img_array = preprocess_image(img)
        
        # Run inference
        predictions = predict_defect(img_array)
        
        defect_prob = float(predictions[0])
        normal_prob = float(predictions[1])
        
        # Determine result
        is_defective = defect_prob > 0.5
        confidence = max(defect_prob, normal_prob)
        
        result = {
            'success': True,
            'is_defective': is_defective,
            'defect_probability': round(defect_prob * 100, 2),
            'normal_probability': round(normal_prob * 100, 2),
            'confidence': round(confidence * 100, 2),
            'message': 'Defect detected!' if is_defective else 'No defect detected',
            'defect_types': get_defect_types(defect_prob) if is_defective else []
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

def get_defect_types(probability):
    """Return potential defect types based on probability"""
    if probability > 0.8:
        return ['Crack', 'Surface defect', 'Alignment issue']
    elif probability > 0.6:
        return ['Minor crack', 'Surface wear']
    else:
        return ['Potential surface defect']

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
