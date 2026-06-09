import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# 1. Initialize the Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Global variable to hold our model
model = None

# 2. Lazy-loader function: Loads the model ONLY when needed to save RAM
def get_model():
    global model
    if model is None:
        print("Lazy Loading MobileNetV2 Model...")
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(3, activation='softmax')
        ])
        model.load_weights('songket_mobilenetv2_model.keras')
        print("Model Loaded Successfully into RAM!")
    return model

# 3. Define class names
CLASS_NAMES = ['Motif Cendawan', 'Motif Pucuk Rebung', 'Motif Tampuk Manggis']

# 4. Route for the Homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# 5. Route for the AI Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0 

        # Run Prediction (Uses lazy-loader)
        current_model = get_model()
        predictions = current_model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence_score = float(np.max(predictions[0])) * 100

        return jsonify({
            'motif': CLASS_NAMES[predicted_class_index],
            'confidence': f"{confidence_score:.2f}%",
            'image_path': filepath
        })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Render assigns its own port, so we use the environment variable PORT or default to 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port)