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

# 2. THE BULLETPROOF FIX: Build architecture in code, load weights only!
print("Constructing MobileNetV2 Architecture...")
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')
])

print("Injecting Trained Weights...")
# This completely ignores Colab's version config and only extracts the learned math!
model.load_weights('songket_mobilenetv2_model.keras')
print("Model Ready for Deployment!")

# 3. Define the exact class names (Alphabetical)
CLASS_NAMES = ['Motif Cendawan', 'Motif Pucuk Rebung', 'Motif Tampuk Manggis']

# 4. Route for the Homepage (Frontend)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# 5. Route for the AI Prediction API (Backend)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        # Save the uploaded image temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale to 0-1

        # Run the AI Prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence_score = float(np.max(predictions[0])) * 100

        # Return the result back to the website
        return jsonify({
            'motif': CLASS_NAMES[predicted_class_index],
            'confidence': f"{confidence_score:.2f}%",
            'image_path': filepath
        })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=8080)