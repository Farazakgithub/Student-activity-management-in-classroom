import os

import cv2
import numpy as np
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from flask import Flask, jsonify, render_template, request

categories = ["FIGHTING", "SITTING", "SLEEPING", "USE_MOBILE", "WRITING"]


app = Flask(__name__, static_url_path='/static')
CORS(app)

# Load the trained model
model = keras.models.load_model('activity_model_updated.h5')
# Replace with the path to your saved model

# Load label encoder
label_encoder = LabelEncoder()
# np.savetxt('label_encoder.txt', label_encoder.classes_, fmt='%s')
loaded_classes = np.loadtxt('label.txt', dtype=str)

label_encoder.classes_ = loaded_classes
 # Use the correct filename

@app.route('/')
def index():
    print(label_encoder.classes_)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (100, 100))
            image = image / 255.0


            # Predict the category
            predictions = model.predict(np.array([image]))
            predicted_class = np.argmax(predictions)
            predicted_category = label_encoder.classes_[predicted_class]

            # Return the prediction as JSON
            return jsonify({'prediction': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)

