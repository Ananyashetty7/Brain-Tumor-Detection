from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('brain_tumor_model.keras')  # or .h5

IMG_SIZE = 150

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, IMG_SIZE, IMG_SIZE, 3)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected")
        
        img_bytes = file.read()
        img_array = prepare_image(img_bytes)
        pred_prob = model.predict(img_array)[0][0]
        
        if pred_prob > 0.5:
            prediction = f"Brain Tumor Detected (Confidence: {pred_prob:.2f})"
        else:
            prediction = f"No Brain Tumor Detected (Confidence: {1 - pred_prob:.2f})"
        
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
