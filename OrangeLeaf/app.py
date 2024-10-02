from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('disease_model.h5')  # Update with the correct model file name

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        file_path = f'static/uploads/{file.filename}'

        # Ensure the directory exists
        os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)

        file.save(file_path)

        img = image.load_img(file_path, target_size=(64, 64))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255

        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        classes = {0: 'Black Spot', 1: 'Canker', 2: 'Greening', 3: 'Healthy', 4:  'Melanose'}  # Update with your actual class names
        predicted_class = classes[class_index]

        # Pass details to result.html
        return render_template('result.html', file_path=file_path, predicted_class=predicted_class)

    return render_template('index.html', file_path=None, predicted_class=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=9090)
