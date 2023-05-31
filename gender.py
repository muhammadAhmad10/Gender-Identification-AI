from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os


model = load_model('./genderClassificationUpgrade1.h5')

app = Flask(__name__)


# @app.route('/', methods=['GET', 'POST'])
@app.route('/')
def home():
    return render_template('UI.html')


@app.route('/UI', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']

        # Save the uploaded file
        upload_dir = 'image'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        image_path = os.path.join(upload_dir, file.filename)
        file.save(image_path)

        # Load and preprocess the image
        # Adjust the target size according to your model's input requirements
        target_size = (64, 64)
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32')
        img_array /= 255.0  # Normalize the image data

        # Make the prediction
        prediction = model.predict(img_array)
        if prediction[0] < 0.5:
            result = "Female"
        else:
            result = "Male"

        return render_template('UI.html', result=result,image_path=image_path)

    else:
        return render_template('UI.html')


if __name__ == '__main__':
    app.run(debug=True)
