from flask import *
import sqlite3, hashlib, os
from werkzeug.utils import secure_filename
import requests


app = Flask(__name__)
app.secret_key = 'random string'
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = set(['jpeg', 'jpg', 'png', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template("upload.html")

@app.route('/getratio',methods=['GET', 'POST'])
def calculateRatios():
    image = request.files['pimage']
    if image:
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    imagename = filename
    return modelratiofetcher(imagename)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
def modelratiofetcher(imagename):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model
    model = tensorflow.keras.models.load_model('models\keras_model.h5')
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image_nameee = "static/upload/" + imagename
    image = Image.open(image_nameee)
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # display the resized image
    #image.show()
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model.predict(data)
    print(prediction)
    labels = {
        0:"Ariel",
        1:"Tide",
        2:"gillette"
    }
    test = prediction[0]
    print(test[2])
    # new_object = []
    # for keys in labels.keys():
    #     new_object.append({round(test[keys]*100,4) : ""+labels.get(keys)})
    #     print(round(test[keys]*100,4),labels.get(keys))
    return ' '.join([str(elem) for elem in test])




if __name__ == '__main__':
    app.run(debug=True)
