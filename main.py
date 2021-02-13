from flask import *
import os
from werkzeug.utils import secure_filename
import cv2 
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import json
import threading

# APP INIT
app = Flask(__name__)
#loads the model
model = tensorflow.keras.models.load_model('models\keras_model.h5')
#Python Flask Secret key
app.secret_key = 'random string'
#Images Upload Folder
UPLOAD_FOLDER = 'static/upload'
#! Allowed image extension - Not used
ALLOWED_EXTENSIONS = set(['jpeg', 'jpg', 'png', 'gif'])
# Adding folder to config var
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



#
#* This route serves the upload.html on request
#
@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template("upload.html")
#
#* This page serves the data prediction on request
#* & write data to a data.json file & plotting graph
#
@app.route('/getratio',methods=['GET', 'POST'])
def calculateRatios():
    try:
        #get this variable from user
        more_data = 1
        image = request.files['pimage']
        if image:
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        imagename = filename
    except Exception:
        return render_template_string("<h2>Some Error Occured</h2>")
    data = productRatio(imagename)
    #if more_data == 1:
        #This will return values into varibales
        # row1, row2, row3 = cropImageIntoRows("static/upload/" + imagename)
        # row1_data = productRatio(row1)
        # row2_data = productRatio(row2)
        # row3_data = productRatio(row3)
        # with open('static/data_row1.json', 'w') as outfile:
        #     json.dump(row1_data, outfile)
        # with open('static/data_row2.json', 'w') as outfile:
        #     json.dump(row2_data, outfile)
        # with open('static/data_row3.json', 'w') as outfile:
        #     json.dump(row3_data, outfile)

    with open('static/data.json', 'w') as outfile:
        json.dump(data, outfile)
    return render_template("data-prediction.html",data=data)

#
#* This method loads the model and & classify the brand and
#* Also gives the probabilistic output on the image
#
def productRatio(imagename):
    try:
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        # Load the model
        global model
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
            0: "Ariel",
            1:  "Tide",
            2:  "horlicks",
            3 : "bournvita",
            4 : "complan",
            5 : "protienx",
            6 : "societyTea",
            7:  "TataTea",
            8 : "waghbakri",
            9 : "head_and_sho",
            10:  "pantene",
            11:  "treseme",
            12 : "ponds_white_",
            13:  "fair_and_lov",
            14 : "vaselline",
            15 : "exo",
            16:  "pril_bar",
            17 : "pril_liquid",
            18 : "vim_liquid",
            19 : "lizol",
            20 : "lifebuoy_han",
            21:  "santoor_hand",
            22:  "domex",
            23 : "harpic",
            24 : "kurkure",
            25 : "doritos",
            26 : "mad_angles",
            27 : "cornitos",
            28 : "pears",
            29 : "lux",
            30 : "lifebuoy",
            31:  "medimix",
            32 : "dove",
            33:  "dettol",
            34 :  "margo",
            35 : "vim_bar",
            36 :  "lays"
        }
        # with open("labels.txt") as f:
        #     for line in f:
        #         (key, val) = line.split()
        #         labels[int(key)] = val
        test = prediction[0]
        #delete file after work done
        os.remove(image_nameee)
        #print(test[2])
        data = {}
        #* JSON object is created
        for keys in labels.keys():
            #new_object.append({round(test[keys]*100,4) : ""+labels.get(keys)})
            data[labels.get(keys)] = round(test[keys]*100,4)
            print(round(test[keys]*100,4),labels.get(keys))
    except Exception:
        return {1:"Some Error Occured!"}
    return data

def convertVideoToImage(video_source):
    """
    This Function converts the video into frames and then into images.
    """
    # Read the video from specified path 
    cam = cv2.VideoCapture("test_video.mp4") 
    try: 
        # creating a folder named data 
        if not os.path.exists('data'): 
            os.makedirs('data')    
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data') 
    # frame 
    currentframe = 0
    while(True):     
        # reading from frame 
        ret,frame = cam.read() 
        if ret:
            # if video is still left continue creating images 
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name) 
            # writing the extracted images 
            cv2.imwrite(name, frame) 
            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 30
            cam.set(1,currentframe)
        else:          
            break   
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 

def cropImageIntoRows(image_source):
    """
    This will crop the images and return the tuple of file names
    """
    try:
        image = Image.open(image_source)
        image1 =image
        image2 = image
        width, height = image.size

        row1 = (0,0,width,height//3)

        row2 = (0,height//3,width,(height//3)*2)

        row3 = (0,(height//3)*2,width,height)

        filename1  = 'static\cropped\saverow1.jpg'
        filename2 = 'static\cropped\saverow2.jpg'
        filename3 = 'static\cropped\saverow3.jpg'

        cropped_image = image.crop(row1)
        cropped_image.save(filename1)

        cropped_image1 = image1.crop(row2)
        cropped_image1.save(filename2)

        cropped_image2 = image2.crop(row3)
        cropped_image2.save(filename3)

        ## Remove comments to auto delete the file
        #os.remove(filename)
        #os.remove(filename2)
        #os.remove(filename3)

    except Exception:
        print("Some error occured")
    return (filename1,filename2,filename3)

if __name__ == '__main__':
    app.run(debug=True)