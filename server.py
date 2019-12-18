#!/usr/bin/env python3 

# import the necessary packages
import tensorflow as tf 
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import sys
import ast
from keras.utils.np_utils import to_categorical

app = flask.Flask(__name__)
model=None
labels=dict()

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_labels(label_path):
    global labels
    with open(label_path,"r")as f:
        for line in f:
            d=dict(ast.literal_eval(line))
            labels.update(d)
    
    return labels            

    
def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image
    
    
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(128, 128))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict_proba(image)
            pred_classes = np.argmax(preds)
            
            label_class=labels[pred_classes]
            data["predictions"] = []
            r={"label": label_class,"probability":  str(preds[0][pred_classes])}
            data["predictions"].append(r)
          
            # indicate that the request was a success
            data["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    model = tf.keras.models.load_model(sys.argv[1])
    labels = load_labels(sys.argv[2])
    app.run()
        
