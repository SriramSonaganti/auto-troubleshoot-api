from PIL import Image, ImageOps
from flask import Flask, jsonify, request
from numpy import asarray
import numpy as np
import librosa
import tensorflow as tf

app = Flask(__name__)


@app.route('/')
def hello():
    return "API FOR IMAGE AND AUDIO CLASSIFCATION {made by patient care dev team}"


@app.route('/predict-image', methods=['POST','GET'])
def detect():
    if (request.method == 'GET'):
        return 'hello'
    if (request.method == 'POST'):
        if request.files:
            image = Image.open(request.files['image'])
            numpydata = asarray(image)
            result = detectWifiRouter(numpydata)
            return result
        return "file not saved"
    

if __name__ == "__main__":
    app.run(debug=True)
    
