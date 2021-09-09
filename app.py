from PIL import Image, ImageOps
from flask import Flask, jsonify, request
from yolo_detection_images import detectWifiRouter
from numpy import asarray
import numpy as np
import librosa
import tensorflow as tf

app = Flask(__name__)


@app.route('/')
def hello():
    return "API FOR IMAGE AND AUDIO CLASSIFCATION {made by patient care dev team}"


ans=[]
classes=["Cpu_Fan_Noise","Ram_Disconnected_Sound"]


model = tf.saved_model.load('model')

def loadmodel():
    global model
    #tf.saved_model.load('model')
    model = tf.saved_model.load('model')


@app.route('/predict-audio',methods=['POST','GET'])
def prediction():
    
    if request.method=='GET':
        if len(ans)!=0:
            return str(ans[-1])
        else:
            return "upload your audio file"
    if request.method=='POST':
        if request.files:
            f=request.files.get('file')
            x, sr = librosa.load(f, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T, axis=0)
            mfccs = mfccs.reshape(1, -1)
            predict_prob = model(mfccs)
            predicted_label = np.argmax(predict_prob, axis=1)
            for i in predicted_label:
                print(classes[i])
                ans.append(classes[i])
            return str(ans[-1])
        return "file not saved"

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
    
