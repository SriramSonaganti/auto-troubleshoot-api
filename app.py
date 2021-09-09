from PIL import Image, ImageOps
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from numpy import asarray

from Yolo.yolo_detection_images import detectWifiRouter
import numpy as np
import cv2
import librosa
from keras.models import load_model

app = Flask(__name__)

ans=[]
classes=["Cpu_Fan_Noise","Ram_Disconnected_Sound"]

model=None

def loadmodel():
    global model
    model=load_model('audio/my_model.h5')


@app.route('/',methods=['POST','GET'])
def hello():
    return "API FOR IMAGE AND AUDIO CLASSIFCATION"


# @app.route('/')
@app.route('/predict-audio',methods=['POST','GET'])
def prediction():
    if request.method=='GET':
        if len(ans)!=0:
            return str(ans[-1])
        else:
            return "send your audio format"
    if request.method=='POST':
        if request.files:
            f=request.files.get('file')
            x, sr = librosa.load(f, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T, axis=0)
            mfccs = mfccs.reshape(1, -1)
            # print(mfccs)
            predict_prob = model.predict(mfccs)
            # print(predict_prob)
            predicted_label = np.argmax(predict_prob, axis=1)
            # print(predicted_label)
            for i in predicted_label:
                print(classes[i])
                ans.append(classes[i])
            return str(ans[-1])
    return "files not saved"



#yolo-v3-detection
@app.route('/predict-image', methods=['POST','GET'])
def detect():
    # img=request.args['image']
    # img_path='images/'+img
    # results=detectWifiRouter(img_path)
    # return jsonify(results)
    if (request.method == 'POST'):
        if request.files:
            #img_path = request.files.get('image')
            #img_path = 'images/1.jpeg'
            image = Image.open(request.files['image'])
            # print(image.format)
            # print(image.size)
            # print(image.mode)
            numpydata = asarray(image)
            # print(type(numpydata))
            #
            # shape
            # print(numpydata.shape)
            result = detectWifiRouter(numpydata)
            return result

        return "file not saved"

#app.run()
#host="0.0.0.0"
if __name__ == "__main__":
    app.debug = False
    loadmodel()
    app.run()
