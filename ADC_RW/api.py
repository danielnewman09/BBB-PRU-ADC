#!flask/bin/python
from flask import Flask, jsonify
from flask import request
import os
import binascii

from subprocess import call

import datetime

import json

app = Flask(__name__)

import numpy as np

from scipy import signal
from scipy.stats import describe

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import mean_squared_error
#import tflite_runtime.interpreter as tflite
import tensorflow as tf
# import tensorflow.keras as keras
# from Custom_Layers import Dropout_Live

from joblib import dump, load

pca_gmm_model = None
cnn_ae_model = None
ae_model = None
cnn_ae_lite_model = None

pca_gnb_model = None
mlp_model = None
cnn_mlp_model = None
cnn_mlp_lite_model = None


cwd = os.path.dirname(os.path.abspath(__file__))

import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
broker_url = "mqtt.eclipse.org"
broker_port = 1883
client = mqtt.Client()

PRELOAD_MODELS = True

basePath = '/home/daniel/Git/Edge-Analytics-IoT-Framework/'
#basePath = '/home/debian/Git/Edge-Analytics-IoT-Framework/EMCO-Case-Study/'

#basePath = '/home/dnewman/Documents/Github/Edge-Analytics-IoT-Framework/'

if PRELOAD_MODELS == True:

    pca_gmm_model = load(basePath + "Models/GMM/PCA-GMM.joblib")
    #cnn_ae_model = load_model(basePath + "Models/Autoencoder/Full/CNN-AE.h5")
    #ae_model = load_model(basePath + "Models/Autoencoder/Full/AE.h5")
    cnn_ae_lite_model = tf.lite.Interpreter(model_path=basePath + "Models/Autoencoder/Lite/CNN-AE-Lite.tflite")

    pca_gnb_model = load(basePath + "Models/GNB/PCA-GNB.joblib")
    #mlp_model = load_model(basePath + "Models/MLP-Classifier/Full/MLP.h5")
    #cnn_mlp_model = load_model(basePath + "Models/MLP-Classifier/Full/CNN-MLP.h5")
    cnn_mlp_lite_model = tf.lite.Interpreter(model_path=basePath + "Models/MLP-Classifier/Lite/CNN-MLP-Lite.tflite")



@app.route('/models/save',methods=['POST'])
def save_model():

    hex_vals = request.json['values']
    file_path = request.json['path']
    file_name = request.json['filename']

    if not os.path.exists(file_path):
        os.makedirs(file_path)


    with open(file_path + file_name,'wb') as fout:
        fout.write(binascii.unhexlify(hex_vals))

    return jsonify({'Output':True}),201


@app.route('/features/vibration/inference',methods=['POST'])
def inference_vibration_route():
    receive_time = int(datetime.datetime.now().timestamp() * 1000)
    fftPoints = request.json['fftPoints']
    samplingInterval = request.json['samplingInterval']
    scalingCoeff = request.json['accelerationCoeff1']
    offsetCoeff = request.json['accelerationCoeff0']
    modelId = request.json['modelId']
    returnPSD = request.json['returnPSD']

    output = parse_vibration(scalingCoeff,offsetCoeff,fftPoints,samplingInterval)

    xInference = np.array(output['fftAmps'][:1024])

    parse_time = int(datetime.datetime.now().timestamp() * 1000)

    if modelId == 'CNN-AE-Lite':
        value = model_inference_lite(xInference)
    elif modelId == 'CNN-MLP-Lite':
        value = classifier_inference_lite(xInference)
    elif modelId == 'PCA-GMM':
        value = model_gmm(xInference)
    elif modelId == 'PCA-GNB':
        value = model_gnb(xInference)
    else:
        value = 0.
    
    output['modelId'] = modelId
    output['values'] = value

    inference_time = int(datetime.datetime.now().timestamp() * 1000)

    if returnPSD:
        pass
    else:
        output['fftAmps'] = 0.

    output['Vibration'] = 0.
    output['parseTime'] = int(parse_time - receive_time)
    output['inferenceTime'] = int(inference_time - parse_time)

    return jsonify(output), 201

@app.route('/features/parse/rawvib',methods=['POST'])
def parse_raw_vibration_route():
    
    fftPoints = request.json['fftPoints']
    samplingInterval = request.json['samplingInterval']
    scalingCoeff = request.json['accelerationCoeff1']
    offsetCoeff = request.json['accelerationCoeff0']
    
    
    f = open('/usr/local/lib/node_modules/node-red/output.0','rb')

    raw_data = f.read()

    data = np.frombuffer(raw_data,dtype=np.uint16).astype(float)
    data = (scalingCoeff * data) + offsetCoeff
    

    #output = parse_vibration(scalingCoeff,offsetCoeff,fftPoints,samplingInterval)
    
    payload = {}
    
    payload['values'] = raw_data
    #payload['dateTime-Sent'] = int(datetime.datetime.now().timestamp() * 1000)
    #payload['modelId'] = 'CNN-AE-Lite'
    #payload['fftPoints'] = fftPoints
    #payload['samplingInterval'] = samplingInterval
    #payload['accelerationCoeff1'] = scalingCoeff
    #payload['accelerationCoeff0'] = offsetCoeff
    #client.connect(broker_url,broker_port)
    #publish.single("Asset/Chapter5/Vibration/Amazon-EC2",str(payload),hostname="mqtt.eclipse.org")
    
    #output = {'success':True}
    output = payload
    return raw_data, 201
    

def parse_vibration_remote(data,fftPoints,samplingInterval,scalingCoeff,offsetCoeff):

    data = (scalingCoeff * data) + offsetCoeff

    _,minmax,mean,variance,skewness,kurtosis = describe(data)

    NyquistFrequency = 0.5 / samplingInterval

    freqs,amps = signal.welch(data, fs=1 / samplingInterval, nperseg=fftPoints, scaling='spectrum')

    frequencyInterval = freqs[1] - freqs[0]
    # amps = lin_log_interp(amps)

    sampleRMS = np.sqrt(1 / data.shape[0] * np.sum((data - mean)**2))

    output = {'frequencyInterval':frequencyInterval,
              'fftAmps':amps[1:].tolist(),
              'Vibration':data.tolist(),
              'RMS':sampleRMS,
              'Kurtosis':kurtosis,
              'Mean':mean,
              'Skewness':skewness,
              'Variance':variance}
    return output

def parse_vibration(scalingCoeff,offsetCoeff,fftPoints=None,samplingInterval=None,returnVib=False):

    f = open('/usr/local/lib/node_modules/node-red/output.0','rb')

    raw_data = f.read()

    data = np.frombuffer(raw_data,dtype=np.uint16).astype(float)
    data = (scalingCoeff * data) + offsetCoeff

    _,minmax,mean,variance,skewness,kurtosis = describe(data)
    if fftPoints != None and samplingInterval != None:
        freqs,amps = signal.welch(data, fs=1 / samplingInterval, nperseg=fftPoints, scaling='spectrum')
        frequencyInterval = freqs[1] - freqs[0]
    else:
        freqs = np.array([])
        amps = np.array([])
        frequencyInterval = 0.

    sampleRMS = np.sqrt(1 / data.shape[0] * np.sum((data - mean)**2))

    output = {'frequencyInterval':frequencyInterval,
              'fftAmps':amps[1:].tolist(),
              'Vibration':data.tolist(),
              'RMS':sampleRMS,
              'Kurtosis':kurtosis,
              'Mean':mean,
              'Skewness':skewness,
              'Variance':variance}

    return output


def model_inference_lite(xInference):
    if PRELOAD_MODELS == True:
        global cnn_ae_lite_model
        interpreter = cnn_ae_lite_model
    else:
        interpreter = tflite.Interpreter(model_path=basePath + "Models/Autoencoder/Lite/CNN-AE-Lite.tflite")

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = xInference.reshape(input_shape).astype(np.float32)

    num_samples = 1

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index']).reshape(input_shape)

    input_data = np.repeat(input_data,num_samples,axis=0)

    mse = mean_squared_error(output_data,input_data).numpy().flatten()[0].astype(float)

    return mse


def classifier_inference_lite(xInference):

    xInference = np.atleast_2d(xInference)

    if PRELOAD_MODELS:
        global cnn_mlp_lite_model
        interpreter = cnn_mlp_lite_model
    else:
        global basePath
        interpreter = tflite.Interpreter(model_path=basePath + "Models/MLP-Classifier/Lite/CNN-MLP-Lite.tflite")

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = xInference.reshape(input_shape).astype(np.float32)
    output_shape = output_details[0]['shape']


    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index']).flatten()
    
    classification = output_data[0].astype(float)

    return classification



def model_gmm(xInference):

    if PRELOAD_MODELS:
        global pca_gmm_model
        model = pca_gmm_model
    else:
        global basePath
        model = load(basePath + "Models/GMM/PCA-GMM.joblib")

    log_likelihood = model.score_samples(np.expand_dims(xInference,axis=0))

    return log_likelihood.flatten()[0].astype(float)



def model_gnb(xInference):

    if PRELOAD_MODELS:
        global pca_gnb_model
        model = pca_gnb_model
    else:
        global basePath
        model = load(basePath + "Models/GNB/PCA-GNB.joblib")

    classification = model.predict_proba(xInference).flatten()[0]

    return classification


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

