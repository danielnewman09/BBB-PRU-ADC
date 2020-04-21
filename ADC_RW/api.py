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

DESKTOP = True
PRELOAD_MODELS = False
basePath = '/home/debian/'


if PRELOAD_MODELS == True:

    pca_gmm_model = load(basePath + "Models/GMM/PCA-GMM.joblib")
    #cnn_ae_model = load_model(basePath + "Models/Autoencoder/Full/CNN-AE.h5")
    #ae_model = load_model(basePath + "Models/Autoencoder/Full/AE.h5")
    cnn_ae_lite_model = tflite.Interpreter(model_path=basePath + "Models/Autoencoder/Lite/CNN-AE-Lite.tflite")

    pca_gnb_model = load(basePath + "Models/GNB/PCA-GNB.joblib")
    #mlp_model = load_model(basePath + "Models/MLP-Classifier/Full/MLP.h5")
    #cnn_mlp_model = load_model(basePath + "Models/MLP-Classifier/Full/CNN-MLP.h5")
    cnn_mlp_lite_model = tflite.Interpreter(model_path=basePath + "Models/MLP-Classifier/Lite/CNN-MLP-Lite.tflite")


@app.route('/features/capture/vibration',methods=['POST'])
def capture_vibration():
    samplePoints = request.json['samplePoints']
    samplingInterval = float(request.json['samplingInterval'])
    fftPoints = request.json['fftPoints']

    samplingIntervalNS = int(samplingInterval * 1e9)

    call(['./rb_file',str(samplePoints),str(samplingIntervalNS)])

    raw_file = open(cwd + '/output.0','rb')
    raw_data = raw_file.read()
    numpy_data = np.frombuffer(raw_data,dtype=np.uint16)
    numpy_data = np.atleast_2d(numpy_data)

    np.savetxt(cwd + '/output.0.txt',numpy_data,delimiter=',')

    return parse_data(numpy_data.flatten(),fftPoints,samplingInterval),201
    # return jsonify({'Vibration':numpy_data.flatten().tolist()}),201

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

def parse_data(data,fftPoints,samplingInterval):

    _,minmax,mean,variance,skewness,kurtosis = describe(data)

    minimum = minmax[0]
    maximum = minmax[1]

    freqs,amps = signal.welch(data, fs=1 / samplingInterval, nperseg=fftPoints, scaling='spectrum')

    frequencyInterval = freqs[1] - freqs[0]

    sampleRMS = np.sqrt(1 / data.shape[0] * np.sum((data - mean)**2))

    output = {'frequencyInterval':frequencyInterval,
              'fftAmps':amps.tolist(),
              'fftFreq':freqs.tolist(),
              'Vibration':data.tolist(),
              'RMS':float(sampleRMS),
              'Kurtosis':float(kurtosis),
              'Mean':float(mean),
              'Skewness':float(skewness),
              'Variance':float(variance),
              'Minimum':float(minimum),
              'Maximum':float(maximum)}

    return jsonify(output)

@app.route('/features/parse/vibration',methods=['POST'])
def parse_vibration():
    data = np.array(request.json['values']).astype(float)
    fftPoints = request.json['fftPoints']
    samplingInterval = request.json['samplingInterval']

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
    return jsonify(output), 201

@app.route('/models/autoencoder/lite',methods=['POST'])
def model_inference_lite():

    xInference = np.array(request.json['values']).astype(np.float32)
    modelId = request.json['modelId']

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

    output = {
        'values':mse,
    }

    return jsonify(output), 201


@app.route('/models/mlp-classifier/lite',methods=['POST'])
def classifier_inference_lite():

    xInference = np.atleast_2d(np.array(request.json['values']).astype(np.float32))
    modelId = request.json['modelId']

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

    output = {
        'values':classification,
    }

    return jsonify(output), 201

@app.route('/models/gmm',methods=['POST'])
def model_gmm():

    xInference = np.array(request.json['values']).astype(np.float32)
    modelId = request.json['modelId']

    if PRELOAD_MODELS:
        global pca_gmm_model
        model = pca_gmm_model
    else:
        global basePath
        model = load(basePath + "Models/GMM/PCA-GMM.joblib")

    log_likelihood = model.score_samples(np.expand_dims(xInference,axis=0))

    output = {
        'values':log_likelihood.flatten()[0].astype(float)
    }

    return jsonify(output),201

@app.route('/models/gnb',methods=['POST'])
def model_gnb():

    xInference = np.atleast_2d(np.array(request.json['values']).astype(np.float32))
    modelId = request.json['modelId']

    if PRELOAD_MODELS:
        global pca_gnb_model
        model = pca_gnb_model
    else:
        global basePath
        model = load(basePath + "Models/GNB/PCA-GNB.joblib")

    classification = model.predict_proba(xInference).flatten()[0]

    output = {
        'values':classification
    }

    return jsonify(output),201


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

