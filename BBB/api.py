#!flask/bin/python
from flask import Flask, jsonify
from flask import request
import os
import binascii

import datetime
import json

app = Flask(__name__)

import numpy as np

from scipy import signal
from scipy.stats import describe

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import mean_squared_error
import tensorflow as tf

from joblib import dump, load

from utils import lin_log_interp

basePath = '/home/debian/Git/Edge-Analytics-IoT-Framework/'

pca_gmm_model = load(basePath + "Models/GMM/PCA-GMM.joblib")
autoencoder_lite_model = tf.lite.Interpreter(model_path=basePath + "Models/Autoencoder/Lite/CNN-AE-Lite.tflite")

pca_gnb_model = load(basePath + "Models/GNB/PCA-GNB.joblib")
classifier_lite_model = tf.lite.Interpreter(model_path=basePath + "Models/MLP-Classifier/Lite/CNN-MLP-Lite.tflite")



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

    fftPoints = request.json['fftPoints']
    samplingInterval = request.json['samplingInterval']
    scalingCoeff = request.json['accelerationCoeff1']
    offsetCoeff = request.json['accelerationCoeff0']
    modelId = request.json['modelId']

    output = parse_vibration(scalingCoeff,offsetCoeff,fftPoints,samplingInterval)

    xInference = lin_log_interp(np.array(output['fftAmps']))

    if modelId == 'CNN-AE-Lite':
        value = autoencoder_lite(xInference)
    elif modelId == 'CNN-MLP-Lite':
        value = classifier_lite(xInference)
    elif modelId == 'PCA-GMM':
        value = model_gmm(xInference)
    elif modelId == 'PCA-GNB':
        value = model_gnb(xInference)
    else:
        value = 0.
    
    output['values'] = value

    return jsonify(output), 201

def parse_vibration(scalingCoeff,
                    offsetCoeff,
                    fftPoints,
                    samplingInterval,
                    returnVib=False):

    f = open('/usr/local/lib/node_modules/node-red/output.0','rb')

    raw_data = f.read()

    data = np.frombuffer(raw_data,dtype=np.uint16).astype(float)
    data = (scalingCoeff * data) + offsetCoeff

    _,minmax,mean,variance,skewness,kurtosis = describe(data)

    freqs,amps = signal.welch(data, fs=1 / samplingInterval, nperseg=fftPoints, scaling='spectrum')
    frequencyInterval = freqs[1] - freqs[0]

    sampleRMS = np.sqrt(1 / data.shape[0] * np.sum((data - mean)**2))

    output = {'frequencyInterval':frequencyInterval,
              'fftAmps':amps.tolist(),
              # 'Vibration':data.tolist(),
              'RMS':sampleRMS,
              'Kurtosis':kurtosis,
              'Mean':mean,
              'Skewness':skewness,
              'Variance':variance}

    return output


def autoencoder_lite(xInference):
    

    global autoencoder_lite_model
    interpreter = autoencoder_lite_model

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    fft_dimension = np.amax(input_shape)
    
    input_data = xInference[:fft_dimension].reshape(input_shape).astype(np.float32)

    num_samples = 1

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index']).reshape(input_shape)

    input_data = np.repeat(input_data,num_samples,axis=0)

    mse = mean_squared_error(output_data,input_data).numpy().flatten()[0].astype(float)

    return mse


def classifier_lite(xInference):

    global classifier_lite_model
    interpreter = classifier_lite_model
    
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test model on random input data.
    input_shape = input_details[0]['shape']
    fft_dimension = np.amax(input_shape)
    
    input_data = xInference[:fft_dimension].reshape(input_shape).astype(np.float32)
    output_shape = output_details[0]['shape']

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index']).flatten()
    
    classification = output_data[0].astype(float)

    return classification



def model_gmm(xInference):
    '''
    Perform inference on a PCA-GMM model
    '''
    
    global pca_gmm_model
    model = pca_gmm_model
    
    fft_dimension = model[0].n_features_

    log_likelihood = model.score_samples(np.expand_dims(xInference[:fft_dimension],axis=0))

    return log_likelihood.flatten()[0].astype(float)

def model_gnb(xInference):
    '''
    Perform inference on a PCA-GNB model
    '''

    global pca_gnb_model
    model = pca_gnb_model
    
    fft_dimension = model[0].n_features_

    classification = model.predict_proba(np.expand_dims(xInference[:fft_dimension],axis=0)).flatten()[0]

    return classification


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

