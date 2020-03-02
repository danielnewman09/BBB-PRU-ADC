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

# import tensorflow as tf
# import tensorflow.keras as keras
# from Custom_Layers import Dropout_Live

keras_model = None
keras_path = None

cwd = os.path.dirname(os.path.abspath(__file__))

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
    amps = lin_log_interp(amps)

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

@app.route('/models/inference/full',methods=['POST'])
def model_inference_full():

    global keras_model
    global keras_path

    assetId = request.json['assetId']
    dataItemId = request.json['dataItemId']
    isWarmUp = request.json['isWarmUp']
    spindleSpeed = request.json['spindleSpeed']
    xInference = np.array(request.json['xInference']).astype(float)
    basePath = request.json['basePath']
    modelId = request.json['modelId']
    feature = request.json['feature']


    model_path = basePath + 'Models/' + assetId + '/' + dataItemId + '/' + str(isWarmUp).lower() + '/' + str(spindleSpeed) + '/'

    if not os.path.exists(model_path):
        return jsonify({'output':False}),201

    with open(model_path + 'control_params_{}_{}_full.json'.format(modelId,feature), 'r') as fp:
        param_dict = json.load(fp)

    if keras_path == model_path:
        new_model = keras_model
    else:
        keras_model = tf.keras.models.load_model(model_path + "model_{}_{}_full.h5".format(modelId,feature),custom_objects={'Dropout_Live': Dropout_Live})
        new_model = keras_model
        keras_path = model_path

    num_samples = 1

    xInference = xInference.reshape(1,512,1)
    X_predict = np.repeat(xInference,num_samples,axis=0)

    predict = new_model.predict(X_predict)

    mse = keras.metrics.mean_squared_error(X_predict,predict)
    means = np.mean(mse,axis=1)
    means = np.mean(means)

    #means = np.mean(mse,axis=1).flatten()
    variances = np.var(mse,axis=1).flatten()

    #print(means)

    #zMeans = means
    zStds = variances
    zMeans = (means - float(param_dict['avgMean'])) / float(param_dict['avgStd'])
    #zStds = (variances - float(param_dict['varMean'])) / float(param_dict['varStd'])

    output = {
        'valueMean':zMeans.tolist(),
        'valueStd':zStds.tolist(),
        'dataItemId':dataItemId,
        'state':spindleSpeed,
        'modelId':modelId,
        'feature':feature
    }

    return jsonify(output), 201


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

