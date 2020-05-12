import paho.mqtt.client as mqtt
import json
from api import *
import datetime

broker_url = "mqtt.eclipse.org"
broker_port = 1883

def on_connect(client, userdata, flags, rc):
   print("Connected With Result Code "+rc)

def on_message(client, userdata, message):
   receive_time = int(datetime.datetime.now().timestamp() * 1000)
   msg = message.payload.decode("utf-8")
   msg_json = msg.replace("'", "\"")
   msg_json = json.loads(msg_json)


   modelId = msg_json['modelId']
   print(np.frombuffer(bytearray(msg_json['values']['data']),dtype=np.uint16))
   xInference = np.frombuffer(bytearray(msg_json['values']['data']),dtype=np.uint16).astype(np.float32)
   print(xInference[:10])
   #xInference = np.array(msg_json['values']).astype(np.float32)
   fftPoints = msg_json['fftPoints']
   samplingInterval = msg_json['samplingInterval']
   scalingCoeff = msg_json['accelerationCoeff1']
   offsetCoeff = msg_json['accelerationCoeff0']

   xInference = parse_vibration_remote(xInference,fftPoints,samplingInterval,scalingCoeff,offsetCoeff)['fftAmps']

   xInference = np.array(xInference[:1024])

   print(xInference)

   dateTimeSent = msg_json['dateTime-Sent']

   if modelId == 'PCA-GMM':
      returnval = model_gmm(xInference)
   elif modelId == 'PCA-GNB':
      returnval = model_gnb(xInference)
   elif modelId == 'CNN-AE-Lite':
      returnval = model_inference_lite(xInference)
   elif modelId == 'CNN-MLP-Lite':
      returnval = classifier_inference_lite(xInference)

   print(returnval)

   send_time = int(datetime.datetime.now().timestamp() * 1000)
   compute_time = send_time - receive_time
   return_str = '{"modelId": "' + modelId + '" ,"dateTime-Sent":' + str(dateTimeSent) + ' ,"computeTime":' + str(compute_time)  + '}'

   client.publish("Asset/Chapter5/Inference/Amazon-EC2",return_str,qos=0, retain=False)
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker_url, broker_port)

client.subscribe("Asset/Chapter5/Vibration/Amazon-EC2", qos=0)

client.loop_forever()
