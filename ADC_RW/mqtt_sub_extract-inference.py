import paho.mqtt.client as mqtt
import json
from api import *

broker_url = "mqtt.eclipse.org"
broker_port = 1883

def on_connect(client, userdata, flags, rc):
   print("Connected With Result Code "+rc)

def on_message(client, userdata, message):
   msg = message.payload.decode("utf-8")
   msg_json = msg.replace("'", "\"")
   msg_json = json.loads(msg_json)

   modelId = msg_json['modelId']
   xInference = np.array(msg_json['values']).astype(np.float32)
   xInference = xInference[:1024]

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

   return_str = '{"modelId": "' + modelId + '" ,"dateTime-Sent":' + str(dateTimeSent) + '}'

   client.publish("Asset/Chapter5/Inference/Amazon-EC2",return_str,qos=0, retain=False)
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker_url, broker_port)

client.subscribe("Asset/Chapter5/Vibration/Amazon-EC2", qos=0)

client.loop_forever()
