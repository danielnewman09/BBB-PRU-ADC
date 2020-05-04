import paho.mqtt.client as mqtt

broker_url = "mqtt.eclipse.org"
broker_port = 1883

client = mqtt.Client()
client.connect(broker_url, broker_port)

client.publish(topic="TestingTopic", payload="TestingPayload", qos=1, retain=False)
