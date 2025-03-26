import time
import paho.mqtt.client as mqtt
import ssl
import json
import _thread as thread
import psutil
import json

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))


client = mqtt.Client()
client.on_connect = on_connect
client.tls_set(ca_certs='./rootCA.pem', certfile='./aws-certificate.pem.crt', keyfile='./aws-private.pem.key', tls_version=ssl.PROTOCOL_SSLv23)
client.tls_insecure_set(True)
client.connect("xxxxxxxx-ats.iot.ap-southeast-1.amazonaws.com", 8883, 60) #Copy end point from your AWS IoT Core Console


def justADummyFunction(Dummy):
    while (1):
        # This is where you can put your edge analytics to generate data from your sensors
        # processed/raw data can be sent to AWS IoT core for further analytics/processing on the cloud
        message = "Hello from INF2009 RaspberryPi Device#1"
        print(message)
        client.publish("device/data", payload=message , qos=0, retain=False)
        time.sleep(5)

thread.start_new_thread(justADummyFunction,("Create Thread",))

client.loop_forever()