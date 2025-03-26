import paho.mqtt.client as mqtt

BROKER = "localhost"  # Change this to your broker's IP if necessary
PORT = 1883
REQUEST_TOPIC = "image/request"
RESPONSE_TOPIC = "image/response"
IMAGE_FILE = "received_image.jpg"

def on_message(client, userdata, message):
    """Callback function triggered when an image is received."""
    print(f"Received image data on topic '{message.topic}'")

    # Save the received image data
    with open(IMAGE_FILE, "wb") as img_file:
        img_file.write(message.payload)
        print(f"Image saved as {IMAGE_FILE}")

# Set up MQTT client
client = mqtt.Client("Image_Publisher")
client.on_message = on_message
client.connect(BROKER, PORT)
client.subscribe(RESPONSE_TOPIC)

# Request an image by publishing a message
print(f"Requesting an image by publishing to topic '{REQUEST_TOPIC}'...")
client.publish(REQUEST_TOPIC, "capture")

print(f"Listening for images on topic '{RESPONSE_TOPIC}'...")
client.loop_forever()

