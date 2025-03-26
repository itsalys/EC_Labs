import cv2
import paho.mqtt.client as mqtt
import time

# MQTT Configuration
BROKER = "localhost"  # Change if needed
PORT = 1883
REQUEST_TOPIC = "image/request"
RESPONSE_TOPIC = "image/response"
IMAGE_FILE = "captured_image.jpg"

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)  # Try 1 or -1 if 0 doesn't work

if not cap.isOpened():
    raise IOError("Cannot open webcam")

def capture_photo():
    """Ensures a valid image is captured before saving."""
    print("Warming up camera...")

    time.sleep(3)  # Wait for the camera to adjust

    for _ in range(10):  # Capture multiple frames to stabilize
        ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: No valid frame received from the camera.")
        return False

    cv2.imwrite(IMAGE_FILE, frame)
    print(f"Image saved as {IMAGE_FILE}")
    return True

def publish_image(client):
    """Reads and publishes the captured image to MQTT."""
    with open(IMAGE_FILE, "rb") as img_file:
        img_data = img_file.read()
        client.publish(RESPONSE_TOPIC, img_data)
        print(f"Image published to topic '{RESPONSE_TOPIC}'")

def on_message(client, userdata, message):
    """Callback when an MQTT message is received."""
    print(f"Received request on topic '{message.topic}': {message.payload.decode()}")

    if capture_photo():
        publish_image(client)

# Setup MQTT Client
client = mqtt.Client("MQTT_Camera_Subscriber")
client.on_message = on_message
client.connect(BROKER, PORT)
client.subscribe(REQUEST_TOPIC)

print(f"Listening for image requests on '{REQUEST_TOPIC}'...")
client.loop_forever()

# Release the camera when the script exits
cap.release()
cv2.destroyAllWindows()
