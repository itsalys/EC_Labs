## **IoT Communications: MQTT**

**Objective:** To learn how to install and configure an MQTT broker, create MQTT publisher and subscriber clients, and test communication between them on a Raspberry Pi 400.

**Prerequisites:**
1. Raspberry Pi 400 with an operating system (e.g., Raspbian) already set up.
2. Internet connection for the Raspberry Pi.

**Materials:**
1. Raspberry Pi 400.
2. Power supply for Raspberry Pi.
3. Keyboard, mouse, and monitor for Raspberry Pi (if not using SSH).
4. MQTT broker (e.g., Mosquitto).
5. MQTT publisher and subscriber clients (e.g., Python Paho MQTT library).

**Introduction:**
MQTT stands for Message Queue Telemetry Transport. It is a publish/subscribe messaging transport protocol that is lightweight, open, and designed to be easy to implement. These characteristics make it ideal for use in many situations, including constrained environments such as for communication in the Internet of Things (IoT) contexts where a small code footprint is required and network bandwidth is scarce. The protocol runs over TCP/IP, or over other similar network protocols that provide ordered, lossless, bidirectional connections.

MQTT can be broken down into the following components:

**MQTT Broker:** The broker accepts messages from clients (publishers) and then delivers them to any interested clients (subscribers). Each message must belong to a specific topic. The broker is a program running on a device that acts as an intermediary between clients who publish messages and clients who have made subscriptions. 

**Topic:** A namespace (or place) for messages on the broker. Clients must subscribe to and/or publish to a topic.

**MQTT Client:** The client is a ‘device’ that either publishes a message to a specific topic or subscribes to a topic, or both. A publisher is a client that sends a message to the broker, using a topic name. While a subscriber informs the broker which topics it is interested in. Once subscribed, the broker sends messages published to that topic. A client can subscribe to multiple topics. However, a client must first establish a connection with the broker and can take the following actions:
•    Publish messages that other clients might be interested in (subscribed to).
•    Subscribe to a specific topic to receive messages from the publishers.
•    Unsubscribe to remove a request for the messages.
•    Disconnect from the Broker.

![image](https://github.com/drfuzzi/INF2009_MQTT/assets/108112390/26517ab1-d700-48cd-bfbd-4d7511ecfc9a)

Figure 1: An example of MQTT implementation

**Lab Exercise:**


**1. Install and Configure the MQTT Broker on a Raspberry Pi 400:**

   a. Update the Raspberry Pi's package list (optional):
   ```
   sudo apt update
   ```

   b. Install the Mosquitto MQTT broker:
   ```
   sudo apt install mosquitto
   ```

   c. Locate the Mosquitto Configuration File:
   - Use a text editor to open the `mosquitto.conf` file located in `/etc/mosquitto/` using the following command:
     ```
     sudo nano /etc/mosquitto/mosquitto.conf
     ```
   
   d. Edit the Mosquitto Configuration file:
   - Inside the `mosquitto.conf` file, you'll find various configuration options. Be careful when editing this file to avoid introducing errors.
   - Include the following into the configuration file to allow the brokker to listen to port 1883 and allow anonymous clients to connect and use the MQTT broker which means that clients can connect without providing a username and password:
     ```
     listener 1883
     allow_anonymous true    
     ```

   e. Start your mosquitto broker manually:
   ```
   sudo mosquitto -c /etc/mosquitto/mosquitto.conf 
   ```


**2. Enable Mosquitto Broker to run on boot (optional)**

   a. Start and enable Mosquitto to run on boot:
   ```
   sudo systemctl start mosquitto
   sudo systemctl enable mosquitto
   ```

   b. Restarting Mosquitto Broker to apply the new configuration by using the following command:
   ```
   sudo systemctl restart mosquitto
   ```

   c. Verify that Mosquitto is running:
   ```
   systemctl status mosquitto
   ```

   d. Disable and stop the Mosquitto MQTT broker:
   ```
   sudo systemctl disable mosquitto
   sudo systemctl stop mosquitto
   ```


**3. Install and Configure the MQTT Client (Publisher and/or Subscriber) on another Raspberry Pi 400:**

   a. (Remember) to activate the Virtual Environment:
   ```
   source myenv/bin/activate
   ```
   
   b. Install the Python Paho MQTT library for both publisher and subscriber:
   ```
   pip install paho-mqtt
   ```

   c. Create a Python script for the MQTT Publisher (`mqtt_publisher.py`). Ensure the "localhost" is changed to the IP address of the Broker, e.g. "192.168.50.115":
   ```python
   import paho.mqtt.client as mqtt
   import time

   client = mqtt.Client("Publisher")
   client.connect("localhost", 1883)

   while True:
       client.publish("test/topic", "Hello, MQTT!")
       time.sleep(5)
   ```

   d. Create a Python script for the MQTT Subscriber (`mqtt_subscriber.py`). Ensure the "localhost" is changed to the IP address of the Broker, e.g. "192.168.50.115":
   ```python
   import paho.mqtt.client as mqtt

   def on_message(client, userdata, message):
       print(f"Received message '{message.payload.decode()}' on topic '{message.topic}'")

   client = mqtt.Client("Subscriber")
   client.on_message = on_message
   client.connect("localhost", 1883)
   client.subscribe("test/topic")
   client.loop_forever()
   ```


**4. Testing your MQTT Communication:**

   a. Open two terminal windows on the Raspberry Pi.

   b. In the first terminal, run the MQTT subscriber:
   ```
   python3 mqtt_subscriber.py
   ```

   c. In the second terminal, run the MQTT publisher:
   ```
   python3 mqtt_publisher.py
   ```
   
   d. (Remember) to activate the Virtual Environment **before** running the python scripts:
   ```
   source myenv/bin/activate
   ```

   e. Observe the messages being published and received in the subscriber terminal.



**Lab Assignment:**
Create two Python scripts building upon this lab session to capture an image from a webcam when it receives a message through a subscriber topic (pick an appropriate topic) and subsequently transmit the captured image as a publisher via MQTT. This lab assignment will challenge you to combine webcam access, MQTT communication, and message handling to develop a practical IoT application. Your task is to refine the provided script, ensuring seamless image capture and MQTT integration.

![image](https://github.com/drfuzzi/INF2009_MQTT/assets/108112390/bd2e0190-e973-4565-b8bb-1311d804a436)

Figure 2: Overview of the Image Request System via MQTT

