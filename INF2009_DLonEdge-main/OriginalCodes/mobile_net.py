import time

import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights

import cv2
from PIL import Image

quantize = False

if quantize:
    torch.backends.quantized.engine = 'qnnpack'


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

weights = MobileNet_V2_QuantizedWeights.DEFAULT
classes = weights.meta["categories"]
net = models.quantization.mobilenet_v2(pretrained=True, quantize=quantize)

started = time.time()
last_logged = time.time()
frame_count = 0

with torch.no_grad():
    while True:
        # read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")

        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        permuted = image

        # preprocess
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # run model
        output = net(input_batch)


        # Uncomment below 5 lines to print top 10 predictions
        #top = list(enumerate(output[0].softmax(dim=0)))
        #top.sort(key=lambda x: x[1], reverse=True)
        #for idx, val in top[:10]:
        #    print(f"{val.item()*100:.2f}% {classes[idx]}")
        #print(f"========================================================================")
        
        # log model performance
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"============={frame_count / (now-last_logged)} fps =================")
            last_logged = now
            frame_count = 0

