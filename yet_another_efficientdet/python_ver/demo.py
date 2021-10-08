import cv2
import numpy as np
import time
import os
 
# モジュール読み込み 
import sys
from predictor import Predictor
sys.path.append('/opt/intel/openvino/python/python3.5/armv7l')
from openvino.inference_engine import IENetwork, IEPlugin

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
obj_list = ["trafficcone", "can", "box"]
threshold = 0.3
iou_threshold = 0.2
input_size = 512
windowwidth=1024
windowheight=1024
ver = "d0"
model_dir = os.path.join("..", "model", ver)
model_name = "test"

# ターゲットデバイスの指定 
plugin = IEPlugin(device="MYRIAD")
net = IENetwork(model=os.path.join(model_dir, model_name+".xml"), weights=os.path.join(model_dir, model_name+".bin"))

exec_net = plugin.load(network=net)
input_blob_name = list(net.inputs.keys())[0]
output_blob_name = sorted(list(net.outputs.keys()))

print("input", input_blob_name)
print("output", output_blob_name)

predictor = Predictor(exec_net, image_size = input_size, input = input_blob_name, output = output_blob_name)

cap = cv2.VideoCapture(0)

if cap.isOpened() != True:
    print("camera open error!")
    quit()
else:
    print("camera open!")


cap.set(cv2.CAP_PROP_FRAME_WIDTH, windowwidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, windowheight)

while True:
    ret, frame = cap.read()
    # Reload on error 
    if ret == False:
        print("error")
        continue

    out = predictor.predict(frame,1, threshold)
    cv2.imshow('frame', out)