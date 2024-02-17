# -*- coding: utf-8 -*-
# train.py

"""Main Program Execution File for Sugar Beet Segmentation and Counting"""

from ultralytics import YOLO
import os

# Set working direction to YOLOv6 folder
new_cwd = '../../YOLOv6' # Adjust path if needed
os.chdir(new_cwd)

# Build a YOLOv6n model from scratch
model = YOLO('yolov6n.yaml')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs

results = model.train(data='data/dataset.yaml', epochs=5, imgsz=512)

