from ultralytics import YOLO

# Build a YOLOv6n model from scratch
model = YOLO('yolov6n.yaml')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='YOLOv6/data/dataset.yaml', epochs=10, imgsz=512)

# Run inference with the YOLOv6n model on the 'bus.jpg' image
results = model('data/custom_dataset/images/val/IMG_20230514_155238_26.jpg')