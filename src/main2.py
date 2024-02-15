from ultralytics import YOLO

# Build a YOLOv6n model from scratch
model = YOLO('yolov6n.yaml')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='../YOLOv6/data/dataset.yaml', epochs=10, imgsz=512)
