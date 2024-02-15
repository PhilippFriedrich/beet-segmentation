from ultralytics import YOLO
import os

# Load a model
model = YOLO("src/runs/detect/train2/weights/best.pt")  # pretrained YOLOv8n model

# Define the input image folder
input_folder = "data/zugeschnitteneUndGeoreferenzierteBilder"

# Get a list of image files in the input folder
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.tiff', '.tif'))]

i = 0

# Process each image in the input folder
for image_file in image_files:
    # Run inference on the current image
    results = model(image_file)

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        print(boxes)
        print(image_file)
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs

        # Display or save the result
        result.show()  # display to screen
        result.save(filename=f'result_{i}.tif')  # save to disk with a unique filename
        i = i + 1