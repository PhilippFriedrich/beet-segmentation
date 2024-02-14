import onnxruntime
from PIL import Image
import numpy as np


# Load the ONNX model

model_path = r"C:\Users\Gustav Schimmer\Desktop\Studium\Semester_3\deepLearnung\BeetSegmentation\beet-segmentation\src\best_ckpt_(zuVielBoxen).onnx"
session = onnxruntime.InferenceSession(model_path)

# Load and preprocess the image
image_path = r"C:\Users\Gustav Schimmer\Desktop\Studium\Semester_3\deepLearnung\BeetSegmentation\beet-segmentation\data\20230514\field_1_allPictures_(1500x2000)_cut_to_512p\IMG_20230514_155219_3.jpg"
#image_path = r"C:\Users\Gustav Schimmer\Desktop\Studium\Semester_3\deepLearnung\Tag3\custom_dataset-20231012T095854Z-001\custom_dataset\images\train\17.67681.42715.png"

image = Image.open(image_path)


# Preprocess the image (replace with your own preprocessing logic)
# For example, resizing the image to match the expected input size
#input_size = (256, 256)
input_size = (512, 512)

image = image.resize(input_size)
input_data = np.array(image, dtype=np.float32)
input_data = np.transpose(input_data, (2, 0, 1))  # Channels-first format if needed

# Normalize the input data (replace with your own normalization logic)
input_data = (input_data - 128.0) / 128.0

# Reshape the input data if needed
input_data = np.expand_dims(input_data, axis=0)

# Run the model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: input_data})

# Print the result
print("Model output:", result)

input_info = session.get_inputs()
output_info = session.get_outputs()

print("Input information:")
for input_tensor in input_info:
    print(input_tensor.name, input_tensor.shape, input_tensor.type)

print("\nOutput information:")
for output_tensor in output_info:
    print(output_tensor.name, output_tensor.shape, output_tensor.type)

result = session.run([output_name], {input_name: input_data})
print("Model output:", result)


print("-----------------")
for idx, output_array in enumerate(result):
    print(f"Output Array {idx + 1} shape:", output_array.shape)
    print("Output Array values:")
    print(output_array)

import cv2

# Laden des Eingabebildes
input_image = cv2.imread(image_path)

# Überprüfen, ob Bounding Boxes vorhanden sind
if result and len(result[0][0]) > 0:
    # Iteriere über die Bounding Boxes und zeichne sie auf das Bild
    for box in result[0][0]:
        x, y, w, h, _, _ = box.astype(int)
        cv2.rectangle(input_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Zeige das Bild mit den gezeichneten Bounding Boxes
    cv2.imshow("Output Image", input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Keine Bounding Boxes gefunden.")

onnx.checker.check_model(onnx_model)

    