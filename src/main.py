import onnxruntime
from PIL import Image
import numpy as np

# Load the ONNX model
model_path = 'src/course_best_ckpt.onnx'
session = onnxruntime.InferenceSession(model_path)

# Load and preprocess the image
image_path = '../data/custom_dataset/images/train/17.67821.44655.png'
image = Image.open(image_path)

# Preprocess the image (replace with your own preprocessing logic)
# For example, resizing the image to match the expected input size
input_size = (256, 256)
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