import onnxruntime
from PIL import Image
import numpy as np

# Load the ONNX model
model_path = r"C:\Users\phili\Documents\Studium\Master_Geographie\5_Semester\S_Deep_Learning\beet-segmentation\src\best_ckptNEU.onnx"
session = onnxruntime.InferenceSession(model_path)

# Load and preprocess the image
image_path = r"C:\Users\phili\Documents\Studium\Master_Geographie\5_Semester\S_Deep_Learning\data\20230514\field_1_allPictures_(1500x2000)_cut_to_512p\IMG_20230514_155225_4.jpg"
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
