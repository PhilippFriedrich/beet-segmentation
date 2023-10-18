import os
import cv2

# Pfad zum Ordner mit den Originalbildern
input_folder = r'C:\Users\Gustav Schimmer\Desktop\Studium\Semester_3\deepLearnung\BeetSegmentation\beet-segmentation\data\20230514\10testbilder_Field_1'

# Pfad zum Ausgabeordner, in dem die zugeschnittenen Bilder gespeichert werden sollen
output_folder = r'C:\Users\Gustav Schimmer\Desktop\Studium\Semester_3\deepLearnung\BeetSegmentation\beet-segmentation\data\20230514\10testbilder_Field_1_zugeschnitten'

# Größe, auf die die Bilder verkleinert werden sollen (2000x1500)
target_width, target_height = 1500, 2000

# Größe der zugeschnittenen Quadrate
square_size = 256

# Funktion zum Zuschneiden der Bilder
def crop_and_resize_image(input_path, output_folder):
    image = cv2.imread(input_path)
    if image is not None:
        # Verkleinere das Bild auf die Zielgröße
        image = cv2.resize(image, (target_width, target_height))
        # Zuschneiden in 256x256 Quadraten
        for y in range(0, target_height - square_size + 1, square_size):
            for x in range(0, target_width - square_size + 1, square_size):
                square = image[y:y + square_size, x:x + square_size]
                # Speichere das Quadrat mit einem fortlaufenden Index
                output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_path))[0]}_{y // square_size * (target_width // square_size) + x // square_size}.jpg")
                cv2.imwrite(output_path, square)

# Erstelle den Ausgabeordner, wenn er nicht existiert
os.makedirs(output_folder, exist_ok=True)

# Durchlaufe alle Bilder im Eingabeordner
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        input_path = os.path.join(input_folder, filename)
        crop_and_resize_image(input_path, output_folder)

print("Die Bilder wurden zugeschnitten und verkleinert.")
