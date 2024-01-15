from PIL import Image
import os

# Pfad zum Originalordner und Zielordner
original_folder = r'C:\Users\Gustav Schimmer\Desktop\Studium\Semester_3\deepLearnung\BeetSegmentation\beet-segmentation\data\20230514\field_1_allPictures_(1500x2000)_cut_to_500p'
target_folder = r'C:\Users\Gustav Schimmer\Desktop\Studium\Semester_3\deepLearnung\BeetSegmentation\SchwarzeBilder'

# Größe des Originalbilds und der gewünschten Bildgröße
original_size = (500, 500)
new_size = (512, 512)

# Breite der zusätzlichen schwarzen Ränder
border_width = 12

# Erstelle den Zielordner, wenn er noch nicht existiert
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Durchlaufe alle Bilder im Originalordner
for filename in os.listdir(original_folder):
    if filename.endswith('.jpg'):
        # Lade das Originalbild
        original_image_path = os.path.join(original_folder, filename)
        original_image = Image.open(original_image_path)

        # Erstelle ein neues Bild mit der gewünschten Größe und fülle es mit Schwarz
        new_image = Image.new('RGB', new_size, 'black')

        # Füge das Originalbild in das neue Bild ein
        new_image.paste(original_image, (0, 0))

        # Speichere das neue Bild im Zielordner mit dem gleichen Dateinamen
        target_path = os.path.join(target_folder, filename)
        new_image.save(target_path)

# Optional: Schließe das Originalbild
original_image.close()
