import rasterio
from rasterio.windows import Window
import numpy as np
import os
import json

def crop_and_save_tiles(input_tiff_path, output_dir, tile_size=512):
    image_info = []  # Liste zum Speichern der Bildinformationen
    with rasterio.open(input_tiff_path) as src:
        height, width = src.height, src.width
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                window = Window(j, i, min(tile_size, width - j), min(tile_size, height - i))
                tile = src.read(window=window)
                if np.any(tile != 0):  # Überprüfe, ob die Kachel nicht komplett schwarz ist
                    output_path = os.path.join(output_dir, f"tile_{i}_{j}.tif")
                    transform = src.window_transform(window)
                    profile = src.profile
                    profile.update({
                        'height': tile_size,
                        'width': tile_size,
                        'transform': transform
                    })
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(tile)
                    # Speichere Bildinformationen
                    image_info.append({
                        'file_name': os.path.basename(output_path),
                        'start_x': transform[2],
                        'start_y': transform[5] + tile_size  # Berücksichtige die Umkehrung der Y-Koordinate
                    })
    # Speichere Bildinformationen in einer JSON-Datei
    json_file_path = os.path.join(output_dir, 'image_info.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(image_info, json_file, indent=4)


# Beispielaufruf:
input_tiff_path = r"C:\Users\Gustav Schimmer\Desktop\Studium\Semester_3\deepLearnung\BeetSegmentation\field1_orthomosaic_UTM32N.tif"
output_dir = r"C:\Users\Gustav Schimmer\Desktop\Studium\Semester_2\tilesZugeschnittenUndJsonZurGeoRef"
crop_and_save_tiles(input_tiff_path, output_dir)
