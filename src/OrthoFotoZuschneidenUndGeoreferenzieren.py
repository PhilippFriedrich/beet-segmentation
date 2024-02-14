import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

def cut_image_into_tiles(input_file, output_dir, tile_size=512):
    # Öffne das GeoTIFF-Bild
    with rasterio.open(input_file) as src:
        # Lese die Georeferenzierungsinformationen
        transform = src.transform
        width = src.width
        height = src.height

        # Berechne die Anzahl der Zeilen und Spalten für die Tiles
        num_rows = int(np.ceil(height / tile_size))
        num_cols = int(np.ceil(width / tile_size))

        # Iteriere über die Tiles und schneide jedes Teilbild aus
        for row in range(num_rows):
            for col in range(num_cols):
                # Berechne die Grenzen des aktuellen Tiles
                y_start = row * tile_size
                y_end = min((row + 1) * tile_size, height)
                x_start = col * tile_size
                x_end = min((col + 1) * tile_size, width)

                # Lese den entsprechenden Ausschnitt des Bildes
                tile_data = src.read(window=((y_start, y_end), (x_start, x_end)))

                # Passe die Georeferenzierungsinformationen für das Teilbild an
                tile_transform = from_origin(transform[2] + x_start * transform[0],
                                             transform[5] - y_end * transform[4],
                                             transform[0],
                                             transform[4])

                # Speichere das Teilbild als GeoTIFF-Datei
                output_file = os.path.join(output_dir, f"tile_{row}_{col}.tif")
                with rasterio.open(output_file, 'w', driver='GTiff',
                                   height=tile_data.shape[1],
                                   width=tile_data.shape[2],
                                   count=tile_data.shape[0],
                                   dtype=tile_data.dtype,
                                   crs=src.crs,
                                   transform=tile_transform) as dst:
                    dst.write(tile_data)

# Beispielaufruf
if __name__ == "__main__":
    input_file = r"C:\Users\Gustav Schimmer\Desktop\Studium\Semester_3\deepLearnung\BeetSegmentation\field1_orthomosaic_UTM32N.tif"  # Passe den Dateinamen entsprechend an

    output_dir = r"C:\Users\Gustav Schimmer\Desktop\Studium\Semester_3\deepLearnung\BeetSegmentation\test"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cut_image_into_tiles(input_file, output_dir)

