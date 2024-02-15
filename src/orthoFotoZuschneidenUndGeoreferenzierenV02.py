import rasterio
from rasterio.windows import Window
import numpy as np
import os

def crop_and_save_tiles(input_tiff_path, output_dir, tile_size=512):
    with rasterio.open(input_tiff_path) as src:
        height, width = src.height, src.width
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                window = Window(j, i, min(tile_size, width - j), min(tile_size, height - i))
                tile = src.read(window=window)
                if tile.shape[1] == tile_size and tile.shape[2] == tile_size:
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

# Example usage:
input_tiff_path = r"C:\Users\Gustav Schimmer\Desktop\Studium\Semester_3\deepLearnung\BeetSegmentation\field1_orthomosaic_UTM32N.tif"  # Passe den Dateinamen entsprechend an
output_dir = r"C:\Users\Gustav Schimmer\Desktop\Studium\Semester_2\tilesVonV02"
crop_and_save_tiles(input_tiff_path, output_dir)
