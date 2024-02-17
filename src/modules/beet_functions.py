# -*- coding: utf-8 -*-
# beet_functions.py

"""Functions for beet segmentation pre/postprocessing and counting"""

import rasterio
from rasterio.windows import Window
import numpy as np
import os
import ultralytics


# Function to create new output folder
def create_output_folder(folder_name: str):
    """
    Creates a new output folder if it doesn't exist
    :param folder_name: Name of the folder that shall be created
    """
    
    # Create a new folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")


# Function to crop GeoTIFF into single tiles and store them to defined directory
def crop_and_save_tiles(input_path_tif: str, output_dir_tiles: str, tile_size: int = 512):
    """
    Crops GeoTIFF into single tiles of defined size and stores them into given directory
    :param input_tif_path: Path to GeoTIFF file
    :param output_dir: Directory to store the single tiles to
    :param tile_size: Tile size (default: 512)
    """

    with rasterio.open(input_path_tif) as src:
        height, width = src.height, src.width

        # Crop GeoTIFF to single tiles 
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                window = Window(j, i, min(tile_size, width - j), min(tile_size, height - i))
                tile = src.read(window=window)

                # Store tiles to output directory
                if np.any(tile != 0):  # Check if image doesn't contain only 0 pixel values
                    output_path = os.path.join(output_dir_tiles, f"tile_{i}_{j}.tif")
                    transform = src.window_transform(window)
                    profile = src.profile
                    profile.update({
                        'height': tile_size,
                        'width': tile_size,
                        'transform': transform
                    })
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(tile)


# Function to convert pixel to geographic coordinates
def pixel_to_geo(pixel_x, pixel_y, geotransform):
    """
    Converts pixel to geographic coordinates using GDAL geotransform information
    :param pixel_x: Pixel x-coordinate
    :param pixel_y: Pixel y-coordinate
    :param geotransform: Geotransformation information of the image
    :return: Geographic x- and y-coordinates
    """

    geo_x = geotransform[0] + pixel_x * geotransform[1] + pixel_y * geotransform[2]
    geo_y = geotransform[3] + pixel_x * geotransform[4] + pixel_y * geotransform[5]
    return geo_x, geo_y


# Function to get bounding box x-/y-min and x-/y-max coordinates from torch object
def get_xyxy_coords(result: ultralytics.engine.results.Results):
    """
    Gets bounding box coordinates in xyxy format from ultralytics result
    :param result: result of ultralytics YOLOv6 model implementation
    :return: x-min, y-min, x-max, y-max bbox pixel coordinates
    """

    pixel_xmin = result.boxes.xyxy[0][0].item()
    pixel_ymin = result.boxes.xyxy[0][1].item()
    pixel_xmax = result.boxes.xyxy[0][2].item()
    pixel_ymax = result.boxes.xyxy[0][3].item()

    return pixel_xmin, pixel_ymin, pixel_xmax, pixel_ymax
