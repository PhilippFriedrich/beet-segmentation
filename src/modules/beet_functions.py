# -*- coding: utf-8 -*-
# beet_functions.py

"""Functions for beet segmentation pre/postprocessing and counting"""

import rasterio
from rasterio.windows import Window
import numpy as np
import os
import ultralytics
import sys
import fiona


# Function to check input arguments
def check_input_arguments():
    """
    Checks if provided command-line arguments are valid
    """

    # Ensure that the correct number of command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py arg1 arg2")
        print("Please provide two arguments.")
        sys.exit(1)  # Exit with a non-zero status code to indicate an error

    # Extract command-line arguments
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    def is_integer(value):
        try:
            int(value)
            return True
        except ValueError:
            return False

    if not isinstance(arg1, str):
        print("Error. arg1 must be of type string.")
        sys.exit(1)

    elif not isinstance(arg2, str):
        print("Error. arg2 must be of type string.")
        sys.exit(1)

    else:
        print("System Arguments correct. Start processing..")

    return arg1, arg2


# Function to check if input is file
def check_geojson(input_file):
    """
    Checks if input file is valid GeoJSON
    :param input_file: Path to GeoJSON file
    :return: Boolean 
    """

    def is_valid_geojson(file_path):
        try:
            with fiona.open(file_path) as f:
                # Check if the file format is GeoJSON
                if f.driver == "GeoJSON":
                    return True
                else:
                    print("File is not in GeoJSON format.")
                    return False
        except Exception as e:
            print("Error reading file:", e)
            return False

    # Check if the file exists
    if not os.path.isfile(input_file):
        print(f"File '{input_file}' does not exist.")
        return False

    # Check if the file is a valid GeoJSON file
    if not is_valid_geojson(input_file):
        return False

    return True


# Function to check if input is valid GeoTIFF
def check_geotiff(input_file):
    """
    Checks if input file is valid GeoTIFF
    :param input_file: Path to GeoTIFF file
    :return: Boolean 
    """

    def is_valid_geotiff(file_path):
        try:
            # Attempt to open the file using rasterio
            with rasterio.open(file_path) as src:
                # Check if it's a GeoTIFF file
                if src.driver == 'GTiff':
                    return True
                else:
                    return False
        except Exception as e:
            print("Error opening file:", e)
            return False

    # Check if the file exists
    if not os.path.isfile(input_file):
        print(f"File '{input_file}' does not exist.")
        return False

    # Check if the file is a valid GeoJSON file
    if not is_valid_geotiff(input_file):
        return False

    return True


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
