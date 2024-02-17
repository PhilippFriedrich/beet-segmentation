# -*- coding: utf-8 -*-
# main.py

"""Main Program Execution File for Sugar Beet Segmentation and Counting"""


from ultralytics import YOLO
import os
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import Polygon, CAP_STYLE, JOIN_STYLE
from shapely.ops import unary_union
import  modules.beet_functions as bf
import shutil


def main():
    """
    Main function to process command line arguments and execute the specified analysis.
    """

    # Define GeoTIFF paths here
    input_path_tif = r"C:\Users\phili\Documents\Studium\Master_Geographie\5_Semester\S_Deep_Learning\project\field1_orthomosaic_UTM32N.tif"
    input_path_area = "../data/study_area/sb_field_1_area.geojson"

    # Define output paths
    output_dir_tiles = "../results/orthophoto_tiles"
    output_file_bbox = "../results/sb_bbox.geojson"
    output_file_centroid = "../results/sb_point.geojson"

    # Crop orthophoto
    bf.create_output_folder(output_dir_tiles)
    bf.crop_and_save_tiles(input_path_tif, output_dir_tiles)

    # Load a model
    model = YOLO("../YOLOv6/runs/detect/train/weights/best.pt")  # pretrained YOLOv6n model

    # Get a list of image files in the input folder
    image_files = [os.path.join(output_dir_tiles, f) for f in os.listdir(output_dir_tiles) if f.endswith(('.jpg', '.png', '.tiff', '.tif'))]

    polygon_list = []

    # Process each image in the input folder
    for image_file in image_files:
        # Run inference on the current image
        results = model(image_file)

        dataset = gdal.Open(image_file)
        if dataset is None:
            raise FileNotFoundError("Image file not found")

        # Get geotransform
        geotransform = dataset.GetGeoTransform()

        # Close the dataset
        dataset = None

        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            if boxes.shape[0] != 0:
                pixel_xmin, pixel_ymin, pixel_xmax, pixel_ymax = bf.get_xyxy_coords(result)

                # Convert pixel coordinates to geographic coordinates
                geo_xmin, geo_ymin = bf.pixel_to_geo(pixel_xmin, pixel_ymin, geotransform)
                geo_xmax, geo_ymax = bf.pixel_to_geo(pixel_xmax, pixel_ymax, geotransform)
                
                # Create a bounding box as a Shapely Polygon object
                bounding_box = Polygon([(geo_xmin, geo_ymin), (geo_xmax, geo_ymin), (geo_xmax, geo_ymax), (geo_xmin, geo_ymax)])

                polygon_list.append(bounding_box)

            else:
                continue

    # Create a GeoDataFrame from the list of polygons
    gdf = gpd.GeoDataFrame(geometry=polygon_list)

    # Buffer single polygons
    buffered_poly = gdf.buffer(0.025, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered_poly)

    # Perform unary union to combine overlapping polygons into single geometries and create new dataframe
    union_poly = unary_union(buffered_gdf.geometry)
    union_gdf = gpd.GeoDataFrame(geometry=[union_poly])

    # Create a new GeoDataFrame with the union result and dissolve geometries
    dissolved_gdf = union_gdf.explode()

    # Reset buffer
    bbox_poly = dissolved_gdf.buffer(-0.025, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
    bbox_gdf = gpd.GeoDataFrame(geometry=bbox_poly)

    # Get centoids of each polygon as points
    centroid_point = bbox_gdf.geometry.centroid
    centroid_gdf = gpd.GeoDataFrame(geometry=centroid_point)

    # Export the GeoDataFrame to a GeoJSON file
    bbox_gdf.to_file(output_file_bbox, driver='GeoJSON')
    centroid_gdf.to_file(output_file_centroid, driver='GeoJSON')

    print("Sugar beet bbox exported to:", output_file_bbox)
    print("Sugar beet points exported to:", output_file_centroid)

    # Read area polygon and count beets within 
    area_gdf = gpd.read_file(input_path_area)
    beets_in_area = gpd.sjoin(centroid_gdf, area_gdf, how='inner', op='within')

    print("-----------------------------------")
    print("-----------------------------------")
    print(f"Number of beets detected in area of interest: {len(beets_in_area)}")
    print(f"Plant density in area of interest: {round(len(beets_in_area)/area_gdf.geometry.area.iloc[0], 2)} plants/m²")
    print("-----------------------------------")
    print("-----------------------------------")

    # Remove folder containing orthophoto tiles
    shutil.rmtree(output_dir_tiles)
    print(f"Folder '{output_dir_tiles}' deleted successfully.")


if __name__ == "__main__":
    main()
