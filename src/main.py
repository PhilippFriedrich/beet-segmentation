from ultralytics import YOLO
import os
from osgeo import gdal
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, CAP_STYLE, JOIN_STYLE
from shapely.ops import unary_union

# Load a model
model = YOLO("runs/detect/train/weights/best.pt")  # pretrained YOLOv6n model

# Define the input image folder
input_folder = "data/tilesZugeschnittenUndJsonZurGeoRef"

# Get a list of image files in the input folder
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.tiff', '.tif'))]

# Define function for coordinate conversion
def pixel_to_geo(pixel_x, pixel_y, geotransform):
    geo_x = geotransform[0] + pixel_x * geotransform[1] + pixel_y * geotransform[2]
    geo_y = geotransform[3] + pixel_x * geotransform[4] + pixel_y * geotransform[5]
    return geo_x, geo_y

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

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        if boxes.shape[0] != 0:
            pixel_xmin = boxes.xyxy[0][0].item()
            pixel_ymin = boxes.xyxy[0][1].item()
            pixel_xmax = boxes.xyxy[0][2].item()
            pixel_ymax = boxes.xyxy[0][3].item()

            # Convert pixel coordinates to geographic coordinates
            geo_xmin, geo_ymin = pixel_to_geo(pixel_xmin, pixel_ymin, geotransform)
            geo_xmax, geo_ymax = pixel_to_geo(pixel_xmax, pixel_ymax, geotransform)
            
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

# Define the output file path
output_file_bbox = "results/sb_bbox.geojson"
output_file_centroid = "results/sb_point.geojson"

# Export the GeoDataFrame to a GeoJSON file
bbox_gdf.to_file(output_file_bbox, driver='GeoJSON')
centroid_gdf.to_file(output_file_centroid, driver='GeoJSON')

print("Sugar beet bbox exported to:", output_file_bbox)
print("Sugar beet points exported to:", output_file_centroid)
