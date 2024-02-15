import cv2
import json
import numpy as np

def draw_bounding_boxes(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    image_ids = set()
    for annotation in data['annotations']:
        image_ids.add(annotation['image_id'])

    for image_id in image_ids:
        image_annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]
        img = np.zeros((512, 512, 3), dtype=np.uint8)

        for annotation in image_annotations:
            bbox = annotation['bbox']
            x, y, w, h = map(int, bbox)
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Bounding Boxes', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

# Beispielaufruf:
json_file = r"C:\Users\Gustav Schimmer\Downloads\instances_val.json"
draw_bounding_boxes(json_file)
