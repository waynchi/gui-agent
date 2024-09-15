import os
import json
import pandas as pd
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument("--input_file_path", type=str, required=True)
argparse.add_argument("--output_file_path", type=str, required=True)
argparse.add_argument("--image_folder", type=str, required=True)

args = argparse.parse_args()

# Load the COCO JSON file
with open(args.input_file_path, 'r') as f:
    coco = json.load(f)

# Initialize the list to store each row (bounding box) information
data = []

image_id_to_file_name = {image['id']: image['file_name'] for image in coco['images']}
image_id_to_box_index = {image['id']: 0 for image in coco['images']}
# Loop through the images


# Assuming 'interactable', 'confidence', 'environment', 'box_index' are custom fields or to be computed separately
for annotation in coco['annotations']:
    image_id = annotation['image_id']
    category_id = 0
    
    # COCO bbox format: [x_min, y_min, width, height]
    bbox = annotation['bbox']
    top = bbox[1]
    left = bbox[0]
    bottom = bbox[1] + bbox[3]
    right = bbox[0] + bbox[2]
    center_x = bbox[0] + bbox[2] / 2
    center_y = bbox[1] + bbox[3] / 2

    environment = image_id_to_file_name[image_id]
    image_file_name = os.path.join(args.image_folder, image_id_to_file_name[image_id])
    
    # Example placeholders for additional data
    interactable = True  # Placeholder
    confidence = 1.0  # Placeholder
    x_offset = 0  # Placeholder or to be calculated
    y_offset = 0  # Placeholder or to be calculated
    box_index = image_id_to_box_index[image_id]
    image_id_to_box_index[image_id] += 1
    
    # Find image file name from images section
    
    # Append the data for this bounding box
    data.append([top, left, bottom, right, center_x, center_y, interactable, confidence, category_id, x_offset, y_offset, environment, box_index, image_file_name])

# Create a DataFrame
df = pd.DataFrame(data, columns=['top', 'left', 'bottom', 'right', 'center_x', 'center_y', 'interactable', 'confidence', 'class_id', 'x_offset', 'y_offset', 'environment', 'box_index', 'image'])

# Export the DataFrame to a CSV file
df.to_csv(args.output_file_path, index=False)

print("COCO dataset has been converted to CSV format.")
