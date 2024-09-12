
import os
import csv

# Path to the root directory that contains subfolders with images
root_dir = "../data/raw"

# Output file paths
image_csv_path = "image_paths_and_classes.csv"
class_mapping_csv_path = "class_mapping.csv"

# Step 1: Get all class names (subfolder names)
class_names = sorted(os.listdir(root_dir))  # Sort to ensure consistent ordering
class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}

# Step 2: Create the image_paths_and_classes.csv
with open(image_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["image_path", "class"])  # Header row
    
    for class_name, class_idx in class_to_index.items():
        class_folder = os.path.join(root_dir, class_name)
        if os.path.isdir(class_folder):
            # Iterate through all files in the class folder
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                # Make sure it's a file (in case there are non-image files or directories)
                if os.path.isfile(img_path):
                    csvwriter.writerow([img_path, class_idx])

# Step 3: Create the class_mapping.csv (class index -> class name)
with open(class_mapping_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["class_index", "class_name"])  # Header row
    for class_name, class_idx in class_to_index.items():
        csvwriter.writerow([class_idx, class_name])

print(f"CSV files created: {image_csv_path}, {class_mapping_csv_path}")
