
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Paths for the processed data
CSV_PATH = "./image_paths_and_classes.csv"
PROCESSED_DATA_DIR = "../data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")

# Function to create directories if they don't exist
def create_dirs():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

# Function to copy images to the respective directory and update CSV
def copy_images_and_save_csv(df, split_dir, split_csv_path):
    image_paths = []
    labels = []
    
    # Create directory for the split (train/test)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    
    # Copy images to the appropriate folder and update paths in CSV
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying images to {split_dir}"):
        image_path = row['image_path']
        label = row['class']
        
        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Image {image_path} not found, skipping...")
            continue
        
        # Copy the image to the split directory (train/test)
        image_filename = os.path.basename(image_path)
        destination_path = os.path.join(split_dir, image_filename)
        shutil.copy(image_path, destination_path)
        
        # Store the new path and the label
        image_paths.append(destination_path)
        labels.append(label)
    
    # Create a new CSV with the updated image paths and labels
    new_df = pd.DataFrame({
        'image_path': image_paths,
        'class': labels
    })
    new_df.to_csv(split_csv_path, index=False)
    print(f"CSV saved to {split_csv_path}")

def main():
    # Load the dataset from the CSV
    df = pd.read_csv(CSV_PATH)
    
    # Check that the CSV has the correct columns
    if 'image_path' not in df.columns or 'class' not in df.columns:
        raise ValueError("CSV must contain 'image_path' and 'class' columns")
    
    # Split the dataset into train and test sets (80% train, 20% test, random state 32)
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=32)
    
    # Create directories for the processed data
    create_dirs()
    
    # Copy train and test images and save the corresponding CSVs
    copy_images_and_save_csv(train_df, TRAIN_DIR, os.path.join(PROCESSED_DATA_DIR, "train.csv"))
    copy_images_and_save_csv(test_df, TEST_DIR, os.path.join(PROCESSED_DATA_DIR, "test.csv"))
    
    print(f"Data split completed. Train and test data saved in {PROCESSED_DATA_DIR}.")

if __name__ == "__main__":
    main()
