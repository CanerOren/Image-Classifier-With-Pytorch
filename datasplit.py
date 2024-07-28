import os
import random
import shutil

# Configure your paths and split ratio here
dataset_path = r"dataset"  # Replace with your dataset path
train_path = "model_dataset/train"  # Replace with your desired train directory path
test_path = "model_dataset/test"  # Replace with your desired test directory path
split_ratio = 0.8  # 80% for train, 20% for test

# Create train and test directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Iterate over each class folder
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)

    # Skip if not a directory
    if not os.path.isdir(class_path):
        continue

    # List all files in the class folder
    files = [
        f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))
    ]

    # Shuffle files
    random.shuffle(files)

    # Split files into train and test
    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    test_files = files[split_index:]

    # Create corresponding class folders in train and test directories
    os.makedirs(os.path.join(train_path, class_folder), exist_ok=True)
    os.makedirs(os.path.join(test_path, class_folder), exist_ok=True)

    # Move files to train and test directories
    for f in train_files:
        shutil.copy2(
            os.path.join(class_path, f), os.path.join(train_path, class_folder)
        )

    for f in test_files:
        shutil.copy2(os.path.join(class_path, f), os.path.join(test_path, class_folder))

print("Dataset splitting completed.")
