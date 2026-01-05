#!/usr/bin/env python3
"""
Generate gallery.json mapping FAISS index IDs to image paths.
Must scan the dataset in the SAME ORDER as the FAISS index was built.
"""
import os
import json

train_folder = 'fashion-dataset/train'
output_path = 'models/gallery.json'

if not os.path.exists(train_folder):
    print(f"ERROR: {train_folder} not found!")
    exit(1)

# Gather all image paths in the SAME ORDER as FAISS indexing
# (sorted class folders, then sorted files within each)
train_image_paths = []
for class_folder in sorted(os.listdir(train_folder)):
    class_path = os.path.join(train_folder, class_folder)
    if os.path.isdir(class_path):
        for file in sorted(os.listdir(class_path)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                train_image_paths.append(os.path.join(class_path, file))

print(f"Found {len(train_image_paths)} images")

# Create mapping: index_id -> path
gallery = {i: path for i, path in enumerate(train_image_paths)}

# Save as JSON
with open(output_path, 'w') as f:
    json.dump(gallery, f, indent=2)

print(f"Saved gallery mapping to {output_path}")
print(f"Sample entries:")
for i in range(min(3, len(gallery))):
    print(f"  {i}: {gallery[i]}")
