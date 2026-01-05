#!/usr/bin/env python3
"""
Convert fashion_train_image_paths.pkl to gallery.json
"""
import pickle
import json
import os

pkl_path = 'models/fashion_train_image_paths.pkl'
output_path = 'models/gallery.json'

if not os.path.exists(pkl_path):
    print(f"ERROR: {pkl_path} not found!")
    exit(1)

# Load the pickle file
with open(pkl_path, 'rb') as f:
    image_paths = pickle.load(f)

print(f"Loaded {len(image_paths)} image paths from pickle")

# Create mapping: index_id -> path
gallery = {str(i): path for i, path in enumerate(image_paths)}

# Save as JSON
with open(output_path, 'w') as f:
    json.dump(gallery, f, indent=2)

print(f"Saved gallery mapping to {output_path}")
print(f"Sample entries:")
for i in range(min(5, len(gallery))):
    print(f"  {i}: {gallery[str(i)]}")

print(f"\nTotal images in gallery: {len(gallery)}")
