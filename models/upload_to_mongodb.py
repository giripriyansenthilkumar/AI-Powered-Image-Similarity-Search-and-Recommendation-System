#!/usr/bin/env python3
"""
Upload fashion dataset images and metadata to MongoDB.
Run this once to migrate your 30GB dataset to the database.

Requirements:
  pip install pymongo python-dotenv
  
MongoDB Setup:
  1. Install MongoDB locally or use MongoDB Atlas (cloud)
  2. Create a database: fashion_db
  3. Get connection string
"""
import os
import json
import sys
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import pickle

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = 'fashion_db'
COLLECTION_NAME = 'images'
BATCH_SIZE = 100

def connect_to_mongodb():
    """Connect to MongoDB."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        print("✅ Connected to MongoDB")
        return client
    except ServerSelectionTimeoutError:
        print(f"❌ Cannot connect to MongoDB at {MONGO_URI}")
        print("   Make sure MongoDB is running or provide MONGO_URI environment variable")
        sys.exit(1)

def upload_images_from_pickle(pickle_path):
    """Upload images from pickle file to MongoDB."""
    client = connect_to_mongodb()
    db = client[DB_NAME]
    images_collection = db[COLLECTION_NAME]
    
    # Create index on image_id for fast lookups
    images_collection.create_index('image_id', unique=True)
    
    print(f"Loading image paths from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        image_paths = pickle.load(f)
    
    print(f"Found {len(image_paths)} images")
    
    # Check if already uploaded
    existing_count = images_collection.count_documents({})
    if existing_count > 0:
        print(f"⚠️  Database already contains {existing_count} images")
        response = input("Overwrite? (y/n): ")
        if response.lower() == 'y':
            images_collection.delete_many({})
            print("Cleared existing images")
        else:
            print("Aborting")
            return
    
    # Upload images in batches
    uploaded = 0
    failed = 0
    batch_docs = []
    
    for idx, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"⚠️  Skipping {idx}: {image_path} not found")
            failed += 1
            continue
        
        try:
            # Read image file
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Create document
            doc = {
                'image_id': idx,
                'path': image_path,
                'filename': os.path.basename(image_path),
                'image_data': image_data,  # Binary image data
                'size_bytes': len(image_data)
            }
            batch_docs.append(doc)
            
            # Upload batch
            if len(batch_docs) >= BATCH_SIZE:
                images_collection.insert_many(batch_docs)
                uploaded += len(batch_docs)
                print(f"Uploaded {uploaded}/{len(image_paths)} images ({(uploaded/len(image_paths)*100):.1f}%)")
                batch_docs = []
        
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            failed += 1
    
    # Upload remaining batch
    if batch_docs:
        images_collection.insert_many(batch_docs)
        uploaded += len(batch_docs)
        print(f"Uploaded {uploaded}/{len(image_paths)} images (100%)")
    
    print(f"\n✅ Upload complete!")
    print(f"   Uploaded: {uploaded}")
    print(f"   Failed: {failed}")
    print(f"   Total in DB: {images_collection.count_documents({})}")
    
    # Save stats
    total_size_gb = images_collection.aggregate([
        {'$group': {'_id': None, 'total': {'$sum': '$size_bytes'}}}
    ])
    for stat in total_size_gb:
        size_gb = stat['total'] / (1024**3)
        print(f"   Total storage: {size_gb:.2f} GB")
    
    client.close()

def upload_images_from_directory(dataset_root):
    """Upload images directly from fashion-dataset folder."""
    client = connect_to_mongodb()
    db = client[DB_NAME]
    images_collection = db[COLLECTION_NAME]
    
    # Create index
    images_collection.create_index('image_id', unique=True)
    
    # Scan directory
    print(f"Scanning {dataset_root}...")
    image_paths = []
    for root, dirs, files in os.walk(os.path.join(dataset_root, 'train')):
        for file in sorted(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images")
    
    # Check if already uploaded
    existing_count = images_collection.count_documents({})
    if existing_count > 0:
        print(f"⚠️  Database already contains {existing_count} images")
        response = input("Overwrite? (y/n): ")
        if response.lower() == 'y':
            images_collection.delete_many({})
        else:
            print("Aborting")
            return
    
    # Upload in batches
    uploaded = 0
    failed = 0
    batch_docs = []
    
    for idx, image_path in enumerate(image_paths):
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            doc = {
                'image_id': idx,
                'path': image_path,
                'filename': os.path.basename(image_path),
                'image_data': image_data,
                'size_bytes': len(image_data)
            }
            batch_docs.append(doc)
            
            if len(batch_docs) >= BATCH_SIZE:
                images_collection.insert_many(batch_docs)
                uploaded += len(batch_docs)
                pct = (idx + 1) / len(image_paths) * 100
                print(f"Uploaded {uploaded}/{len(image_paths)} images ({pct:.1f}%)")
                batch_docs = []
        
        except Exception as e:
            print(f"❌ Error with {image_path}: {e}")
            failed += 1
    
    if batch_docs:
        images_collection.insert_many(batch_docs)
        uploaded += len(batch_docs)
        print(f"Uploaded {uploaded}/{len(image_paths)} images (100%)")
    
    print(f"\n✅ Upload complete!")
    print(f"   Total: {images_collection.count_documents({})}")
    
    client.close()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--from-directory':
        dataset_root = sys.argv[2] if len(sys.argv) > 2 else 'fashion-dataset'
        upload_images_from_directory(dataset_root)
    else:
        pickle_path = sys.argv[1] if len(sys.argv) > 1 else 'models/fashion_train_image_paths.pkl'
        upload_images_from_pickle(pickle_path)
