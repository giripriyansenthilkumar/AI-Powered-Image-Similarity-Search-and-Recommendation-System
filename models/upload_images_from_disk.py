#!/usr/bin/env python3
"""
Upload fashion images directly from disk to MongoDB Atlas.
Run this on the system where the fashion-dataset folder is located.

Usage:
  python upload_images_from_disk.py /path/to/fashion-dataset
"""
import os
import sys
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    print("❌ MONGO_URI not set in .env file")
    sys.exit(1)

DB_NAME = 'fashion_db'
COLLECTION_NAME = 'images'
BATCH_SIZE = 50

def upload_images_from_dataset(dataset_root):
    """Scan dataset and upload images directly to MongoDB."""
    
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        print(f"❌ Dataset folder not found: {dataset_root}")
        sys.exit(1)
    
    train_path = dataset_path / 'train'
    if not train_path.exists():
        print(f"❌ train folder not found in {dataset_root}")
        sys.exit(1)
    
    # Connect to MongoDB
    print("🔗 Connecting to MongoDB Atlas...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    db = client[DB_NAME]
    images_collection = db[COLLECTION_NAME]
    
    # Create index
    images_collection.create_index('image_id', unique=True)
    
    # Scan dataset
    print(f"📁 Scanning {train_path}...")
    image_paths = []
    for root, dirs, files in os.walk(train_path):
        for file in sorted(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_paths.append(os.path.join(root, file))
    
    print(f"📸 Found {len(image_paths)} images")
    
    # Check existing
    existing_count = images_collection.count_documents({})
    if existing_count > 0:
        print(f"⚠️  Database already has {existing_count} documents")
        response = input("Delete and re-upload? (y/n): ")
        if response.lower() != 'y':
            print("Aborting")
            client.close()
            return
        images_collection.delete_many({})
        print("🗑️  Cleared database")
    
    # Upload in batches
    print("📤 Uploading images...")
    batch_docs = []
    total_size_gb = 0
    
    for idx, image_path in enumerate(image_paths):
        try:
            # Read image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            total_size_gb += len(image_data) / (1024**3)
            
            # Create document
            doc = {
                'image_id': idx,
                'path': image_path,
                'filename': os.path.basename(image_path),
                'image_data': image_data,
                'size_bytes': len(image_data)
            }
            batch_docs.append(doc)
            
            # Upload batch
            if len(batch_docs) >= BATCH_SIZE:
                images_collection.insert_many(batch_docs)
                uploaded = (idx + 1)
                pct = (idx + 1) / len(image_paths) * 100
                print(f"  [{uploaded:5d}/{len(image_paths)}] {pct:5.1f}% | {total_size_gb:.2f}GB uploaded")
                batch_docs = []
        
        except Exception as e:
            print(f"  ⚠️  Error with {image_path}: {e}")
    
    # Upload remaining
    if batch_docs:
        images_collection.insert_many(batch_docs)
        print(f"  [{len(image_paths):5d}/{len(image_paths)}] 100.0% | {total_size_gb:.2f}GB uploaded")
    
    # Stats
    total_docs = images_collection.count_documents({})
    print(f"\n✅ Upload complete!")
    print(f"   Total images in database: {total_docs}")
    print(f"   Total storage: {total_size_gb:.2f} GB")
    
    # Sample
    print(f"\n📸 Sample entries:")
    for doc in images_collection.find({}).limit(3):
        print(f"   ID {doc['image_id']:4d}: {doc['filename']} ({doc['size_bytes']/1024:.1f}KB)")
    
    client.close()
    print("\n🎉 Done! Images are now in MongoDB Atlas")
    print("   The web app can fetch them from anywhere")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python upload_images_from_disk.py /path/to/fashion-dataset")
        print("\nExample:")
        print("  python upload_images_from_disk.py D:\\fashion-dataset")
        print("  python upload_images_from_disk.py /home/user/fashion-dataset")
        sys.exit(1)
    
    dataset_root = sys.argv[1]
    upload_images_from_dataset(dataset_root)
