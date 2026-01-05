#!/usr/bin/env python3
"""
Upload image metadata to MongoDB Atlas.
This creates database entries with image_id and path.
When actual files are available, they can be added to the database.
"""
import os
import sys
import pickle
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = 'fashion_db'
COLLECTION_NAME = 'images'

def upload_image_metadata(pickle_path):
    """Upload image metadata from pickle file to MongoDB."""
    
    # Connect
    print("Connecting to MongoDB Atlas...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("✅ Connected!")
    
    db = client[DB_NAME]
    images_collection = db[COLLECTION_NAME]
    
    # Create index
    images_collection.create_index('image_id', unique=True)
    print("✅ Index created")
    
    # Load paths
    print(f"Loading image paths from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        image_paths = pickle.load(f)
    
    print(f"Found {len(image_paths)} image paths")
    
    # Check if already exists
    existing_count = images_collection.count_documents({})
    if existing_count > 0:
        print(f"⚠️  Database already has {existing_count} documents")
        response = input("Delete and re-upload? (y/n): ")
        if response.lower() == 'y':
            images_collection.delete_many({})
            print("Cleared database")
        else:
            print("Aborting")
            return
    
    # Upload metadata
    print("Uploading metadata...")
    docs = []
    for idx, image_path in enumerate(image_paths):
        doc = {
            'image_id': idx,
            'path': image_path,
            'filename': os.path.basename(image_path),
            # 'image_data' will be added later when files are available
        }
        docs.append(doc)
    
    images_collection.insert_many(docs)
    print(f"✅ Uploaded {len(docs)} metadata entries to MongoDB")
    
    # Show stats
    total = images_collection.count_documents({})
    print(f"\n📊 Database Stats:")
    print(f"   Total entries: {total}")
    
    # Sample
    print(f"\n📸 Sample entries:")
    for doc in images_collection.find({}).limit(3):
        print(f"   ID {doc['image_id']}: {doc['path']}")
    
    client.close()
    print("\n✅ Done! Metadata is ready.")
    print("   When you have the image files available, run: python models/add_images_to_mongodb.py")

if __name__ == '__main__':
    pickle_path = sys.argv[1] if len(sys.argv) > 1 else 'models/fashion_train_image_paths.pkl'
    upload_image_metadata(pickle_path)
