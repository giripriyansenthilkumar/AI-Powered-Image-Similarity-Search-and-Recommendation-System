import os
import random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class TripletDataset(Dataset):
    def __init__(self, class_to_images, transform=None):
        self.class_to_images = class_to_images
        self.classes = list(class_to_images.keys())
        self.transform = transform

    def __len__(self):
        return sum(len(imgs) for imgs in self.class_to_images.values())

    def __getitem__(self, idx):
        # Sample anchor and positive from the same class
        anchor_class = random.choice(self.classes)
        anchor, positive = random.sample(self.class_to_images[anchor_class], 2)

        # Sample negative from a different class
        negative_class = random.choice([c for c in self.classes if c != anchor_class])
        negative = random.choice(self.class_to_images[negative_class])

        # Load and preprocess images
        anchor_img = Image.open(anchor).convert('RGB')
        positive_img = Image.open(positive).convert('RGB')
        negative_img = Image.open(negative).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

def get_triplet_dataloader(class_to_images, batch_size=32, transform=None):
    dataset = TripletDataset(class_to_images=class_to_images, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader
