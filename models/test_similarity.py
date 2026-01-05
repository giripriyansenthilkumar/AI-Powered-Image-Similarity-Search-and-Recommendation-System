import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

print("🚀 NEW IMAGE vs FASHION DATASET (TOP 5 - EPOCH 2)!")

# LOAD EPOCH 2 MODEL
class TripletModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50
        self.encoder = resnet50(weights=None)
        self.encoder.fc = torch.nn.Identity()
        self.embed = torch.nn.Linear(2048, 128)
    def forward(self, x):
        features = self.encoder(x)
        emb = self.embed(features)
        return F.normalize(emb, p=2, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TripletModel().to(device)
checkpoint2 = torch.load("models/triplet_model_epoch_2.pth", map_location=device)
model.load_state_dict(checkpoint2['model_state_dict'], strict=False)
model.eval()
print("✅ EPOCH 2 LOADED!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(path):
    return transform(Image.open(path).convert('RGB'))

# YOUR NEW EXTERNAL IMAGE (CHANGE THIS PATH!)
query_path = r"C:\Users\Akarshana\Downloads\BLACK DRESS.webp"  
print(f"🎯 NEW IMAGE: {os.path.basename(query_path)}")

# 1000 FASHION GALLERY
gallery_paths = []
count = 0
for root, dirs, files in os.walk('fashion-dataset'):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            gallery_paths.append(os.path.join(root, file))
            count += 1
            if count >= 1000: break
    if count >= 1000: break

print(f"✅ Gallery: {len(gallery_paths)} images")

# GALLERY EMBEDDINGS
print("Extracting gallery...")
batch_size = 64
gallery_embs = []
for i in range(0, len(gallery_paths), batch_size):
    batch_paths = gallery_paths[i:i+batch_size]
    batch_imgs = torch.stack([load_image(p) for p in batch_paths]).to(device)
    with torch.no_grad():
        batch_embs = model(batch_imgs).cpu().numpy()
    gallery_embs.append(batch_embs)

gallery_embs = np.vstack(gallery_embs)
print(f"✅ Gallery ready: {gallery_embs.shape}")

# NEW IMAGE EMBEDDING
query_img = load_image(query_path).unsqueeze(0).to(device)
with torch.no_grad():
    query_emb = model(query_img).cpu().numpy()
print(f"✅ New image embedding: {query_emb.shape}")

# TOP 5 SIMILARITY
similarities = cosine_similarity(query_emb, gallery_embs)[0]
top_k = np.argsort(similarities)[::-1][:5]

print("\n🚀 TOP 5 FASHION MATCHES:")
print("="*50)
for i, idx in enumerate(top_k):
    score = similarities[idx]
    img_name = os.path.basename(gallery_paths[idx])
    print(f"{i+1}. {img_name:<35} | Cosine: {score:.4f}")

# PERFECT 1x6 PLOT (TOP 5)
fig, axs = plt.subplots(1, 6, figsize=(18, 3))
fig.suptitle(f'{os.path.basename(query_path)} → TOP 5 FASHION MATCHES (EPOCH 2)', fontsize=16)

# YOUR NEW IMAGE
axs[0].imshow(Image.open(query_path))
axs[0].set_title('YOUR\nNEW IMAGE', fontsize=14, color='red', weight='bold')
axs[0].axis('off')

# TOP 5 MATCHES
for i, idx in enumerate(top_k):
    axs[i+1].imshow(Image.open(gallery_paths[idx]))
    score = similarities[idx]
    color = 'green' if score > 0.7 else 'orange' if score > 0.6 else 'red'
    axs[i+1].set_title(f'Top {i+1}\n{score:.3f}', fontsize=12, color=color)
    axs[i+1].axis('off')

plt.tight_layout()
plt.savefig('top5_new_image.jpg', dpi=200, bbox_inches='tight')
plt.show()

print(f"\n🎉 COMPLETE! Saved: top5_new_image.jpg")
