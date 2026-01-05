import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

print("🚀 EPOCH 2 - ERROR PROOF!")

# MODEL
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
checkpoint = torch.load("models/triplet_model_epoch_2.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(path):
    return transform(Image.open(path).convert('RGB'))

# 🔥 SIMPLE: GET FIRST IMAGE (NO LISTS!)
query_path = None
for root, dirs, files in os.walk('fashion-dataset'):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            query_path = os.path.join(root, file)
            break
    if query_path:
        break

print(f"🎯 Query: {os.path.basename(query_path)}")

# 1000 GALLERY (NO QUERY)
gallery_paths = []
count = 0
for root, dirs, files in os.walk('fashion-dataset'):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and os.path.join(root, file) != query_path:
            gallery_paths.append(os.path.join(root, file))
            count += 1
            if count >= 1000:
                break
    if count >= 1000:
        break

print(f"✅ Gallery: {len(gallery_paths)} images")

# GALLERY EMBEDDINGS
print("Extracting...")
batch_size = 64
gallery_embs = []
for i in range(0, len(gallery_paths), batch_size):
    batch_paths = gallery_paths[i:i+batch_size]
    batch_imgs = torch.stack([load_image(p) for p in batch_paths]).to(device)
    with torch.no_grad():
        batch_embs = model(batch_imgs).cpu().numpy()
    gallery_embs.append(batch_embs)

gallery_embs = np.vstack(gallery_embs)

# QUERY
query_img = load_image(query_path).unsqueeze(0).to(device)
with torch.no_grad():
    query_emb = model(query_img).cpu().numpy()

# COSINE
similarities = cosine_similarity(query_emb, gallery_embs)[0]
top_k = np.argsort(similarities)[::-1][:5]

print("\n🚀 TOP 5:")
print("="*40)
for i, idx in enumerate(top_k):
    score = similarities[idx]
    print(f"{i+1}. {os.path.basename(gallery_paths[idx])} | {score:.3f}")

# PLOT
fig, axs = plt.subplots(1, 6, figsize=(18, 3))
fig.suptitle(f'EPOCH 2: {os.path.basename(query_path)}', fontsize=16)

axs[0].imshow(Image.open(query_path))
axs[0].set_title('Query')
axs[0].axis('off')

for i, idx in enumerate(top_k):
    axs[i+1].imshow(Image.open(gallery_paths[idx]))
    axs[i+1].set_title(f'#{i+1}: {similarities[idx]:.3f}')
    axs[i+1].axis('off')

plt.tight_layout()
plt.savefig('epoch2_safe.jpg', dpi=200, bbox_inches='tight')
plt.show()

print("✅ ZERO ERRORS!")
