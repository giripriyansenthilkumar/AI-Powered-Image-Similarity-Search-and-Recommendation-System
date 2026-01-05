import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from triplet_dataset import get_triplet_dataloader
from preprocess import preprocess


# ============================
# MODEL
# ============================
class TripletNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = torch.hub.load(
            "pytorch/vision",
            "resnet50",
            pretrained=True
        )
        base.fc = nn.Identity()
        self.encoder = base
        self.embed = nn.Linear(2048, 128)

    def forward(self, x):
        x = self.encoder(x)
        x = self.embed(x)
        return F.normalize(x, p=2, dim=1)


# ============================
# LOSS
# ============================
def triplet_loss(anchor, positive, negative, margin=0.3):
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()


# ============================
# DATASET UTILS
# ============================
def build_class_to_images(root_dir):
    class_to_images = defaultdict(list)

    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img in os.listdir(class_dir):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                class_to_images[class_name].append(
                    os.path.join(class_dir, img)
                )

    total = sum(len(v) for v in class_to_images.values())
    print(f"Found {total} images across {len(class_to_images)} classes")
    return class_to_images


# ============================
# TRAINING
# ============================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TripletNet().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    train_class_to_images = build_class_to_images(
        "fashion-dataset/train"
    )

    dataloader = get_triplet_dataloader(
        train_class_to_images,
        batch_size=32,
        transform=preprocess
    )

    os.makedirs("models", exist_ok=True)
    num_epochs = 20

    # ============================
    # LOAD EPOCH 2
    # ============================
    start_epoch = 1
    loaded_epoch_loss = None
    checkpoint_path = "models/triplet_model_epoch_2.pth"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        loaded_epoch_loss = checkpoint["loss"]

        print(
            f"✅ Loaded epoch {checkpoint['epoch']} "
            f"| Avg Loss: {loaded_epoch_loss:.4f}"
        )
        print(f"➡️ Starting from epoch {start_epoch}")
    else:
        print("❌ epoch_2.pth not found. Starting from epoch 1")

    # ============================
    # TRAIN LOOP
    # ============================
    try:
        for epoch in range(start_epoch, num_epochs + 1):
            model.train()
            total_loss = 0.0

            for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                optimizer.zero_grad()

                a_embed = model(anchor)
                p_embed = model(positive)
                n_embed = model(negative)

                loss = triplet_loss(a_embed, p_embed, n_embed)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )
                optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch [{epoch}/{num_epochs}] "
                        f"Batch [{batch_idx}/{len(dataloader)}] "
                        f"Loss: {batch_loss:.4f}"
                    )

            avg_loss = total_loss / len(dataloader)
            print(
                f"✅ Epoch {epoch} completed | "
                f"Avg Loss: {avg_loss:.4f}"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                f"models/triplet_model_epoch_{epoch}.pth"
            )

            print(f"💾 Saved models/triplet_model_epoch_{epoch}.pth")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n⏹️ Ctrl+C detected")
        print(
            f"📉 Current Batch Loss: {batch_loss:.4f} | "
            f"Loaded Epoch Loss: {loaded_epoch_loss:.4f}"
        )

        if loaded_epoch_loss is not None and batch_loss < loaded_epoch_loss:
            partial_path = f"models/triplet_model_epoch_{epoch}_partial.pth"

            torch.save(
                {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": batch_loss,
                },
                partial_path
            )

            print(
                f"💾 Partial model saved "
                f"(batch loss {batch_loss:.4f} < "
                f"loaded loss {loaded_epoch_loss:.4f})"
            )
        else:
            print(
                f"❌ Partial model NOT saved "
                f"(batch loss {batch_loss:.4f} >= "
                f"loaded loss {loaded_epoch_loss:.4f})"
            )

        print("🛑 Training stopped")
        return

    print("🎉 Training finished successfully!")


# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    train()
