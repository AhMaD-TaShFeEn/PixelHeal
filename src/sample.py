import os
import glob
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.model import PixelRNN

# 🧩 Auto-detect the latest checkpoint
checkpoints = sorted(glob.glob("outputs/checkpoints/pixelrnn_epoch*.pth"))
CHECKPOINT_PATH = checkpoints[-1] if checkpoints else None

if CHECKPOINT_PATH is None:
    raise FileNotFoundError("❌ No checkpoint found in outputs/checkpoints/")

# 🩵 Output directory for reconstructed samples
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🧩 Data paths
OCCLUDED_DIR = "data/test"

# 🧩 Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = PixelRNN().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# 🧩 Load occluded test images only
test_images = sorted(os.listdir(OCCLUDED_DIR))[:5]  # show top 5 images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

occluded_batch = []
image_names = []
for name in test_images:
    img_path = os.path.join(OCCLUDED_DIR, name)
    img = Image.open(img_path).convert("RGB")
    occluded_batch.append(transform(img))
    image_names.append(name)

occluded = torch.stack(occluded_batch).to(device)

# 🧩 Run model inference
with torch.no_grad():
    reconstructed = model(occluded).cpu().clamp(0, 1)

# 🧩 Plot occluded vs reconstructed
fig, axes = plt.subplots(2, len(test_images), figsize=(12, 4))
for i, name in enumerate(image_names):
    axes[0, i].imshow(occluded[i].permute(1, 2, 0).cpu())
    axes[0, i].set_title("Occluded")
    axes[1, i].imshow(reconstructed[i].permute(1, 2, 0))
    axes[1, i].set_title("Reconstructed")
    for row in range(2):
        axes[row, i].axis("off")

fig.suptitle(f"Reconstructed Test Samples ({os.path.basename(CHECKPOINT_PATH)})", fontsize=13)
save_path = os.path.join(OUTPUT_DIR, "test_reconstructions.png")
fig.savefig(save_path)
plt.close(fig)

print(f"\n✅ Reconstructed test samples saved to: {save_path}")

# 🧩 Optional: Save each pair (occluded + reconstructed) side-by-side
pair_dir = os.path.join(OUTPUT_DIR, "pairs")
os.makedirs(pair_dir, exist_ok=True)
for i, name in enumerate(image_names):
    pair_fig, axarr = plt.subplots(1, 2, figsize=(4, 2))
    axarr[0].imshow(occluded[i].permute(1, 2, 0).cpu())
    axarr[0].set_title("Occluded")
    axarr[1].imshow(reconstructed[i].permute(1, 2, 0))
    axarr[1].set_title("Reconstructed")
    for ax in axarr:
        ax.axis("off")
    pair_path = os.path.join(pair_dir, f"{os.path.splitext(name)[0]}_pair.png")
    pair_fig.savefig(pair_path)
    plt.close(pair_fig)

print(f"🖼️ Individual pairs saved in: {pair_dir}")
