import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import glob
import streamlit as st

# 🧩 Add project root (PixelHeal/) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import PixelRNN


st.set_page_config(page_title="🧠 PixelRevive", layout="centered")
st.title("🧠 PixelRevive — AI Image Completion")
st.write("Upload an **occluded face image**, and let the PixelRNN model reconstruct it!")

# 🧩 Auto-detect latest checkpoint
checkpoints = sorted(glob.glob("outputs/checkpoints/pixelrnn_epoch*.pth"))
CHECKPOINT_PATH = checkpoints[-1] if checkpoints else None
if CHECKPOINT_PATH is None:
    st.error("❌ No checkpoint found in `outputs/checkpoints/`.")
    st.stop()

# 🧩 Load model
@st.cache_resource
def load_model():
    model = PixelRNN()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# 🧩 Image transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()

# 🖼️ Upload Section
uploaded = st.file_uploader("Upload an occluded image", type=['png', 'jpg', 'jpeg'])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Occluded Image", use_container_width=True)

    # 🧩 Run model inference
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).squeeze(0)

    reconstructed = to_pil(pred.clamp(0, 1))

    # 🎨 Show results side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Occluded", use_container_width=True)
    with col2:
        st.image(reconstructed, caption="Reconstructed", use_container_width=True)

    # 💾 Save reconstructed output
    output_path = os.path.join("outputs", "streamlit_reconstructed.png")
    reconstructed.save(output_path)
    st.success(f"✅ Reconstruction complete! Saved to `{output_path}`.")
