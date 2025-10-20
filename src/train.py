import os
from src import model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from src.dataset import PairedImageDataset as FaceDataset
from src.model import PixelRNN
import matplotlib.pyplot as plt

def train_model(
    occluded_dir="data/train/occluded",
    original_dir="data/train/original",
    batch_size=8,
    num_epochs=20,
    lr=1e-3,
    img_size=64,
    checkpoint_dir="outputs/checkpoints",
    log_dir="outputs/logs",
    figures_dir="outputs/figures",
    save_figures=True
):
    # 🩵 Always resolve paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    checkpoint_dir = os.path.join(project_root, checkpoint_dir)
    log_dir = os.path.join(project_root, log_dir)
    figures_dir = os.path.join(project_root, figures_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if save_figures:
        os.makedirs(figures_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 🔹 Data loading
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    dataset = FaceDataset(occluded_dir, original_dir, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 🔹 Model, loss, optimizer
    model = PixelRNN(img_size=img_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 🔹 Training loop
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            occluded, original = batch[0], batch[1]

            occluded, original = occluded.to(device), original.to(device)
            optimizer.zero_grad()
            output = model(occluded)
            loss = criterion(output, original)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

        # 🧩 Save checkpoint
        torch.save(model.state_dict(),
                   os.path.join(checkpoint_dir, f"pixelrnn_epoch{epoch+1}.pth"))

        # 🧩 Save sample reconstructions
        if save_figures and (epoch + 1) % 2 == 0:  # every 2 epochs
            model.eval()
            with torch.no_grad():
                batch = next(iter(dataloader))
                occluded, original = batch[0].to(device), batch[1].to(device)
                output = model(occluded)

            fig, axes = plt.subplots(3, 5, figsize=(12, 6))
            for j in range(5):
                axes[0, j].imshow(occluded[j].permute(1, 2, 0).cpu())
                axes[0, j].set_title("Occluded")
                axes[1, j].imshow(output[j].permute(1, 2, 0).cpu().clamp(0, 1))
                axes[1, j].set_title("Predicted")
                axes[2, j].imshow(original[j].permute(1, 2, 0).cpu())
                axes[2, j].set_title("Original")
                for row in range(3):
                    axes[row, j].axis("off")
            fig.suptitle(f"Epoch {epoch+1} Reconstructions")
            fig.savefig(os.path.join(figures_dir, f"epoch_{epoch+1}_samples.png"))
            plt.close(fig)

    # 🔹 Plot training loss
    plt.figure()
    plt.plot(range(1, num_epochs+1), epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig(os.path.join(log_dir, "loss_curve.png"))
    plt.close()

    print("\n✅ Training complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")
    if save_figures:
        print(f"Sample figures saved to: {figures_dir}")

    return model
