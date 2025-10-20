import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PairedImageDataset(Dataset):
    def __init__(self, occluded_dir, original_dir, img_size=64):
        self.occluded_dir = occluded_dir
        self.original_dir = original_dir
        self.img_size = img_size
        self.image_names = sorted(os.listdir(occluded_dir))
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        occluded_path = os.path.join(self.occluded_dir, img_name)
        # Remove "occluded_" prefix when looking for the original
        orig_name = img_name.replace("occluded_", "")
        original_path = os.path.join(self.original_dir, orig_name)

        occluded = Image.open(occluded_path).convert("RGB")
        original = Image.open(original_path).convert("RGB")

        return self.transform(occluded), self.transform(original), img_name
