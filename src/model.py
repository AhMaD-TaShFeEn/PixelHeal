import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    """
    Convolution layer with a mask to preserve autoregressive property.
    mask_type: 'A' for first layer (excludes current pixel),
                'B' for later layers (includes current pixel).
    """
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ['A', 'B']
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, h, w = self.weight.size()

        # Mask future pixels
        self.mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelRNN(nn.Module):
    """
    Simplified PixelRNN/PixelCNN for image inpainting.
    Input: Occluded image (C,H,W)
    Output: Reconstructed image (C,H,W)
    """
    def __init__(self, input_channels=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            MaskedConv2d('A', input_channels, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            MaskedConv2d('B', hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            MaskedConv2d('B', hidden_dim, input_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)
