import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DCENet(nn.Module):
    def __init__(self, n_filters=32):
        super(DCENet, self).__init__()

        # --- Convolutional Layers ---
        # The network takes a 3-channel (RGB) image as input.
        self.conv1 = nn.Conv2d(3, n_filters, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias=True)

        # The symmetrical layers (decoders) that will receive skip connections
        # Input channels = n_filters (from previous layer) + n_filters (from skip connection)
        self.conv5 = nn.Conv2d(n_filters * 2, n_filters, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(n_filters * 2, n_filters, 3, 1, 1, bias=True)

        # Final layer:
        # Input: n_filters * 2
        # Output: 24 channels. Why 24?
        # We use 8 iterations of curve application. Each iteration needs 3 parameters (one for R, G, B).
        # 8 iterations * 3 channels = 24 parameter maps total.
        self.conv7 = nn.Conv2d(n_filters * 2, 24, 3, 1, 1, bias=True)

        # Activation function for intermediate layers
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # --- Feature Extraction (Encoder) ---
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))

        # --- Feature Reconstruction (Decoder with Skip Connections) ---
        # Concatenate output of conv4 with output of conv3
        x5 = self.relu(self.conv5(torch.cat([x4, x3], 1)))
        # Concatenate output of conv5 with output of conv2
        x6 = self.relu(self.conv6(torch.cat([x5, x2], 1)))

        # Final layer: Concatenate output of conv6 with output of conv1
        # Use Tanh to keep the estimated curve parameters bounded between [-1, 1]
        enhanced_curve_params = F.tanh(self.conv7(torch.cat([x6, x1], 1)))

        return enhanced_curve_params


# --- The Core Enhancement Function ---
# This function sits outside the class (or could be static inside) to be used easily.
# It takes the original image and the parameters from DCENet to create the final image.
def enhance(x, active_maps, n_iterations=8):
    """
    Applies the estimated curve parameters iteratively to the input image.
    
    Args:
        x: Input low-light image (Batch, 3, Height, Width)
        active_maps: The 24-channel output from DCENet (Batch, 24, Height, Width)
        n_iterations: Number of iterations to apply the curve (default 8)
    
    Returns:
        enhanced_image: The final enhanced result.
        active_maps: Returns the maps too, sometimes needed for visualization or regularizers.
    """
    
    x_enhanced = x
    
    # We split the 24-channel map into 8 separate chunks of 3 channels each.
    # Each chunk contains the (R,G,B) curve parameters for one iteration.
    curve_maps = torch.split(active_maps, 3, dim=1) 

    # Iteratively apply the curve formula:
    # LE(n) = LE(n-1) + A_n * LE(n-1) * (1 - LE(n-1))
    for i in range(n_iterations):
        A_n = curve_maps[i]
        x_enhanced = x_enhanced + A_n * x_enhanced * (1 - x_enhanced)
        
    return x_enhanced

# --- Simple test to verify standard implementation ---
if __name__ == '__main__':
    # Create a dummy input image (batch_size=1, channels=3, height=256, width=256)
    img = torch.randn(1, 3, 256, 256)
    net = DCENet()
    
    # 1. Get parameters
    A = net(img)
    print(f"Parameter maps shape: {A.shape}") # Should be (1, 24, 256, 256)

    # 2. Enhance image
    res = enhance(img, A)
    print(f"Enhanced image shape: {res.shape}") # Should be (1, 3, 256, 256)
    print("Model implemented successfully!")