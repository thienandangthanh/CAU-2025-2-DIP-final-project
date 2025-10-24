import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorConstancyLoss(nn.Module):
    """
    Enforces color constancy by penalizing deviations in the color channels of the enhanced image.
    This is based on the "gray-world" hypothesis.
    """
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, enhanced_image):
        # Calculate the mean of each color channel
        mean_r = torch.mean(enhanced_image[:, 0, :, :])
        mean_g = torch.mean(enhanced_image[:, 1, :, :])
        mean_b = torch.mean(enhanced_image[:, 2, :, :])

        # Calculate the squared differences between the channel means
        d_rg = (mean_r - mean_g) ** 2
        d_rb = (mean_r - mean_b) ** 2
        d_gb = (mean_g - mean_b) ** 2

        # The loss is the sum of these differences
        loss = d_rg + d_rb + d_gb
        return loss

class ExposureControlLoss(nn.Module):
    """
    Controls the exposure level of the image.
    Penalizes patches that are too dark or too bright.
    """
    def __init__(self, patch_size=16, mean_val=0.6):
        super(ExposureControlLoss, self).__init__()
        self.patch_size = patch_size
        self.mean_val = mean_val
        # Use average pooling to efficiently calculate the mean of non-overlapping patches
        self.pool = nn.AvgPool2d(self.patch_size)

    def forward(self, enhanced_image):
        # Convert the image to grayscale for intensity calculation
        # Using the standard luminosity formula: Y = 0.299*R + 0.587*G + 0.114*B
        img_gray = 0.299 * enhanced_image[:, 0, :, :] + \
                   0.587 * enhanced_image[:, 1, :, :] + \
                   0.114 * enhanced_image[:, 2, :, :]
        
        # Add a channel dimension for the pooling layer
        img_gray = img_gray.unsqueeze(1)

        # Get the mean intensity of each patch
        mean_intensity_patches = self.pool(img_gray)
        
        # Calculate the L1 distance from the desired mean value
        loss = torch.mean(torch.abs(mean_intensity_patches - self.mean_val))
        return loss

class IlluminationSmoothnessLoss(nn.Module):
    """
    Enforces smoothness in the illumination maps (the curve parameters A).
    This is a Total Variation (TV) loss on the parameter maps.
    """
    def __init__(self):
        super(IlluminationSmoothnessLoss, self).__init__()

    def forward(self, curve_params):
        # Split the 24 channels into 8 separate 3-channel maps
        batch_size, _, h, w = curve_params.shape
        curve_maps = torch.split(curve_params, 3, dim=1)
        
        loss = 0
        for A_n in curve_maps:
            # Horizontal and vertical gradients
            grad_h = torch.abs(A_n[:, :, :, :-1] - A_n[:, :, :, 1:])
            grad_v = torch.abs(A_n[:, :, :-1, :] - A_n[:, :, 1:, :])
            loss += torch.sum(grad_h) + torch.sum(grad_v)
        
        # Normalize by batch size
        return loss / batch_size

class SpatialConsistencyLoss(nn.Module):
    """
    Preserves the spatial coherence of the image by maintaining contrast
    between neighboring regions in the original and enhanced images.
    """
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        # Kernels to compute differences with left, right, up, and down neighbors
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        
        self.kernels = nn.Parameter(torch.cat([kernel_left, kernel_right, kernel_up, kernel_down], dim=0), requires_grad=False)

    def forward(self, original_image, enhanced_image):
        # Convert images to grayscale
        original_gray = 0.299 * original_image[:, 0, :, :] + 0.587 * original_image[:, 1, :, :] + 0.114 * original_image[:, 2, :, :]
        enhanced_gray = 0.299 * enhanced_image[:, 0, :, :] + 0.587 * enhanced_image[:, 1, :, :] + 0.114 * enhanced_image[:, 2, :, :]
        
        original_gray = original_gray.unsqueeze(1)
        enhanced_gray = enhanced_gray.unsqueeze(1)

        # Compute the gradients (differences with neighbors)
        # padding='same' would be ideal, but for simplicity we use 'replicate' which is close
        d_original = F.conv2d(original_gray, self.kernels, padding='same')
        d_enhanced = F.conv2d(enhanced_gray, self.kernels, padding='same')
        
        # Calculate the L1 loss between the gradients
        loss = torch.sum(torch.abs(d_original - d_enhanced))
        
        return loss

class TotalLoss(nn.Module):
    """
    Combines all the individual loss components with their respective weights.
    """
    def __init__(self, W_spa=1.0, W_exp=10.0, W_col=5.0, W_tvA=200.0):
        super(TotalLoss, self).__init__()
        self.W_spa = W_spa
        self.W_exp = W_exp
        self.W_col = W_col
        self.W_tvA = W_tvA
        
        self.loss_spa = SpatialConsistencyLoss()
        self.loss_exp = ExposureControlLoss()
        self.loss_col = ColorConstancyLoss()
        self.loss_tvA = IlluminationSmoothnessLoss()

    def forward(self, original_image, enhanced_image, curve_params):
        loss_spa = self.loss_spa(original_image, enhanced_image)
        loss_exp = self.loss_exp(enhanced_image)
        loss_col = self.loss_col(enhanced_image)
        loss_tvA = self.loss_tvA(curve_params)

        total_loss = self.W_spa * loss_spa + \
                     self.W_exp * loss_exp + \
                     self.W_col * loss_col + \
                     self.W_tvA * loss_tvA
                     
        return total_loss, loss_spa, loss_exp, loss_col, loss_tvA

# --- Simple test to verify implementation ---
if __name__ == '__main__':
    # Create dummy tensors
    original_img = torch.randn(1, 3, 256, 256)
    enhanced_img = torch.randn(1, 3, 256, 256).clamp(0,1) # Enhanced should be in [0,1]
    curve_params = torch.randn(1, 24, 256, 256)

    # Instantiate the total loss function
    criterion = TotalLoss()

    # Calculate the loss
    total_loss, spa, exp, col, tvA = criterion(original_img, enhanced_img, curve_params)

    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"  - Spatial Consistency Loss: {spa.item():.4f}")
    print(f"  - Exposure Control Loss: {exp.item():.4f}")
    print(f"  - Color Constancy Loss: {col.item():.4f}")
    print(f"  - Illumination Smoothness Loss: {tvA.item():.4f}")
    print("Loss functions implemented successfully!")