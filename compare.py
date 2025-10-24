import os
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# --- Import local modules ---
from model import DCENet, enhance

def apply_gamma_correction(image_bgr, gamma=2.2):
    """Applies gamma correction to a BGR image."""
    # Convert from int [0, 255] to float [0, 1]
    image_float = image_bgr.astype(np.float32) / 255.0
    # Apply gamma correction
    corrected_image = np.power(image_float, 1.0/gamma)
    # Convert back to [0, 255]
    corrected_image_uint8 = (corrected_image * 255.0).astype(np.uint8)
    return corrected_image_uint8

def apply_histogram_equalization(image_bgr):
    """Applies Histogram Equalization to a BGR image."""
    # Convert to YUV color space
    img_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    # Apply histogram equalization to the Y (luminance) channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # Convert back to BGR color space
    equalized_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return equalized_image

def apply_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applies CLAHE to a BGR image."""
    # Convert to YUV color space
    img_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # Apply CLAHE to the Y (luminance) channel
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    # Convert back to BGR color space
    clahe_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return clahe_image

def test_zerodce(image_tensor, model, device):
    """Enhances an image using the trained Zero-DCE model."""
    with torch.no_grad():
        # Add a batch dimension and move to device
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        # Get curve parameters and enhance
        curve_params = model(image_batch)
        enhanced_image = enhance(image_batch, curve_params)
        
        # Remove batch dimension and move to CPU
        enhanced_image = enhanced_image.squeeze(0).cpu()
        
        # Convert from PyTorch tensor (C, H, W) to NumPy array (H, W, C)
        enhanced_np = enhanced_image.permute(1, 2, 0).numpy()
        
        # Convert from RGB [0, 1] to BGR [0, 255] for OpenCV display
        enhanced_bgr = (enhanced_np * 255.0).clip(0, 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced_bgr, cv2.COLOR_RGB2BGR)
        
        return enhanced_bgr

def main(args):
    # --- Setup device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model ---
    model = DCENet().to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.weights_path}")

    # --- Load and Prepare Image ---
    # Load with OpenCV for classical methods (BGR format)
    original_image_bgr = cv2.imread(args.image_path)
    if original_image_bgr is None:
        print(f"Error: Could not load image from {args.image_path}")
        return

    # Prepare image for Zero-DCE (RGB Tensor)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    image_pil = Image.open(args.image_path).convert('RGB')
    image_tensor = transform(image_pil)

    # --- Apply All Enhancement Methods ---
    gamma_result = apply_gamma_correction(original_image_bgr)
    he_result = apply_histogram_equalization(original_image_bgr)
    clahe_result = apply_clahe(original_image_bgr)
    zerodce_result = test_zerodce(image_tensor, model, device)
    
    # Resize all images to the same size for consistent display
    display_size = (args.image_size, args.image_size)
    original_display = cv2.resize(original_image_bgr, display_size)
    gamma_display = cv2.resize(gamma_result, display_size)
    he_display = cv2.resize(he_result, display_size)
    clahe_display = cv2.resize(clahe_result, display_size)
    zerodce_display = cv2.resize(zerodce_result, display_size)

    # --- Display Results ---
    titles = ['Original', 'Gamma Correction', 'Histogram Equalization', 'CLAHE', 'Zero-DCE']
    images = [original_display, gamma_display, he_display, clahe_display, zerodce_display]

    plt.figure(figsize=(20, 5))
    for i, (title, img) in enumerate(zip(titles, images)):
        # Convert BGR (OpenCV) to RGB (Matplotlib) for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 5, i + 1)
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_result.png')
    plt.show()
    print("Comparison figure saved as 'comparison_result.png'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare Zero-DCE with classical methods.")
    
    parser.add_argument('--image_path', type=str, required=True, help='Path to the low-light test image.')
    parser.add_argument('--weights_path', type=str, default='weights/dce_net_epoch_200.pth', help='Path to the trained model weights.')
    parser.add_argument('--image_size', type=int, default=512, help='Image size for processing and display.')

    args = parser.parse_args()
    main(args)