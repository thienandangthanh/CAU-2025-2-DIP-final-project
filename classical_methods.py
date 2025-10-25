"""Classical image enhancement methods for low-light images.

This module provides traditional (non-deep-learning) image enhancement methods
that can be used as baselines for comparison with Zero-DCE. All methods work
on PIL Image objects and return enhanced PIL Images.
"""

import numpy as np
import cv2
from PIL import Image, ImageOps
from typing import Union


def enhance_with_autocontrast(image: Image.Image) -> Image.Image:
    """Enhance using PIL AutoContrast.
    
    AutoContrast automatically normalizes the image by making the darkest
    color black and the lightest color white, then redistributing the values
    in between. This is a simple global contrast adjustment.
    
    Args:
        image: PIL Image (RGB)
    
    Returns:
        Enhanced PIL Image (RGB)
    """
    return ImageOps.autocontrast(image)


def enhance_with_histogram_eq(image: Image.Image) -> Image.Image:
    """Enhance using histogram equalization.
    
    Applies histogram equalization to the luminance (Y) channel in YUV
    color space, which improves contrast while preserving color information.
    This method redistributes pixel intensities to achieve a uniform histogram.
    
    Args:
        image: PIL Image (RGB)
    
    Returns:
        Enhanced PIL Image (RGB)
    """
    img_array = np.array(image)
    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    
    # Apply histogram equalization to Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_output)


def enhance_with_clahe(
    image: Image.Image, 
    clip_limit: float = 2.0, 
    tile_size: int = 8
) -> Image.Image:
    """Enhance using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    CLAHE is an adaptive version of histogram equalization that operates on
    small regions (tiles) of the image. The clip limit prevents over-amplification
    of noise. Applied to the L channel in LAB color space for better color
    preservation.
    
    Args:
        image: PIL Image (RGB)
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_size: Size of grid for histogram equalization (default: 8)
    
    Returns:
        Enhanced PIL Image (RGB)
    """
    img_array = np.array(image)
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    
    img_output = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_output)


def enhance_with_gamma_correction(
    image: Image.Image, 
    gamma: float = 2.2
) -> Image.Image:
    """Enhance using gamma correction.
    
    Gamma correction applies a power-law transformation to adjust image
    brightness. Gamma < 1 brightens the image, gamma > 1 darkens it.
    For low-light enhancement, typically use gamma = 2.2 (1/gamma = 0.45).
    
    Formula: output = input^(1/gamma)
    
    Args:
        image: PIL Image (RGB)
        gamma: Gamma value for correction (default: 2.2)
    
    Returns:
        Enhanced PIL Image (RGB)
    """
    img_array = np.array(image) / 255.0
    img_corrected = np.power(img_array, 1.0 / gamma)
    img_output = (img_corrected * 255).astype(np.uint8)
    return Image.fromarray(img_output)


# Dictionary mapping method names to enhancement functions
CLASSICAL_METHODS = {
    "autocontrast": {
        "name": "AutoContrast",
        "function": enhance_with_autocontrast,
        "description": "Global contrast normalization"
    },
    "histogram-eq": {
        "name": "Histogram Eq",
        "function": enhance_with_histogram_eq,
        "description": "Histogram equalization on YUV"
    },
    "clahe": {
        "name": "CLAHE",
        "function": enhance_with_clahe,
        "description": "Contrast Limited Adaptive Histogram Equalization"
    },
    "gamma": {
        "name": "Gamma Correction",
        "function": enhance_with_gamma_correction,
        "description": "Power-law transformation (gamma=2.2)"
    }
}


def get_available_methods():
    """Get list of available classical enhancement method names.
    
    Returns:
        List of method keys that can be used
    """
    return list(CLASSICAL_METHODS.keys())


def get_method_info(method_key: str) -> dict:
    """Get information about a specific enhancement method.
    
    Args:
        method_key: Key for the enhancement method
    
    Returns:
        Dictionary with name, function, and description
    
    Raises:
        KeyError: If method_key is not found
    """
    return CLASSICAL_METHODS[method_key]
