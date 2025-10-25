"""Inference and comparison tool for Zero-DCE.

This script provides functionality to enhance low-light images using Zero-DCE
and compare the results with classical image enhancement methods including
AutoContrast, Histogram Equalization, CLAHE, and Gamma Correction.
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

from model import ZeroDCE


def load_model_for_inference(weights_path: str) -> ZeroDCE:
    """Load trained Zero-DCE model for inference.
    
    Args:
        weights_path: Path to saved model weights (.h5 or .weights.h5 file)
    
    Returns:
        Loaded ZeroDCE model ready for inference
    """
    model = ZeroDCE()
    model.load_weights(weights_path)
    return model


def enhance_with_zero_dce(image: Image.Image, model: ZeroDCE) -> Image.Image:
    """Enhance image using Zero-DCE model.
    
    Converts PIL Image to tensor, runs inference through the model, and
    converts the result back to PIL Image format.
    
    Args:
        image: PIL Image (RGB) to enhance
        model: Trained ZeroDCE model
    
    Returns:
        Enhanced PIL Image (RGB)
    """
    # Convert PIL Image to numpy array
    image_array = keras.utils.img_to_array(image)
    image_array = image_array.astype("float32") / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    # Run inference
    output_image = model(image_array)
    
    # Convert back to PIL Image
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    
    return output_image


def enhance_with_autocontrast(image: Image.Image) -> Image.Image:
    """Enhance using PIL AutoContrast.
    
    AutoContrast automatically normalizes the image by making the darkest
    color black and the lightest color white, then redistributing the values.
    
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


def enhance_with_clahe(image: Image.Image, clip_limit: float = 2.0, tile_size: int = 8) -> Image.Image:
    """Enhance using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    CLAHE is an adaptive version of histogram equalization that operates on
    small regions (tiles) of the image. The clip limit prevents over-amplification
    of noise. Applied to the L channel in LAB color space.
    
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


def enhance_with_gamma_correction(image: Image.Image, gamma: float = 2.2) -> Image.Image:
    """Enhance using gamma correction.
    
    Gamma correction applies a power-law transformation to adjust image
    brightness. Gamma < 1 brightens the image, gamma > 1 darkens it.
    For low-light enhancement, typically use gamma = 2.2 (1/gamma = 0.45).
    
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


def compare_methods(
    input_path: str,
    weights_path: str,
    output_path: str = None,
    methods: list = None,
    save_individual: bool = False
):
    """Compare enhancement methods on a single image.
    
    Applies multiple enhancement methods to an input image and creates a
    side-by-side comparison visualization. Optionally saves individual
    enhanced images.
    
    Args:
        input_path: Path to input low-light image
        weights_path: Path to trained Zero-DCE weights
        output_path: Path to save comparison image (if None, displays instead)
        methods: List of methods to compare (default: all methods)
        save_individual: Whether to save individual enhanced images (default: False)
    """
    # Load original image
    print(f"Loading image: {input_path}")
    original_image = Image.open(input_path)
    
    # Default methods if not specified
    if methods is None:
        methods = ["zero-dce", "autocontrast", "histogram-eq", "clahe", "gamma"]
    
    # Load Zero-DCE model if needed
    model = None
    if "zero-dce" in methods:
        print(f"Loading Zero-DCE model from {weights_path}...")
        model = load_model_for_inference(weights_path)
        print("Model loaded successfully")
    
    # Apply enhancement methods
    results = {"Original": original_image}
    method_map = {
        "zero-dce": ("Zero-DCE", lambda: enhance_with_zero_dce(original_image, model)),
        "autocontrast": ("AutoContrast", lambda: enhance_with_autocontrast(original_image)),
        "histogram-eq": ("Histogram Eq", lambda: enhance_with_histogram_eq(original_image)),
        "clahe": ("CLAHE", lambda: enhance_with_clahe(original_image)),
        "gamma": ("Gamma Correction", lambda: enhance_with_gamma_correction(original_image))
    }
    
    print("\nApplying enhancement methods...")
    for method_key in methods:
        if method_key in method_map:
            method_name, enhance_fn = method_map[method_key]
            print(f"  - {method_name}")
            results[method_name] = enhance_fn()
    
    # Create comparison plot
    num_images = len(results)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    # Handle single subplot case
    if num_images == 1:
        axes = [axes]
    
    for ax, (title, img) in zip(axes, results.items()):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis("off")
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n✅ Comparison saved to {output_path}")
    else:
        print("\nDisplaying comparison...")
        plt.show()
    
    plt.close()
    
    # Save individual images if requested
    if save_individual and output_path:
        output_dir = Path(output_path).parent / "individual"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving individual enhanced images...")
        for title, img in results.items():
            if title != "Original":
                # Create filename from title
                safe_title = title.lower().replace(' ', '_').replace('-', '_')
                save_path = output_dir / f"{Path(input_path).stem}_{safe_title}.png"
                img.save(save_path)
                print(f"  - {save_path}")
        
        print(f"\n✅ Individual images saved to {output_dir}/")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Compare Zero-DCE with classical image enhancement methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all methods
  python compare.py -i input.jpg -w weights.h5 -o comparison.png

  # Compare specific methods only
  python compare.py -i input.jpg -w weights.h5 --methods zero-dce autocontrast

  # Save individual enhanced images
  python compare.py -i input.jpg -w weights.h5 -o output.png --save-individual
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input low-light image"
    )
    parser.add_argument(
        "-w", "--weights",
        type=str,
        required=True,
        help="Path to trained Zero-DCE model weights (.h5 or .weights.h5 file)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save comparison image (if not specified, displays instead)"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        choices=["zero-dce", "autocontrast", "histogram-eq", "clahe", "gamma"],
        help="Enhancement methods to compare (default: all)"
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save individual enhanced images to 'individual/' subdirectory"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        print(f"❌ Error: Input image not found: {args.input}")
        return 1
    
    if not Path(args.weights).exists():
        print(f"❌ Error: Model weights not found: {args.weights}")
        return 1
    
    # Run comparison
    try:
        compare_methods(
            input_path=args.input,
            weights_path=args.weights,
            output_path=args.output,
            methods=args.methods,
            save_individual=args.save_individual
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
