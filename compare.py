"""Inference and comparison tool for Zero-DCE.

This script provides functionality to enhance low-light images using Zero-DCE
and compare the results with classical image enhancement methods. It serves
as a visualization tool to compare different enhancement approaches side-by-side.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from classical_methods import CLASSICAL_METHODS
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


def compare_methods(
    input_path: str,
    weights_path: str,
    output_path: str = None,
    methods: list = None,
    save_individual: bool = False,
    reference_path: str = None,
):
    """Compare enhancement methods on a single image.

    Applies multiple enhancement methods to an input image and creates a
    side-by-side comparison visualization. Optionally saves individual
    enhanced images. Can include a reference (well-exposed) image for
    comparison.

    Args:
        input_path: Path to input low-light image
        weights_path: Path to trained Zero-DCE weights
        output_path: Path to save comparison image (if None, displays instead)
        methods: List of methods to compare (default: all methods)
        save_individual: Whether to save individual enhanced images (default: False)
        reference_path: Path to reference (high-light) image (optional)
    """
    # Load original image
    print(f"Loading image: {input_path}")
    original_image = Image.open(input_path)

    # Load reference image if provided
    reference_image = None
    if reference_path:
        if Path(reference_path).exists():
            print(f"Loading reference image: {reference_path}")
            reference_image = Image.open(reference_path)
            # Ensure reference image is RGB
            if reference_image.mode != "RGB":
                reference_image = reference_image.convert("RGB")
        else:
            print(f"⚠️  Warning: Reference image not found: {reference_path}")
            print("   Continuing without reference image...")

    # Default methods if not specified
    if methods is None:
        methods = [
            "zero-dce",
            "autocontrast",
            "histogram-eq",
            "clahe",
            "gamma",
            "msrcr",
        ]

    # Load Zero-DCE model if needed
    model = None
    if "zero-dce" in methods:
        print(f"Loading Zero-DCE model from {weights_path}...")
        model = load_model_for_inference(weights_path)
        print("Model loaded successfully")

    # Apply enhancement methods
    # Start with original image
    results = {"Original": original_image}

    # Add reference image if provided (insert after original)
    if reference_image is not None:
        results["Reference (Ground Truth)"] = reference_image

    print("\nApplying enhancement methods...")
    for method_key in methods:
        # Handle Zero-DCE separately
        if method_key == "zero-dce":
            print("  - Zero-DCE")
            results["Zero-DCE"] = enhance_with_zero_dce(original_image, model)
            # Handle classical methods from classical_methods module
        elif method_key in CLASSICAL_METHODS:
            method_info = CLASSICAL_METHODS[method_key]
            method_name = method_info["name"]
            enhance_fn = method_info["function"]
            print(f"  - {method_name}")
            results[method_name] = enhance_fn(original_image)
        else:
            print(f"  ⚠️  Warning: Unknown method '{method_key}', skipping...")

    # Create comparison plot
    num_images = len(results)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    # Handle single subplot case
    if num_images == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, results.items(), strict=True):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight="bold")
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

        print("\nSaving individual enhanced images...")
        for title, img in results.items():
            if title != "Original":
                # Create filename from title
                safe_title = title.lower().replace(" ", "_").replace("-", "_")
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

  # Compare with reference (ground truth) image
  python compare.py -i low/1.png -w weights.h5 -r high/1.png -o comparison.png

  # Compare specific methods only
  python compare.py -i input.jpg -w weights.h5 --methods zero-dce autocontrast

  # Save individual enhanced images
  python compare.py -i input.jpg -w weights.h5 -o output.png --save-individual
        """,
    )

    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to input low-light image"
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        required=True,
        help="Path to trained Zero-DCE model weights (.h5 or .weights.h5 file)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to save comparison image (if not specified, displays instead)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        choices=["zero-dce", "autocontrast", "histogram-eq", "clahe", "gamma", "msrcr"],
        help="Enhancement methods to compare (default: all)",
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save individual enhanced images to 'individual/' subdirectory",
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        default=None,
        help="Path to reference (well-exposed) image for comparison (optional)",
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
            save_individual=args.save_individual,
            reference_path=args.reference,
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
