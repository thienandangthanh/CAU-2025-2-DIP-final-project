"""Training script for Zero-DCE model.

This script provides a command-line interface for training the Zero-DCE model
on the LOL Dataset. It integrates the dataset loading, model architecture, and
loss functions into a complete training pipeline with configurable hyperparameters.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from dataset import get_dataset
from model import ZeroDCE
from plot_utils import configure_publication_style

# Apply consistent publication-quality styling for all plots
configure_publication_style()


def plot_training_history(history, save_dir: str = "./training_plots"):
    """Plot and save training history curves.

    Creates individual plots for each loss metric showing both training and
    validation curves over epochs. Saves plots as PNG files in the specified
    directory.

    Args:
        history: Keras History object returned from model.fit()
        save_dir: Directory to save plot images (default: "./training_plots")
    """
    metrics = [
        "total_loss",
        "illumination_smoothness_loss",
        "spatial_constancy_loss",
        "color_constancy_loss",
        "exposure_loss",
    ]

    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Plot each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history[metric], label=metric)
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.title(f"Train and Validation {metric} Over Epochs")
        plt.legend()
        plt.grid()
        plt.savefig(f"{save_dir}/{metric}.png")
        plt.close()

    print(f"Training plots saved to {save_dir}/")


def main():
    """Main training function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train Zero-DCE model for low-light image enhancement"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./lol_dataset",
        help="Path to LOL dataset directory",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=400,
        help="Maximum number of images for training",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Image size (height and width)"
    )

    # Output arguments
    parser.add_argument(
        "--save-path",
        type=str,
        default="./weights/zero_dce.weights.h5",
        help="Path to save trained model weights",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="./training_plots",
        help="Directory to save training plots",
    )

    args = parser.parse_args()

    # Create output directories
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset, _ = get_dataset(
        dataset_path=args.dataset_path,
        max_train_images=args.max_train_images,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    print(f"Train Dataset: {train_dataset}")
    print(f"Validation Dataset: {val_dataset}")

    # Build and compile model
    print("\nBuilding Zero-DCE model...")
    zero_dce_model = ZeroDCE()
    zero_dce_model.compile(learning_rate=args.learning_rate)

    # Train model
    print(f"\nTraining for {args.epochs} epochs...")
    history = zero_dce_model.fit(
        train_dataset, validation_data=val_dataset, epochs=args.epochs
    )

    # Save model weights
    print(f"\nSaving model weights to {args.save_path}...")
    zero_dce_model.save_weights(args.save_path)

    # Plot training history
    print("\nGenerating training plots...")
    plot_training_history(history, args.plot_dir)

    print("\n✅ Training completed successfully!")
    print(f"   Weights saved: {args.save_path}")
    print(f"   Plots saved: {args.plot_dir}/")


if __name__ == "__main__":
    main()
