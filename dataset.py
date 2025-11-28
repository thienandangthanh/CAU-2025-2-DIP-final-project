"""Dataset loading and preprocessing for Zero-DCE model.

This module handles loading and preprocessing of the LOL Dataset for training
the Zero-DCE low-light image enhancement model. It provides functions to load
individual images, create TensorFlow datasets, and split data into train/val/test sets.
"""

import tensorflow as tf
from glob import glob
from typing import Tuple, List
from pathlib import Path


# Configuration constants
DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_TRAIN_IMAGES = 400


def load_data(image_path: str, image_size: int = DEFAULT_IMAGE_SIZE) -> tf.Tensor:
    """Load and preprocess a single image.
    
    Reads an image from disk, decodes it as PNG with 3 channels, resizes to the
    target size, and normalizes pixel values to [0, 1] range.
    
    Args:
        image_path: Path to the image file (PNG format expected)
        image_size: Target size for resizing (height and width)
    
    Returns:
        Preprocessed image tensor of shape (image_size, image_size, 3)
        with values normalized to [0, 1]
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(images=image, size=[image_size, image_size])
    image = image / 255.0
    return image


def data_generator(
    low_light_images: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> tf.data.Dataset:
    """Create TensorFlow dataset from image paths.
    
    Builds a tf.data.Dataset pipeline that loads images in parallel,
    batches them, and optimizes for training performance.
    
    Args:
        low_light_images: List of image file paths
        batch_size: Batch size for training
        image_size: Target image size for resizing
    
    Returns:
        TensorFlow Dataset object configured for training with batched images
    """
    dataset = tf.data.Dataset.from_tensor_slices((low_light_images))
    dataset = dataset.map(
        lambda x: load_data(x, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def get_dataset(
    dataset_path: str = "./lol_dataset",
    max_train_images: int = DEFAULT_MAX_TRAIN_IMAGES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Load and prepare train, validation, and test datasets.
    
    Loads images from the LOL Dataset directory structure:
    - Training images: {dataset_path}/our485/low/*.png
    - Test images: {dataset_path}/eval15/low/*.png
    
    The training set is split into train (first max_train_images) and 
    validation (remaining images).
    
    Args:
        dataset_path: Root path to LOL dataset directory
        max_train_images: Maximum number of images for training (rest used for validation)
        batch_size: Batch size for datasets
        image_size: Target image size for resizing
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_image_paths)
        - train_dataset: Training dataset (batched)
        - val_dataset: Validation dataset (batched)
        - test_image_paths: List of test image file paths
    
    Raises:
        FileNotFoundError: If dataset_path does not exist
        ValueError: If no images are found in the dataset directories
    """
    # Validate dataset path
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_path}\n"
            f"Please download the LOL Dataset and place it in {dataset_path}"
        )
    
    # Load image paths
    train_low_light_path = str(dataset_root / "our485" / "low" / "*")
    test_low_light_path = str(dataset_root / "eval15" / "low" / "*")
    
    train_low_light_images = sorted(glob(train_low_light_path))
    test_low_light_images = sorted(glob(test_low_light_path))
    
    # Validate images were found
    if len(train_low_light_images) == 0:
        raise ValueError(
            f"No training images found in {train_low_light_path}\n"
            f"Please check the dataset structure."
        )
    
    if len(test_low_light_images) == 0:
        raise ValueError(
            f"No test images found in {test_low_light_path}\n"
            f"Please check the dataset structure."
        )
    
    # Split train and validation
    train_images = train_low_light_images[:max_train_images]
    val_images = train_low_light_images[max_train_images:]
    
    # Validate split
    if len(train_images) == 0:
        raise ValueError(
            f"No training images after split. "
            f"max_train_images ({max_train_images}) may be too large or dataset is empty."
        )
    
    if len(val_images) == 0:
        print(
            f"Warning: No validation images. "
            f"Total images: {len(train_low_light_images)}, "
            f"max_train_images: {max_train_images}"
        )
    
    # Create datasets
    train_dataset = data_generator(train_images, batch_size, image_size)
    val_dataset = data_generator(val_images, batch_size, image_size)
    
    return train_dataset, val_dataset, test_low_light_images