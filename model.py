"""DCE-Net architecture and ZeroDCE training model.

This module implements the Deep Curve Estimation Network (DCE-Net) for learning
pixel-wise curve parameters, and the ZeroDCE wrapper model that combines the
architecture with unsupervised loss functions for training.
"""

import tensorflow as tf
import keras
from keras import layers

from loss import (
    color_constancy_loss,
    exposure_loss,
    illumination_smoothness_loss,
    SpatialConsistencyLoss
)


def build_dce_net() -> keras.Model:
    """Build DCE-Net architecture.
    
    DCE-Net is a lightweight deep neural network consisting of 7 convolutional
    layers with symmetrical skip connections. The network learns to map input
    low-light images to pixel-wise curve parameter maps.
    
    Architecture:
    - Conv1-4: 32 filters, 3x3 kernels, stride 1, ReLU activation, same padding
    - Skip connection 1: Concatenate Conv4 and Conv3
    - Conv5: 32 filters, applied to concatenated features
    - Skip connection 2: Concatenate Conv5 and Conv2
    - Conv6: 32 filters, applied to concatenated features
    - Skip connection 3: Concatenate Conv6 and Conv1
    - Conv7: 24 filters (8 iterations × 3 channels), Tanh activation
    
    The output has 24 channels representing curve parameters for 8 iterations
    of enhancement, with 3 parameters (one per RGB channel) per iteration.
    
    Returns:
        Keras Model that maps input images of shape (None, None, 3) to 
        curve parameter maps of shape (None, None, 24)
    """
    input_img = keras.Input(shape=[None, None, 3])
    
    # First 4 convolutional layers
    conv1 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(input_img)
    conv2 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv1)
    conv3 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv2)
    conv4 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv3)
    
    # Skip connection 1: Conv4 + Conv3
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(int_con1)
    
    # Skip connection 2: Conv5 + Conv2
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(int_con2)
    
    # Skip connection 3: Conv6 + Conv1
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    
    # Output layer: 24 channels for curve parameters
    x_r = layers.Conv2D(
        24, (3, 3), strides=(1, 1), activation="tanh", padding="same"
    )(int_con3)
    
    return keras.Model(inputs=input_img, outputs=x_r)


def get_enhanced_image(data: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
    """Apply learned curve enhancement iteratively.
    
    Applies 8 iterations of pixel-wise curve adjustments using the formula:
        x_{n+1} = x_n + r_n * (x_n^2 - x_n)
    
    where r_n are the learned curve parameters from DCE-Net, and x_n is the
    image after n iterations (x_0 is the original input image).
    
    This curve formulation ensures:
    1. Values remain in valid range (can slightly exceed [0,1])
    2. Monotonicity is preserved (with proper curve parameters)
    3. Differentiability for backpropagation
    
    Args:
        data: Input low-light image of shape (batch, height, width, 3)
        output: Curve parameter maps of shape (batch, height, width, 24)
                Split into 8 iterations × 3 RGB channels
    
    Returns:
        Enhanced image of shape (batch, height, width, 3)
    """
    # Extract curve parameters for each iteration (3 channels each)
    r1 = output[:, :, :, :3]
    r2 = output[:, :, :, 3:6]
    r3 = output[:, :, :, 6:9]
    r4 = output[:, :, :, 9:12]
    r5 = output[:, :, :, 12:15]
    r6 = output[:, :, :, 15:18]
    r7 = output[:, :, :, 18:21]
    r8 = output[:, :, :, 21:24]
    
    # Apply curve iteratively: x = x + r * (x^2 - x)
    x = data + r1 * (tf.square(data) - data)
    x = x + r2 * (tf.square(x) - x)
    x = x + r3 * (tf.square(x) - x)
    x = x + r4 * (tf.square(x) - x)
    enhanced_image = x + r5 * (tf.square(x) - x)
    x = enhanced_image + r6 * (tf.square(enhanced_image) - enhanced_image)
    x = x + r7 * (tf.square(x) - x)
    enhanced_image = x + r8 * (tf.square(x) - x)
    
    return enhanced_image


class ZeroDCE(keras.Model):
    """Zero-DCE training model wrapper.
    
    This class wraps the DCE-Net architecture with the unsupervised loss
    functions required for zero-reference learning. It handles the complete
    training pipeline including:
    - Forward pass through DCE-Net
    - Curve application for image enhancement
    - Computation of all four loss functions
    - Custom training and validation steps
    - Weight saving/loading (only DCE-Net weights, not wrapper)
    
    The model is trained without paired data by using four unsupervised losses:
    1. Illumination smoothness loss (weight: 200)
    2. Spatial consistency loss (weight: 1)
    3. Color constancy loss (weight: 5)
    4. Exposure loss (weight: 10)
    """
    
    def __init__(self, **kwargs):
        """Initialize the Zero-DCE model.
        
        Creates the DCE-Net architecture that will be trained.
        """
        super().__init__(**kwargs)
        self.dce_model = build_dce_net()

    def compile(self, learning_rate: float, **kwargs):
        """Compile the model with optimizer and loss trackers.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            **kwargs: Additional arguments passed to parent compile()
        """
        super().compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistencyLoss(reduction="none")
        
        # Initialize metric trackers for all losses
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.illumination_smoothness_loss_tracker = keras.metrics.Mean(
            name="illumination_smoothness_loss"
        )
        self.spatial_constancy_loss_tracker = keras.metrics.Mean(
            name="spatial_constancy_loss"
        )
        self.color_constancy_loss_tracker = keras.metrics.Mean(
            name="color_constancy_loss"
        )
        self.exposure_loss_tracker = keras.metrics.Mean(name="exposure_loss")

    @property
    def metrics(self):
        """Return list of metrics tracked during training.
        
        Returns:
            List of Keras metrics for total loss and individual loss components
        """
        return [
            self.total_loss_tracker,
            self.illumination_smoothness_loss_tracker,
            self.spatial_constancy_loss_tracker,
            self.color_constancy_loss_tracker,
            self.exposure_loss_tracker,
        ]

    def call(self, data):
        """Forward pass through the model.
        
        Args:
            data: Input low-light images of shape (batch, height, width, 3)
        
        Returns:
            Enhanced images of shape (batch, height, width, 3)
        """
        dce_net_output = self.dce_model(data)
        return get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data: tf.Tensor, output: tf.Tensor) -> dict:
        """Compute all loss components.
        
        Calculates the four unsupervised losses with their respective weights:
        - Illumination smoothness: 200x (encourages smooth curve parameters)
        - Spatial consistency: 1x (preserves spatial structure)
        - Color constancy: 5x (maintains color balance)
        - Exposure: 10x (prevents under/over-exposure)
        
        Args:
            data: Original low-light images of shape (batch, height, width, 3)
            output: Curve parameter maps of shape (batch, height, width, 24)
        
        Returns:
            Dictionary containing total_loss and individual loss components
        """
        enhanced_image = get_enhanced_image(data, output)
        
        # Compute individual losses with their weights
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(
            self.spatial_constancy_loss(enhanced_image, data)
        )
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))
        
        # Total loss is sum of all components
        total_loss = (
            loss_illumination
            + loss_spatial_constancy
            + loss_color_constancy
            + loss_exposure
        )

        return {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_illumination,
            "spatial_constancy_loss": loss_spatial_constancy,
            "color_constancy_loss": loss_color_constancy,
            "exposure_loss": loss_exposure,
        }

    def train_step(self, data):
        """Custom training step.
        
        Performs one training iteration:
        1. Forward pass through DCE-Net
        2. Compute all losses
        3. Calculate gradients
        4. Update weights
        5. Update metric trackers
        
        Args:
            data: Batch of low-light images
        
        Returns:
            Dictionary of metric names and values for this step
        """
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self.compute_losses(data, output)

        # Compute gradients and update weights
        gradients = tape.gradient(
            losses["total_loss"], self.dce_model.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))

        # Update loss trackers
        self.total_loss_tracker.update_state(losses["total_loss"])
        self.illumination_smoothness_loss_tracker.update_state(
            losses["illumination_smoothness_loss"]
        )
        self.spatial_constancy_loss_tracker.update_state(
            losses["spatial_constancy_loss"]
        )
        self.color_constancy_loss_tracker.update_state(losses["color_constancy_loss"])
        self.exposure_loss_tracker.update_state(losses["exposure_loss"])

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        """Custom validation/test step.
        
        Performs one validation iteration:
        1. Forward pass through DCE-Net
        2. Compute all losses
        3. Update metric trackers (no gradient computation or weight updates)
        
        Args:
            data: Batch of low-light images
        
        Returns:
            Dictionary of metric names and values for this step
        """
        output = self.dce_model(data)
        losses = self.compute_losses(data, output)

        # Update loss trackers (no training)
        self.total_loss_tracker.update_state(losses["total_loss"])
        self.illumination_smoothness_loss_tracker.update_state(
            losses["illumination_smoothness_loss"]
        )
        self.spatial_constancy_loss_tracker.update_state(
            losses["spatial_constancy_loss"]
        )
        self.color_constancy_loss_tracker.update_state(losses["color_constancy_loss"])
        self.exposure_loss_tracker.update_state(losses["exposure_loss"])

        return {metric.name: metric.result() for metric in self.metrics}

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """Save model weights.
        
        Only saves the DCE-Net weights, not the wrapper model weights.
        This ensures compatibility and smaller file sizes.
        
        Args:
            filepath: Path to save weights file (.h5 format recommended)
            overwrite: Whether to overwrite existing file (unused in Keras 3)
            save_format: Format to save in (unused in Keras 3)
            options: Additional save options (unused in Keras 3)
        """
        # Keras 3 simplified API - only filepath is needed
        self.dce_model.save_weights(filepath)

    def load_weights(self, filepath, skip_mismatch=False, by_name=False, options=None):
        """Load model weights.
        
        Only loads the DCE-Net weights, not the wrapper model weights.
        This ensures compatibility with saved models.
        
        Args:
            filepath: Path to weights file
            skip_mismatch: Whether to skip layers with mismatched shapes
            by_name: Whether to load weights by layer name (unused in Keras 3)
            options: Additional load options (unused in Keras 3)
        """
        # Keras 3 simplified API
        self.dce_model.load_weights(
            filepath=filepath,
            skip_mismatch=skip_mismatch,
        )
