"""Loss functions for Zero-DCE model.

This module implements the four unsupervised loss functions used to train
the Zero-DCE model without requiring paired training data. These losses
guide the network to produce well-exposed, color-correct, and spatially
coherent enhanced images.
"""

import tensorflow as tf
import keras


def color_constancy_loss(x: tf.Tensor) -> tf.Tensor:
    """Compute color constancy loss.
    
    Measures the deviation between average values of RGB channels to correct
    potential color shifts in enhanced images. This loss encourages the model
    to maintain color balance by penalizing differences between the mean
    values of the R, G, and B channels.
    
    Args:
        x: Enhanced image tensor of shape (batch, height, width, 3)
    
    Returns:
        Color constancy loss value (scalar tensor)
    """
    mean_rgb = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
    mr, mg, mb = (
        mean_rgb[:, :, :, 0],
        mean_rgb[:, :, :, 1],
        mean_rgb[:, :, :, 2],
    )
    d_rg = tf.square(mr - mg)
    d_rb = tf.square(mr - mb)
    d_gb = tf.square(mb - mg)
    return tf.sqrt(tf.square(d_rg) + tf.square(d_rb) + tf.square(d_gb))


def exposure_loss(x: tf.Tensor, mean_val: float = 0.6) -> tf.Tensor:
    """Compute exposure control loss.
    
    Measures the distance between the average intensity value of local regions
    and a preset well-exposedness level. This loss restrains under-exposed and
    over-exposed regions by encouraging local image patches to have an average
    intensity near the target value.
    
    Args:
        x: Enhanced image tensor of shape (batch, height, width, 3)
        mean_val: Target exposure level (default: 0.6, a well-exposed value)
    
    Returns:
        Exposure control loss value (scalar tensor)
    """
    x = tf.reduce_mean(x, axis=3, keepdims=True)
    mean = tf.nn.avg_pool2d(x, ksize=16, strides=16, padding="VALID")
    return tf.reduce_mean(tf.square(mean - mean_val))


def illumination_smoothness_loss(x: tf.Tensor) -> tf.Tensor:
    """Compute illumination smoothness loss.
    
    Preserves the monotonicity relations between neighboring pixels by
    minimizing the total variation in curve parameter maps. This loss
    encourages smooth transitions in the learned curves, preventing
    abrupt changes that could introduce artifacts.
    
    Args:
        x: Curve parameter maps of shape (batch, height, width, channels)
           Typically has 24 channels (8 iterations × 3 RGB channels)
    
    Returns:
        Illumination smoothness loss value (scalar tensor)
    """
    batch_size = tf.shape(x)[0]
    h_x = tf.shape(x)[1]
    w_x = tf.shape(x)[2]
    count_h = (tf.shape(x)[2] - 1) * tf.shape(x)[3]
    count_w = tf.shape(x)[2] * (tf.shape(x)[3] - 1)
    h_tv = tf.reduce_sum(tf.square((x[:, 1:, :, :] - x[:, : h_x - 1, :, :])))
    w_tv = tf.reduce_sum(tf.square((x[:, :, 1:, :] - x[:, :, : w_x - 1, :])))
    batch_size = tf.cast(batch_size, dtype=tf.float32)
    count_h = tf.cast(count_h, dtype=tf.float32)
    count_w = tf.cast(count_w, dtype=tf.float32)
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class SpatialConsistencyLoss(keras.losses.Loss):
    """Spatial consistency loss.
    
    Encourages spatial coherence of the enhanced image by preserving the
    contrast between neighboring regions across the input image and its
    enhanced version. This loss computes the difference between neighboring
    pixels in both the original and enhanced images, then penalizes
    inconsistencies in these differences.
    
    The loss uses four directional kernels (left, right, up, down) to compute
    local differences, ensuring that the enhancement maintains the spatial
    structure of the original image.
    """
    
    def __init__(self, **kwargs):
        """Initialize the spatial consistency loss.
        
        Creates four convolutional kernels for computing differences between
        neighboring pixels in four directions (left, right, up, down).
        """
        super().__init__(reduction="none")

        # Define kernels for computing differences with neighbors
        self.left_kernel = tf.constant(
            [[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.right_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.up_kernel = tf.constant(
            [[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.down_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=tf.float32
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute spatial consistency loss between original and enhanced images.
        
        Args:
            y_true: Original low-light image of shape (batch, height, width, 3)
            y_pred: Enhanced image of shape (batch, height, width, 3)
        
        Returns:
            Spatial consistency loss tensor of shape (batch, height/4, width/4, 1)
            Note: Returns per-pixel loss (not reduced) as reduction="none"
        """
        # Convert to grayscale by averaging channels
        original_mean = tf.reduce_mean(y_true, 3, keepdims=True)
        enhanced_mean = tf.reduce_mean(y_pred, 3, keepdims=True)
        
        # Downsample to reduce computational cost
        original_pool = tf.nn.avg_pool2d(
            original_mean, ksize=4, strides=4, padding="VALID"
        )
        enhanced_pool = tf.nn.avg_pool2d(
            enhanced_mean, ksize=4, strides=4, padding="VALID"
        )

        # Compute differences in four directions for original image
        d_original_left = tf.nn.conv2d(
            original_pool,
            self.left_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_original_right = tf.nn.conv2d(
            original_pool,
            self.right_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_original_up = tf.nn.conv2d(
            original_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_original_down = tf.nn.conv2d(
            original_pool,
            self.down_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        # Compute differences in four directions for enhanced image
        d_enhanced_left = tf.nn.conv2d(
            enhanced_pool,
            self.left_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_enhanced_right = tf.nn.conv2d(
            enhanced_pool,
            self.right_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_enhanced_up = tf.nn.conv2d(
            enhanced_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_enhanced_down = tf.nn.conv2d(
            enhanced_pool,
            self.down_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        # Compute squared differences between original and enhanced differences
        d_left = tf.square(d_original_left - d_enhanced_left)
        d_right = tf.square(d_original_right - d_enhanced_right)
        d_up = tf.square(d_original_up - d_enhanced_up)
        d_down = tf.square(d_original_down - d_enhanced_down)
        
        return d_left + d_right + d_up + d_down
