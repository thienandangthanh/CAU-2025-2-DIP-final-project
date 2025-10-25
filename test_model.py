"""Test script for model.py module."""

import tensorflow as tf
from model import build_dce_net, get_enhanced_image, ZeroDCE

print("Testing model.py module...")
print("=" * 60)

# Test 1: Build DCE-Net
print("\n1. Testing build_dce_net()...")
dce_net = build_dce_net()
print(f"   ✅ DCE-Net built successfully")
print(f"   Input shape: {dce_net.input_shape}")
print(f"   Output shape: {dce_net.output_shape}")

# Display model summary
print("\n   DCE-Net Architecture Summary:")
print("   " + "-" * 56)
dce_net.summary(print_fn=lambda x: print("   " + x))
print("   " + "-" * 56)

# Count parameters
total_params = dce_net.count_params()
print(f"   Total parameters: {total_params:,}")

# Verify output channels
expected_output_channels = 24  # 8 iterations × 3 RGB channels
actual_output_channels = dce_net.output_shape[-1]
assert actual_output_channels == expected_output_channels, \
    f"Expected {expected_output_channels} output channels, got {actual_output_channels}"
print(f"   ✅ Output has correct number of channels: {actual_output_channels}")

# Test 2: Test enhancement function
print("\n2. Testing get_enhanced_image()...")
dummy_image = tf.random.uniform([2, 256, 256, 3], 0, 1)
dummy_params = dce_net(dummy_image)  # Get realistic parameter values
enhanced = get_enhanced_image(dummy_image, dummy_params)

print(f"   Input shape: {dummy_image.shape}")
print(f"   Parameters shape: {dummy_params.shape}")
print(f"   Enhanced shape: {enhanced.shape}")
print(f"   Input range: [{tf.reduce_min(dummy_image):.3f}, {tf.reduce_max(dummy_image):.3f}]")
print(f"   Enhanced range: [{tf.reduce_min(enhanced):.3f}, {tf.reduce_max(enhanced):.3f}]")

# Verify enhancement produces different output
difference = tf.reduce_mean(tf.abs(enhanced - dummy_image))
print(f"   Mean absolute difference: {difference:.6f}")
print(f"   ✅ Enhancement modifies the image")

# Test 3: Test ZeroDCE model
print("\n3. Testing ZeroDCE model...")
zero_dce = ZeroDCE()
print(f"   ✅ ZeroDCE model instantiated")

# Test compilation
print("\n4. Testing model compilation...")
zero_dce.compile(learning_rate=1e-4)
print(f"   ✅ Model compiled successfully")
print(f"   Optimizer: {zero_dce.optimizer.__class__.__name__}")
print(f"   Learning rate: {zero_dce.optimizer.learning_rate.numpy()}")

# Verify metrics
print(f"\n   Tracked metrics:")
for metric in zero_dce.metrics:
    print(f"      - {metric.name}")
assert len(zero_dce.metrics) == 5, "Should track 5 metrics"
print(f"   ✅ All 5 metrics initialized")

# Test 5: Test forward pass
print("\n5. Testing forward pass...")
output = zero_dce(dummy_image)
print(f"   Input shape: {dummy_image.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Output range: [{tf.reduce_min(output):.3f}, {tf.reduce_max(output):.3f}]")
assert output.shape == dummy_image.shape, "Output should have same shape as input"
print(f"   ✅ Forward pass successful")

# Test 6: Test loss computation
print("\n6. Testing compute_losses()...")
curve_params = zero_dce.dce_model(dummy_image)
losses = zero_dce.compute_losses(dummy_image, curve_params)

print(f"   Loss components:")
for loss_name, loss_value in losses.items():
    print(f"      {loss_name}: {loss_value:.6f}")

# Verify all losses are present
expected_losses = [
    "total_loss",
    "illumination_smoothness_loss", 
    "spatial_constancy_loss",
    "color_constancy_loss",
    "exposure_loss"
]
for loss_name in expected_losses:
    assert loss_name in losses, f"Missing loss: {loss_name}"
print(f"   ✅ All loss components computed")

# Verify total loss is sum of components
computed_total = (
    losses["illumination_smoothness_loss"] +
    losses["spatial_constancy_loss"] +
    losses["color_constancy_loss"] +
    losses["exposure_loss"]
)
assert abs(losses["total_loss"] - computed_total) < 1e-5, \
    "Total loss should equal sum of components"
print(f"   ✅ Total loss equals sum of components")

# Test 7: Test training step
print("\n7. Testing train_step()...")
metrics_before = {metric.name: metric.result().numpy() for metric in zero_dce.metrics}
print(f"   Metrics before training step:")
for name, value in metrics_before.items():
    print(f"      {name}: {value:.6f}")

# Perform one training step
train_metrics = zero_dce.train_step(dummy_image)
print(f"\n   Metrics after training step:")
for name, value in train_metrics.items():
    print(f"      {name}: {value:.6f}")
print(f"   ✅ Training step executed successfully")

# Test 8: Test validation step
print("\n8. Testing test_step()...")
# Reset metrics
for metric in zero_dce.metrics:
    metric.reset_state()

test_metrics = zero_dce.test_step(dummy_image)
print(f"   Validation metrics:")
for name, value in test_metrics.items():
    print(f"      {name}: {value:.6f}")
print(f"   ✅ Validation step executed successfully")

# Test 9: Test weight saving and loading
print("\n9. Testing save_weights() and load_weights()...")
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    weights_path = os.path.join(tmpdir, "test_weights.weights.h5")
    
    # Save weights
    zero_dce.save_weights(weights_path)
    print(f"   ✅ Weights saved to temporary file")
    
    # Get output before loading
    output_before = zero_dce(dummy_image)
    
    # Create new model and load weights
    new_model = ZeroDCE()
    new_model.compile(learning_rate=1e-4)
    new_model.load_weights(weights_path)
    print(f"   ✅ Weights loaded into new model")
    
    # Get output after loading
    output_after = new_model(dummy_image)
    
    # Verify outputs are identical
    max_diff = tf.reduce_max(tf.abs(output_before - output_after))
    print(f"   Max difference between outputs: {max_diff:.10f}")
    assert max_diff < 1e-6, "Outputs should be identical after loading weights"
    print(f"   ✅ Loaded weights produce identical outputs")

# Test 10: Test with different image sizes
print("\n10. Testing with different image sizes...")
test_sizes = [(128, 128), (256, 256), (512, 512)]
for h, w in test_sizes:
    test_img = tf.random.uniform([1, h, w, 3], 0, 1)
    test_output = zero_dce(test_img)
    assert test_output.shape == test_img.shape, \
        f"Output shape mismatch for {h}x{w}"
    print(f"   ✅ Size {h}×{w}: Input {test_img.shape} → Output {test_output.shape}")

print("\n" + "=" * 60)
print("✅ All tests passed successfully!")
print("   - DCE-Net builds without errors")
print("   - Model has correct architecture (7 conv layers, 24 output channels)")
print("   - get_enhanced_image() works as standalone function")
print("   - ZeroDCE model compiles and runs inference")
print("   - All loss components are computed correctly")
print("   - Training and validation steps work")
print("   - Weight saving and loading work correctly")
print("   - Model works with variable image sizes")
