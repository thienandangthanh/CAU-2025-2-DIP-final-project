"""Test script for train.py module.

This script tests the training functionality including:
- Module imports
- Dataset loading integration
- Model compilation
- plot_training_history function
- Training for one epoch
- Weight saving and loading
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import sys
import tempfile
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Import modules from train.py
from dataset import get_dataset
from model import ZeroDCE
from train import plot_training_history

print("=" * 70)
print("Testing train.py - Training Script")
print("=" * 70)
print()

# Test 1: Module imports
print("1. Testing module imports...")
try:
    from dataset import get_dataset
    from model import ZeroDCE
    from train import plot_training_history, main
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)
print()

# Test 2: Dataset loading
print("2. Testing dataset loading...")
try:
    train_dataset, val_dataset, test_paths = get_dataset(
        dataset_path="./lol_dataset",
        max_train_images=400,
        batch_size=4,  # Small batch for testing
        image_size=256
    )
    print(f"   ✅ Train dataset: {train_dataset}")
    print(f"   ✅ Val dataset: {val_dataset}")
    print(f"   ✅ Test paths: {len(test_paths)} images")
except Exception as e:
    print(f"   ❌ Dataset loading failed: {e}")
    sys.exit(1)
print()

# Test 3: Model creation and compilation
print("3. Testing model creation and compilation...")
try:
    model = ZeroDCE()
    model.compile(learning_rate=1e-4)
    print("   ✅ Model created and compiled")
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    sys.exit(1)
print()

# Test 4: Training for one epoch
print("4. Testing training for 1 epoch...")
print("   (This will take ~10 seconds)")
try:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        verbose=0  # Suppress training output
    )
    
    # Check that all metrics are present
    expected_metrics = [
        "total_loss",
        "illumination_smoothness_loss",
        "spatial_constancy_loss",
        "color_constancy_loss",
        "exposure_loss",
        "val_total_loss",
        "val_illumination_smoothness_loss",
        "val_spatial_constancy_loss",
        "val_color_constancy_loss",
        "val_exposure_loss"
    ]
    
    for metric in expected_metrics:
        if metric not in history.history:
            print(f"   ❌ Missing metric: {metric}")
            sys.exit(1)
    
    print("   ✅ Training completed successfully")
    print(f"   ✅ Final train total_loss: {history.history['total_loss'][-1]:.4f}")
    print(f"   ✅ Final val total_loss: {history.history['val_total_loss'][-1]:.4f}")
except Exception as e:
    print(f"   ❌ Training failed: {e}")
    sys.exit(1)
print()

# Test 5: Weight saving and loading
print("5. Testing weight saving and loading...")
try:
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    weight_path = os.path.join(temp_dir, "test_weights.weights.h5")
    
    # Save weights
    model.save_weights(weight_path)
    print(f"   ✅ Weights saved to {weight_path}")
    
    # Load weights into a new model
    new_model = ZeroDCE()
    new_model.compile(learning_rate=1e-4)
    new_model.load_weights(weight_path)
    print("   ✅ Weights loaded successfully")
    
    # Clean up
    shutil.rmtree(temp_dir)
    print("   ✅ Temporary files cleaned up")
except Exception as e:
    print(f"   ❌ Weight save/load failed: {e}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    sys.exit(1)
print()

# Test 6: Plot training history
print("6. Testing plot_training_history function...")
try:
    # Create temporary directory for plots
    temp_plot_dir = tempfile.mkdtemp()
    
    # Generate plots
    plot_training_history(history, save_dir=temp_plot_dir)
    
    # Check that all plot files were created
    expected_plots = [
        "total_loss.png",
        "illumination_smoothness_loss.png",
        "spatial_constancy_loss.png",
        "color_constancy_loss.png",
        "exposure_loss.png"
    ]
    
    for plot_name in expected_plots:
        plot_path = os.path.join(temp_plot_dir, plot_name)
        if not os.path.exists(plot_path):
            print(f"   ❌ Missing plot: {plot_name}")
            shutil.rmtree(temp_plot_dir)
            sys.exit(1)
        # Check file size (should be > 0)
        if os.path.getsize(plot_path) == 0:
            print(f"   ❌ Empty plot file: {plot_name}")
            shutil.rmtree(temp_plot_dir)
            sys.exit(1)
    
    print("   ✅ All 5 training plots generated successfully")
    
    # Clean up
    shutil.rmtree(temp_plot_dir)
    print("   ✅ Temporary plot files cleaned up")
except Exception as e:
    print(f"   ❌ Plot generation failed: {e}")
    if os.path.exists(temp_plot_dir):
        shutil.rmtree(temp_plot_dir)
    sys.exit(1)
print()

# Test 7: CLI argument parsing (import check)
print("7. Testing CLI argument parsing...")
try:
    import argparse
    from train import main
    
    # Verify main function exists and is callable
    if not callable(main):
        print("   ❌ main() is not callable")
        sys.exit(1)
    
    print("   ✅ CLI main function is callable")
    print("   ✅ Use 'python train.py --help' to see all options")
except Exception as e:
    print(f"   ❌ CLI test failed: {e}")
    sys.exit(1)
print()

# Summary
print("=" * 70)
print("✅ All train.py tests passed!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Module imports work correctly")
print("  ✅ Dataset integration successful")
print("  ✅ Model compilation works")
print("  ✅ Training completes successfully")
print("  ✅ Weight saving and loading work")
print("  ✅ Plot generation works")
print("  ✅ CLI is properly configured")
print()
print("Next steps:")
print("  - Run full training: python train.py --epochs 100")
print("  - Customize training: python train.py --help")
print()
