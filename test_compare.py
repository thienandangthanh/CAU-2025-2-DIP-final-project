"""Test script for compare.py module.

This script tests the comparison and inference functionality including:
- Module imports
- Model loading for inference
- All enhancement methods (Zero-DCE, AutoContrast, Histogram Eq, CLAHE, Gamma)
- Comparison visualization
- Individual image saving
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

print("=" * 70)
print("Testing compare.py - Inference and Comparison Tool")
print("=" * 70)
print()

# Test 1: Module imports
print("1. Testing module imports...")
try:
    from compare import (
        load_model_for_inference,
        enhance_with_zero_dce,
        enhance_with_autocontrast,
        enhance_with_histogram_eq,
        enhance_with_clahe,
        enhance_with_gamma_correction,
        compare_methods,
        main
    )
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)
print()

# Test 2: Check for test image and weights
print("2. Checking test data availability...")
test_image_path = "./lol_dataset/eval15/low/1.png"
reference_image_path = "./lol_dataset/eval15/high/1.png"
weights_path = "./weights/zero_dce.weights.h5"

if not Path(test_image_path).exists():
    print(f"   ❌ Test image not found: {test_image_path}")
    print("   Please ensure LOL dataset is available")
    sys.exit(1)
print(f"   ✅ Test image found: {test_image_path}")

if Path(reference_image_path).exists():
    print(f"   ✅ Reference image found: {reference_image_path}")
    test_reference = True
else:
    print(f"   ⚠️  Warning: Reference image not found: {reference_image_path}")
    test_reference = False

if not Path(weights_path).exists():
    print(f"   ⚠️  Warning: Weights not found: {weights_path}")
    print("   Will test classical methods only")
    test_zero_dce = False
else:
    print(f"   ✅ Weights found: {weights_path}")
    test_zero_dce = True
print()

# Test 3: Load test image
print("3. Testing image loading...")
try:
    test_image = Image.open(test_image_path)
    print(f"   ✅ Image loaded: {test_image.size} pixels, mode: {test_image.mode}")
    if test_image.mode != "RGB":
        test_image = test_image.convert("RGB")
        print(f"   ✅ Converted to RGB")
except Exception as e:
    print(f"   ❌ Image loading failed: {e}")
    sys.exit(1)
print()

# Test 4: Classical enhancement methods
print("4. Testing classical enhancement methods...")

# Test AutoContrast
try:
    enhanced = enhance_with_autocontrast(test_image)
    print(f"   ✅ AutoContrast: {enhanced.size} pixels")
except Exception as e:
    print(f"   ❌ AutoContrast failed: {e}")
    sys.exit(1)

# Test Histogram Equalization
try:
    enhanced = enhance_with_histogram_eq(test_image)
    print(f"   ✅ Histogram Eq: {enhanced.size} pixels")
except Exception as e:
    print(f"   ❌ Histogram Eq failed: {e}")
    sys.exit(1)

# Test CLAHE
try:
    enhanced = enhance_with_clahe(test_image)
    print(f"   ✅ CLAHE: {enhanced.size} pixels")
except Exception as e:
    print(f"   ❌ CLAHE failed: {e}")
    sys.exit(1)

# Test Gamma Correction
try:
    enhanced = enhance_with_gamma_correction(test_image)
    print(f"   ✅ Gamma Correction: {enhanced.size} pixels")
except Exception as e:
    print(f"   ❌ Gamma Correction failed: {e}")
    sys.exit(1)

print()

# Test 5: Zero-DCE model loading and inference (if weights available)
if test_zero_dce:
    print("5. Testing Zero-DCE model loading and inference...")
    try:
        model = load_model_for_inference(weights_path)
        print("   ✅ Model loaded successfully")
        
        enhanced = enhance_with_zero_dce(test_image, model)
        print(f"   ✅ Zero-DCE enhancement: {enhanced.size} pixels")
        
        # Verify output is valid
        enhanced_array = np.array(enhanced)
        if enhanced_array.shape[:2] != np.array(test_image).shape[:2]:
            print(f"   ❌ Output shape mismatch")
            sys.exit(1)
        
        print(f"   ✅ Output value range: [{enhanced_array.min()}, {enhanced_array.max()}]")
    except Exception as e:
        print(f"   ❌ Zero-DCE test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()
else:
    print("5. Skipping Zero-DCE tests (weights not available)")
    print()

# Test 6: Compare methods function
print("6. Testing compare_methods function...")
try:
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "comparison.png")
    
    # Test with classical methods only (always available)
    methods_to_test = ["autocontrast", "histogram-eq", "clahe", "gamma"]
    if test_zero_dce:
        methods_to_test.insert(0, "zero-dce")
    
    compare_methods(
        input_path=test_image_path,
        weights_path=weights_path if test_zero_dce else None,
        output_path=output_path,
        methods=methods_to_test,
        save_individual=False
    )
    
    # Check output file exists
    if not os.path.exists(output_path):
        print(f"   ❌ Comparison output not created")
        sys.exit(1)
    
    # Check file size
    file_size = os.path.getsize(output_path)
    if file_size == 0:
        print(f"   ❌ Comparison output is empty")
        sys.exit(1)
    
    print(f"   ✅ Comparison created: {file_size} bytes")
    
    # Clean up
    shutil.rmtree(temp_dir)
    print("   ✅ Temporary files cleaned up")
except Exception as e:
    print(f"   ❌ compare_methods failed: {e}")
    import traceback
    traceback.print_exc()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    sys.exit(1)
print()

# Test 7: Reference image inclusion
print("7. Testing reference image inclusion...")
if test_reference:
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "comparison_with_ref.png")
        
        compare_methods(
            input_path=test_image_path,
            weights_path=weights_path if test_zero_dce else None,
            output_path=output_path,
            methods=["autocontrast", "gamma"],  # Test with fewer methods for speed
            save_individual=False,
            reference_path=reference_image_path
        )
        
        # Check output file exists
        if not os.path.exists(output_path):
            print(f"   ❌ Comparison with reference not created")
            sys.exit(1)
        
        # Check file size (should be larger with reference image)
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            print(f"   ❌ Comparison output is empty")
            sys.exit(1)
        
        print(f"   ✅ Comparison with reference created: {file_size} bytes")
        
        # Clean up
        shutil.rmtree(temp_dir)
        print("   ✅ Temporary files cleaned up")
    except Exception as e:
        print(f"   ❌ Reference image test failed: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        sys.exit(1)
else:
    print("   ⚠️  Skipping reference image test (no reference image available)")
print()

# Test 8: Individual image saving
print("8. Testing individual image saving...")
try:
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "comparison.png")
    
    compare_methods(
        input_path=test_image_path,
        weights_path=weights_path if test_zero_dce else None,
        output_path=output_path,
        methods=["autocontrast", "gamma"],  # Test with fewer methods for speed
        save_individual=True
    )
    
    # Check individual directory exists
    individual_dir = Path(temp_dir) / "individual"
    if not individual_dir.exists():
        print(f"   ❌ Individual directory not created")
        sys.exit(1)
    
    # Check individual images
    individual_images = list(individual_dir.glob("*.png"))
    if len(individual_images) == 0:
        print(f"   ❌ No individual images saved")
        sys.exit(1)
    
    print(f"   ✅ Individual directory created: {individual_dir}")
    print(f"   ✅ Individual images saved: {len(individual_images)} images")
    
    # Clean up
    shutil.rmtree(temp_dir)
    print("   ✅ Temporary files cleaned up")
except Exception as e:
    print(f"   ❌ Individual saving failed: {e}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    sys.exit(1)
print()

# Test 9: CLI main function (import check)
print("9. Testing CLI main function...")
try:
    from compare import main
    
    if not callable(main):
        print("   ❌ main() is not callable")
        sys.exit(1)
    
    print("   ✅ CLI main function is callable")
    print("   ✅ Use 'python compare.py --help' to see all options")
except Exception as e:
    print(f"   ❌ CLI test failed: {e}")
    sys.exit(1)
print()

# Summary
print("=" * 70)
print("✅ All compare.py tests passed!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Module imports work correctly")
print("  ✅ Image loading successful")
print("  ✅ All classical methods work (AutoContrast, Histogram Eq, CLAHE, Gamma)")
if test_zero_dce:
    print("  ✅ Zero-DCE model loading and inference work")
else:
    print("  ⚠️  Zero-DCE tests skipped (no weights)")
print("  ✅ Comparison visualization works")
if test_reference:
    print("  ✅ Reference image inclusion works")
else:
    print("  ⚠️  Reference image tests skipped (no reference)")
print("  ✅ Individual image saving works")
print("  ✅ CLI is properly configured")
print()
print("Example usage:")
if test_zero_dce:
    print(f"  python compare.py -i {test_image_path} -w {weights_path} -o output.png")
    if test_reference:
        print(f"  python compare.py -i {test_image_path} -w {weights_path} -r {reference_image_path} -o output.png")
else:
    print("  python compare.py -i <input> -w <weights> -o output.png")
print("  python compare.py --help")
print()
