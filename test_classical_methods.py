"""Test script for classical_methods.py module.

This script tests all classical enhancement methods including:
- AutoContrast
- Histogram Equalization
- CLAHE
- Gamma Correction
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

print("=" * 70)
print("Testing classical_methods.py - Classical Enhancement Methods")
print("=" * 70)
print()

# Test 1: Module imports
print("1. Testing module imports...")
try:
    from classical_methods import (
        CLASSICAL_METHODS,
        enhance_with_autocontrast,
        enhance_with_clahe,
        enhance_with_gamma_correction,
        enhance_with_histogram_eq,
        get_available_methods,
        get_method_info,
    )

    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)
print()

# Test 2: Check CLASSICAL_METHODS dictionary
print("2. Testing CLASSICAL_METHODS dictionary...")
try:
    expected_methods = ["autocontrast", "histogram-eq", "clahe", "gamma"]
    available = get_available_methods()

    if set(available) != set(expected_methods):
        print(f"   ❌ Method mismatch. Expected: {expected_methods}, Got: {available}")
        sys.exit(1)

    print(f"   ✅ All {len(available)} methods available: {available}")

    # Check each method has required keys
    for method_key in available:
        info = get_method_info(method_key)
        if not all(key in info for key in ["name", "function", "description"]):
            print(f"   ❌ Method '{method_key}' missing required keys")
            sys.exit(1)
        print(f"   ✅ {method_key}: {info['name']} - {info['description']}")

except Exception as e:
    print(f"   ❌ Dictionary test failed: {e}")
    sys.exit(1)
print()

# Test 3: Create test image
print("3. Creating test image...")
try:
    # Create a simple gradient image for testing
    test_image_array = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        test_image_array[i, :, :] = i  # Horizontal gradient
    test_image = Image.fromarray(test_image_array)

    print(
        f"   ✅ Test image created: {test_image.size} pixels, mode: {test_image.mode}"
    )
except Exception as e:
    print(f"   ❌ Test image creation failed: {e}")
    sys.exit(1)
print()

# Test 4: Test AutoContrast
print("4. Testing enhance_with_autocontrast()...")
try:
    enhanced = enhance_with_autocontrast(test_image)

    if enhanced.size != test_image.size:
        print(f"   ❌ Size mismatch: {enhanced.size} vs {test_image.size}")
        sys.exit(1)

    if enhanced.mode != "RGB":
        print(f"   ❌ Mode incorrect: {enhanced.mode}")
        sys.exit(1)

    enhanced_array = np.array(enhanced)
    print(f"   ✅ Output size: {enhanced.size}")
    print(f"   ✅ Value range: [{enhanced_array.min()}, {enhanced_array.max()}]")
except Exception as e:
    print(f"   ❌ AutoContrast failed: {e}")
    sys.exit(1)
print()

# Test 5: Test Histogram Equalization
print("5. Testing enhance_with_histogram_eq()...")
try:
    enhanced = enhance_with_histogram_eq(test_image)

    if enhanced.size != test_image.size:
        print("   ❌ Size mismatch")
        sys.exit(1)

    enhanced_array = np.array(enhanced)
    print(f"   ✅ Output size: {enhanced.size}")
    print(f"   ✅ Value range: [{enhanced_array.min()}, {enhanced_array.max()}]")
except Exception as e:
    print(f"   ❌ Histogram Eq failed: {e}")
    sys.exit(1)
print()

# Test 6: Test CLAHE
print("6. Testing enhance_with_clahe()...")
try:
    # Test with default parameters
    enhanced = enhance_with_clahe(test_image)

    if enhanced.size != test_image.size:
        print("   ❌ Size mismatch")
        sys.exit(1)

    enhanced_array = np.array(enhanced)
    print(f"   ✅ Output size: {enhanced.size}")
    print(f"   ✅ Value range: [{enhanced_array.min()}, {enhanced_array.max()}]")

    # Test with custom parameters
    enhanced_custom = enhance_with_clahe(test_image, clip_limit=3.0, tile_size=16)
    print("   ✅ Custom parameters work: clip_limit=3.0, tile_size=16")
except Exception as e:
    print(f"   ❌ CLAHE failed: {e}")
    sys.exit(1)
print()

# Test 7: Test Gamma Correction
print("7. Testing enhance_with_gamma_correction()...")
try:
    # Test with default gamma
    enhanced = enhance_with_gamma_correction(test_image)

    if enhanced.size != test_image.size:
        print("   ❌ Size mismatch")
        sys.exit(1)

    enhanced_array = np.array(enhanced)
    print(f"   ✅ Output size: {enhanced.size}")
    print(f"   ✅ Value range: [{enhanced_array.min()}, {enhanced_array.max()}]")

    # Test with custom gamma
    enhanced_custom = enhance_with_gamma_correction(test_image, gamma=1.5)
    print("   ✅ Custom gamma works: gamma=1.5")
except Exception as e:
    print(f"   ❌ Gamma Correction failed: {e}")
    sys.exit(1)
print()

# Test 8: Test with real image (if available)
print("8. Testing with real image from LOL dataset...")
real_image_path = "./lol_dataset/eval15/low/1.png"
if Path(real_image_path).exists():
    try:
        real_image = Image.open(real_image_path)
        print(f"   ✅ Real image loaded: {real_image.size}")

        # Test all methods on real image
        for method_key in get_available_methods():
            info = get_method_info(method_key)
            enhanced = info["function"](real_image)

            if enhanced.size != real_image.size:
                print(f"   ❌ {info['name']}: Size mismatch")
                sys.exit(1)

        print("   ✅ All methods work on real image")
    except Exception as e:
        print(f"   ❌ Real image test failed: {e}")
        sys.exit(1)
else:
    print("   ⚠️  Skipping (real image not available)")
print()

# Test 9: Test via CLASSICAL_METHODS dictionary
print("9. Testing method invocation via CLASSICAL_METHODS...")
try:
    for _method_key, method_info in CLASSICAL_METHODS.items():
        enhanced = method_info["function"](test_image)

        if enhanced.size != test_image.size:
            print(f"   ❌ {method_info['name']}: Size mismatch")
            sys.exit(1)

    print("   ✅ All methods callable via CLASSICAL_METHODS dictionary")
except Exception as e:
    print(f"   ❌ Dictionary invocation failed: {e}")
    sys.exit(1)
print()

# Summary
print("=" * 70)
print("✅ All classical_methods.py tests passed!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Module imports work correctly")
print("  ✅ CLASSICAL_METHODS dictionary properly structured")
print("  ✅ get_available_methods() returns correct list")
print("  ✅ get_method_info() provides complete information")
print("  ✅ AutoContrast works correctly")
print("  ✅ Histogram Equalization works correctly")
print("  ✅ CLAHE works correctly (with custom parameters)")
print("  ✅ Gamma Correction works correctly (with custom parameters)")
if Path(real_image_path).exists():
    print("  ✅ All methods work on real images")
print("  ✅ Methods callable via CLASSICAL_METHODS dictionary")
print()
print("Available methods:")
for method_key in get_available_methods():
    info = get_method_info(method_key)
    print(f"  - {method_key}: {info['name']}")
print()
