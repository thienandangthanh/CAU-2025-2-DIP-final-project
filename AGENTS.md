# AGENTS.md

> Context and instructions for AI coding agents working on this project

## Project Overview

**Zero-DCE (Zero-Reference Deep Curve Estimation)** is a deep learning approach for low-light image enhancement that doesn't require paired training data. This is a re-implementation in Keras 3 for a graduate-level Digital Image Processing course final project.

### Key Features
- **Zero-reference learning**: No need for paired low-light/normal-light images
- **Lightweight architecture**: DCE-Net has only 7 convolutional layers
- **Curve-based enhancement**: Learns pixel-wise tonal curves for dynamic range adjustment
- **Unsupervised training**: Uses 4 carefully designed loss functions

## Project Status

**Current State:** In active refactoring from monolithic script to modular architecture

- ✅ Original `zero_dce.py` contains complete working implementation (~537 lines)
- 🔄 Refactoring into modular components (in progress)
- 📁 Empty module files created: `dataset.py`, `loss.py`, `model.py`, `train.py`, `compare.py`

**See:** `REFACTOR_PLAN.md` for detailed refactoring roadmap

## Architecture

### Zero-DCE Framework
```
Input Image → DCE-Net → Curve Parameters → Iterative Enhancement → Output Image
                ↓
        Unsupervised Losses
        (guide training)
```

### DCE-Net Architecture
7-layer CNN with symmetrical skip connections:
- Input: Low-light RGB image (H×W×3)
- Conv layers 1-4: 32 filters, 3×3, ReLU
- Skip connections: Concatenate layers 4↔3, 5↔2, 6↔1
- Output: 24 parameter maps (8 iterations × 3 RGB channels), Tanh activation

### Loss Functions (Unsupervised)
1. **Color Constancy Loss** (weight: 5): Corrects color deviations
2. **Exposure Loss** (weight: 10): Prevents under/over-exposure
3. **Illumination Smoothness Loss** (weight: 200): Preserves monotonicity
4. **Spatial Consistency Loss** (weight: 1): Maintains spatial coherence

### Enhancement Process
Iteratively applies 8 curve adjustments:
```
x_{n+1} = x_n + r_n * (x_n^2 - x_n)
```
where `r_n` are learned curve parameters

## Module Structure

### Current Refactoring Target

```
zero-dce-keras/
├── dataset.py          # Dataset loading and TensorFlow data pipelines
├── loss.py            # Four unsupervised loss functions
├── model.py           # DCE-Net architecture + ZeroDCE training model
├── train.py           # Training script with CLI arguments
├── compare.py         # Inference and comparison with classical methods
├── gui/               # GUI application (PyQt6)
│   ├── __init__.py
│   ├── main_window.py
│   ├── widgets/       # Custom UI widgets
│   ├── dialogs/       # Dialog windows
│   ├── utils/         # GUI utilities
│   │   ├── model_loader.py
│   │   └── settings.py
│   └── resources/     # Icons, images, stylesheets
├── tests/             # All test modules (pytest)
│   ├── __init__.py
│   ├── test_gui_utils_model_loader.py
│   └── test_*.py      # More test modules
├── zero_dce.py        # Original monolithic implementation (reference)
├── lol_dataset/       # LOL Dataset (485 train + 15 test images)
│   ├── our485/low/    # Training low-light images
│   ├── our485/high/   # Training normal-light images (not used in training)
│   └── eval15/low/    # Test low-light images
├── weights/           # Trained model weights
├── pytest.ini         # Pytest configuration
├── AGENTS.md          # This file
├── REFACTOR_PLAN.md   # Detailed refactoring guide
└── README.md          # Project documentation
```

### Module Responsibilities

#### `dataset.py`
- Load and preprocess images from LOL Dataset
- Create TensorFlow datasets for training/validation/testing
- Configurable paths, batch size, image size
- **Key functions:** `load_data()`, `data_generator()`, `get_dataset()`

#### `loss.py`
- All loss function implementations
- Pure functions (stateless) except `SpatialConsistencyLoss` class
- **Exports:** `color_constancy_loss()`, `exposure_loss()`, `illumination_smoothness_loss()`, `SpatialConsistencyLoss`

#### `model.py`
- DCE-Net architecture builder
- Enhancement function (applies curves)
- ZeroDCE training wrapper model
- **Key components:** `build_dce_net()`, `get_enhanced_image()`, `ZeroDCE` class

#### `train.py`
- CLI-based training script
- Uses argparse for configuration
- Integrates dataset, model, and losses
- Saves weights and plots training curves
- **Usage:** `python train.py --epochs 100 --batch-size 16`

#### `compare.py`
- Inference on single images
- Comparison with classical methods (AutoContrast, Histogram Eq, CLAHE, Gamma Correction)
- CLI for easy usage
- **Usage:** `python compare.py -i input.jpg -w weights.h5 -o output.jpg`

## Development Guidelines

### Code Style
- Follow PEP 8
- Use type hints where helpful
- Docstrings for all public functions/classes (Google style)
- Keep functions focused (single responsibility)

### Key Principles
1. **Modularity**: Each file has a single, clear purpose
2. **Reusability**: Functions should be usable independently
3. **Configurability**: Use arguments/configs instead of hardcoded values
4. **Maintainability**: Clear names, comments for complex logic
5. **Academic clarity**: Code should be educational (this is a course project)

### Import Structure
```python
# Standard library
import os
from typing import Tuple, Optional

# Third-party
import numpy as np
import tensorflow as tf
import keras
from keras import layers

# Local modules
from dataset import get_dataset
from loss import color_constancy_loss, SpatialConsistencyLoss
from model import build_dce_net, ZeroDCE
```

### Critical Implementation Details

#### Image Preprocessing
- Input images: Resize to 256×256 (configurable)
- Normalization: Divide by 255.0 to get [0, 1] range
- No data augmentation (as per original paper)

#### Training Configuration
- Default learning rate: 1e-4 (Adam optimizer)
- Batch size: 16 (adjust based on GPU memory)
- Epochs: 100 (original paper)
- Train/val split: 400/85 images from LOL Dataset

#### Loss Weights (Critical - Don't Change Without Testing)
```python
loss_illumination = 200 * illumination_smoothness_loss(output)
loss_spatial_constancy = 1 * spatial_constancy_loss(enhanced, original)
loss_color_constancy = 5 * color_constancy_loss(enhanced)
loss_exposure = 10 * exposure_loss(enhanced)
```

#### Model Weights
- Only save/load `dce_model` weights (not the wrapper)
- Use `.h5` format for compatibility
- Default path: `./weights/zero_dce.weights.h5`

## Common Tasks

**Note:** All commands below assume the virtual environment is activated. For agents, prepend activation command (e.g., `source .venv/bin/activate &&` for bash shell).

### Running Training
```bash
# Basic training (assuming venv is activated)
python train.py

# Custom configuration
python train.py --epochs 100 --batch-size 16 --learning-rate 1e-4 \
                --dataset-path ./lol_dataset --save-path ./weights/model.h5

# For agents (example with bash shell):
source .venv/bin/activate && python train.py --epochs 100
```

### Running Inference
```bash
# Enhance single image and compare methods
python compare.py -i ./test_image.jpg -w ./weights/zero_dce.weights.h5 -o ./enhanced.jpg

# Compare specific methods
python compare.py -i input.jpg -w weights.h5 --methods zero-dce autocontrast clahe

# For agents (example with bash shell):
source .venv/bin/activate && python compare.py -i input.jpg -w weights.h5
```

### Running Tests

**Test Location:** All tests are in the `tests/` directory at the project root.

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_gui_utils_model_loader.py

# Run with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=gui --cov-report=html

# For agents (example with bash shell):
source .venv/bin/activate && pytest -v
source .venv/bin/activate && pytest tests/test_gui_utils_model_loader.py
```

**Available Test Markers (defined in pytest.ini):**
```bash
# Run only unit tests (fast, no external dependencies)
pytest -m unit

# Run only integration tests (may require model weights)
pytest -m integration

# Run GUI-related tests
pytest -m gui

# Skip slow tests
pytest -m "not slow"
```

## Refactoring Workflow

**Each refactoring step should:**
1. Be implemented completely before moving to the next
2. Be tested to ensure functionality
3. Be committed to git manually by the student
4. Follow the order in `REFACTOR_PLAN.md`

**Testing after each step:**
- Import the module successfully
- Run basic functionality tests
- Ensure no breaking changes to existing code

## Dependencies

### Required Packages
```toml
[project]
dependencies = [
    "tensorflow>=2.15.0",  # or tensorflow-gpu
    "keras>=3.0.0",
    "numpy>=1.24.0",
    "pillow>=10.0.0",
    "matplotlib>=3.7.0",
    "opencv-python>=4.8.0",  # for comparison methods
]
```

### Environment Setup

#### Initial Setup (After Cloning Repository)

**Step 1: Create Virtual Environment**
```bash
uv venv
```

This creates a `.venv/` directory in the project root.

**Step 2: Activate Virtual Environment**

The activation command depends on your operating system and shell:

**Linux/macOS:**
- **Bash/Zsh:** `source .venv/bin/activate`
- **Fish:** `source .venv/bin/activate.fish`

**Windows:**
- **Command Prompt:** `.venv\Scripts\activate`
- **PowerShell:** `.venv\Scripts\Activate.ps1`

**Step 3: Install Dependencies**
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

#### For AI Agents: Shell Detection and Virtual Environment Activation

**CRITICAL:** Before running any Python scripts or commands, agents MUST:

1. **Detect the current operating system** from the user info context
2. **Activate the virtual environment** using the appropriate command
3. **Run Python commands within the activated environment**

**Default Shell by Operating System:**
- **Linux:** Bash shell (`source .venv/bin/activate`)
- **macOS:** Zsh shell (`source .venv/bin/activate`)
- **Windows:** Command Prompt (`.venv\Scripts\activate`)

**Shell Detection Examples:**
- OS: Linux → Use `source .venv/bin/activate` (bash is default)
- User shell: `/usr/bin/fish` → Use `source .venv/bin/activate.fish` (if explicitly fish)
- OS: Windows → Use `.venv\Scripts\activate`

**Correct Workflow for Running Python:**
```bash
# Linux (default bash shell)
source .venv/bin/activate && python train.py --epochs 100

# macOS (default zsh shell)
source .venv/bin/activate && python train.py --epochs 100

# Linux with fish shell (if user explicitly uses fish)
source .venv/bin/activate.fish && python train.py --epochs 100

# Windows Command Prompt
.venv\Scripts\activate && python train.py --epochs 100
```

**NEVER run Python commands directly without activating the virtual environment first.**

## Dataset Information

### LOL Dataset Structure
- **Training set:** 485 image pairs in `our485/`
  - Low-light images: `our485/low/*.png`
  - Well-exposed images: `our485/high/*.png` (not used during training!)
- **Test set:** 15 image pairs in `eval15/`
  - Low-light images: `eval15/low/*.png`

### Download Instructions
```bash
wget https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip
unzip -q lol_dataset.zip && rm lol_dataset.zip
```

## Testing Guidelines

### Test Organization

**CRITICAL:** This project follows pytest best practices for test organization.

#### Test Directory Structure
```
repo/
├── gui/                              # Source code
│   └── utils/
│       └── model_loader.py           # Module to test
├── tests/                            # All tests go here
│   ├── __init__.py
│   └── test_gui_utils_model_loader.py  # Tests for gui/utils/model_loader.py
└── pytest.ini                        # Pytest configuration
```

**Rules:**
1. **All tests MUST be in the `tests/` directory at the project root**
2. **Test files MUST start with `test_`** (e.g., `test_model_loader.py`)
3. **Test file names should mirror the module path** being tested
   - Module: `gui/utils/model_loader.py` → Test: `tests/test_gui_utils_model_loader.py`
   - Module: `gui/widgets/image_panel.py` → Test: `tests/test_gui_widgets_image_panel.py`
4. **Test classes MUST start with `Test`** (e.g., `class TestModelLoader`)
5. **Test functions MUST start with `test_`** (e.g., `def test_load_model()`)

#### Why `tests/` at Root Level (Not `gui/tests/`)?
- ✅ Separates tests from distributable code
- ✅ Standard Python/pytest convention
- ✅ Allows testing interactions between modules (`gui`, `model`, `dataset`)
- ✅ Simplifies test discovery with pytest
- ✅ Easier to exclude from package distribution

### Writing Tests

#### Test File Template
```python
"""Tests for <module_name> module.

Brief description of what's being tested.
"""

import pytest
from gui.utils.model_loader import ModelLoader  # Import module under test


class TestFeatureName:
    """Tests for specific feature or class."""

    def test_something_works(self):
        """Test that something works as expected."""
        # Arrange
        loader = ModelLoader()
        
        # Act
        result = loader.some_method()
        
        # Assert
        assert result is not None


    def test_error_handling(self):
        """Test that errors are handled correctly."""
        loader = ModelLoader()
        
        with pytest.raises(ValueError):
            loader.invalid_operation()
```

#### Using Test Markers
Mark tests to categorize them (defined in `pytest.ini`):

```python
import pytest

@pytest.mark.unit
def test_fast_unit_test():
    """Fast test with no external dependencies."""
    assert 1 + 1 == 2

@pytest.mark.integration
def test_with_real_model():
    """Test that requires actual model weights."""
    # May skip if weights not available
    pass

@pytest.mark.gui
def test_gui_component():
    """Test for GUI components."""
    pass

@pytest.mark.slow
def test_time_consuming():
    """Test that takes significant time."""
    pass
```

#### Testing Best Practices

1. **Use fixtures for common setup:**
```python
@pytest.fixture
def model_loader():
    """Provide a fresh ModelLoader instance for each test."""
    return ModelLoader()

def test_with_fixture(model_loader):
    assert model_loader.is_model_loaded() is False
```

2. **Use `tmp_path` for file operations:**
```python
def test_file_operation(tmp_path):
    """Test that creates temporary files."""
    test_file = tmp_path / "test.h5"
    test_file.write_bytes(b"dummy")
    # File automatically cleaned up after test
```

3. **Skip tests conditionally:**
```python
import pytest
from pathlib import Path

def test_requires_weights():
    """Test that needs real model weights."""
    weights_path = "./weights/zero_dce.weights.h5"
    if not Path(weights_path).exists():
        pytest.skip("Model weights not found")
    
    # Test code here
```

4. **Group related tests in classes:**
```python
class TestModelLoading:
    """All tests related to loading models."""
    
    def test_load_valid_model(self):
        pass
    
    def test_load_invalid_model(self):
        pass

class TestModelInference:
    """All tests related to model inference."""
    
    def test_inference_on_image(self):
        pass
```

5. **Write descriptive test names and docstrings:**
```python
def test_load_model_raises_file_not_found_when_weights_missing(self):
    """Test that load_model raises FileNotFoundError for missing weights file.
    
    This ensures the user gets a clear error message instead of a cryptic
    exception when the weights file doesn't exist.
    """
    loader = ModelLoader()
    
    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load_model("/nonexistent/path.h5")
    
    assert "not found" in str(exc_info.value).lower()
```

### Running Tests as an Agent

**Always activate the virtual environment first:**

```bash
# Detect operating system from user info context
# For Linux (default bash shell):
source .venv/bin/activate && pytest tests/test_gui_utils_model_loader.py -v

# For macOS (default zsh shell):
source .venv/bin/activate && pytest tests/test_gui_utils_model_loader.py -v

# For Linux with fish shell (if explicitly used):
source .venv/bin/activate.fish && pytest tests/test_gui_utils_model_loader.py -v
```

### Test Coverage

Check test coverage to ensure all code is tested:

```bash
# Generate coverage report
source .venv/bin/activate && pytest --cov=gui --cov-report=term-missing

# Generate HTML coverage report
source .venv/bin/activate && pytest --cov=gui --cov-report=html
# Opens htmlcov/index.html to view detailed coverage
```

### When to Write Tests

**ALWAYS write tests when:**
1. Creating a new module or class
2. Adding new features or methods
3. Fixing bugs (write a test that reproduces the bug first)
4. Refactoring code (ensure tests pass before and after)

**Test-Driven Development (TDD) workflow:**
1. Write a failing test for the desired functionality
2. Implement the minimum code to make the test pass
3. Refactor while keeping tests green
4. Repeat

## Important Notes for AI Agents

### Virtual Environment Requirements
- **ALWAYS activate the virtual environment** before running Python commands
- **Detect the operating system** from the user info context:
  - Linux → `source .venv/bin/activate` (bash is default shell)
  - macOS → `source .venv/bin/activate` (zsh is default shell)
  - Windows → `.venv\Scripts\activate`
- **Special case:** If user explicitly uses `/usr/bin/fish` shell → `source .venv/bin/activate.fish`
- **Use command chaining** with `&&` to ensure activation happens before execution
- **Example:** `source .venv/bin/activate && python train.py`

### When Refactoring
- **Preserve exact logic**: Don't "improve" algorithms unless explicitly asked
- **Keep loss weights**: The specific weights (200, 10, 5, 1) are tuned
- **Maintain compatibility**: Ensure model weights from original code can be loaded
- **Test incrementally**: Each module should be tested before moving to the next

### When Debugging
- Check image shape: Should be (batch, height, width, 3)
- Verify value range: Images should be [0, 1] after preprocessing
- Loss values: Should decrease over training (total loss typically starts ~0.5-1.0)
- Memory issues: Reduce batch size if OOM errors occur

### When Adding Features
- Keep backward compatibility with existing trained models
- Document new hyperparameters and their default values
- Update this file and REFACTOR_PLAN.md accordingly
- Add CLI arguments for configurability

## References

### Paper
- **Title:** Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
- **Authors:** Chunle Guo, Chongyi Li, et al.
- **Link:** https://arxiv.org/abs/2001.06826
- **Published:** CVPR 2020

### Original Implementation
- Keras example: https://keras.io/examples/vision/zero_dce/
- This project is based on the Keras 3 port by Soumik Rakshit

### Related Documentation
- `REFACTOR_PLAN.md`: Step-by-step refactoring instructions
- `README.md`: Project overview and usage guide

## Git Workflow

**Manual commits by student after each step:**
```bash
# After completing a refactoring step
git add <modified_files>
git commit -m "Refactor: Implement <module_name> - <description>"
```

**Agents should NOT:**
- Run git commands automatically
- Create commits without user approval
- Push to remote repositories

---

**Last Updated:** 2025-10-28  
**Maintained By:** Graduate students in Digital Image Processing course  
**For Questions:** Refer to `REFACTOR_PLAN.md` for detailed implementation steps

## Quick Reference for AI Agents

### File Locations Checklist
- ✅ Source code → Root level or `gui/` for GUI modules
- ✅ Tests → `tests/` directory at root
- ✅ Configuration → Root level (`pytest.ini`, `pyproject.toml`)
- ✅ Documentation → Root level or `docs/`

### Before Running Any Command
```bash
# 1. Check operating system from user info (Linux defaults to bash)
# 2. Activate virtual environment
# 3. Run the command

# Example (Linux with default bash shell):
source .venv/bin/activate && <command>
```

### Test File Naming Convention
```
Module: gui/utils/model_loader.py
  ↓
Test: tests/test_gui_utils_model_loader.py

Module: gui/widgets/image_panel.py
  ↓
Test: tests/test_gui_widgets_image_panel.py

Module: dataset.py (root level)
  ↓
Test: tests/test_dataset.py
```
