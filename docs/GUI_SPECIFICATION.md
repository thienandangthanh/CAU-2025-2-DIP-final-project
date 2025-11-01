# Zero-DCE GUI Application - Feature Specification

> **Version:** 2.1.0  
> **Last Updated:** 2025-10-31  
> **Status:** Phase 3 In Progress (Task 3.2 Complete - Comparison View)

## Overview

A cross-platform desktop application that provides an intuitive graphical user interface for enhancing low-light images using the Zero-DCE (Zero-Reference Deep Curve Estimation) deep learning model. The application is designed for end-users who want to improve their low-light photographs without requiring technical knowledge of machine learning or command-line tools.

### Target Platforms
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 20.04+, other major distributions)
- ✅ macOS 10.15+

### Technology Stack
- **Framework:** PyQt6
- **Backend:** TensorFlow/Keras 3
- **Image Processing:** PIL/Pillow
- **Model:** Zero-DCE (from this project)

---

## User Interface Design

### Overall Layout

```
┌─────────────────────────────────────────────────────────────┐
│ File   Help                                    [_] [□] [X]  │ ← Menu Bar
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐     ┌───┐     ┌─────────────────┐      │
│  │                 │     │   │     │                 │      │
│  │   Input Image   │     │ → │     │  Output Image   │      │
│  │   Placeholder   │     │   │     │   Placeholder   │      │
│  │                 │     └───┘     │                 │      │
│  │   (Clickable)   │               │   (Clickable)   │      │
│  │                 │   Enhance     │                 │      │
│  │                 │    Button     │                 │      │
│  └─────────────────┘               └─────────────────┘      │
│                                                             │
│  Status: Ready                                              │
├─────────────────────────────────────────────────────────────┤
│ Model: zero_dce.weights.h5    | Image: 1920x1080 | Ready    │ ← Status Bar
└─────────────────────────────────────────────────────────────┘
```

---

## Core Features

### 1. Menu Bar

#### 1.1 File Menu
- **Open Image... (Ctrl+O)**
  - Opens system file dialog to select an image
  - Supported formats: JPEG, PNG
  - Max file size: 50MB (configurable)
  - Displays selected image in left panel
  - Clears any previous output image

- **Save Enhanced Image... (Ctrl+S)**
  - Only enabled when enhanced image is available
  - Opens system save dialog
  - Default filename: `<original_name>_enhanced.<ext>`
  - Supported save formats: JPG, PNG
  - Quality settings for JPEG (default: 95)

- **Recent Files** ✅ IMPLEMENTED
  - Shows last 10 opened images
  - Clicking reopens the image
  - Stored in application settings
  - "(Empty)" state when no recent files
  - "Clear Recent Files" action
  - Files that don't exist show "(not found)" and are disabled

- **Exit (Ctrl+Q)**
  - Closes the application
  - Prompts to save if enhanced image hasn't been saved

#### 1.2 Edit Menu
- **Clear Input (Ctrl+Del)**
  - Clears the input image
  - Resets both panels to placeholder state

- **Preferences... (Ctrl+,)** ✅ IMPLEMENTED
  - Opens preferences dialog (see Section 4)
  - Modal dialog with tabbed interface
  - Handles unsaved changes warnings (Escape/Cancel/Alt+F4)

#### 1.3 View Menu ✅ NEW (Phase 3)
- **Comparison Mode (C)** ✅ IMPLEMENTED
  - Toggles between single enhancement and comparison modes
  - Checkable menu item (shows active state)
  - Keyboard shortcut: `C`
  - Shows method selection dialog when entering comparison mode

#### 1.4 Model Menu
- **Load Model Weights...**
  - Opens file dialog to select `.weights.h5` weights file
  - Default location: `./weights/` directory
  - Validates model compatibility before loading

- **Default Weights** ✅ IMPLEMENTED
  - Submenu listing available weights in configured directory
  - Quick selection of pre-trained models
  - Shows training epoch in name (e.g., "Epoch 200")
  - Checkmark on currently loaded model
  - Directory configurable in Preferences
  - Refreshes when directory changes in Preferences

- **Model Info** ✅ IMPLEMENTED
  - Shows current loaded model details:
    - File path
    - File size (formatted in MB)
    - Simple message box display

#### 1.5 Help Menu
- **About Zero-DCE**
  - Shows information about the Zero-DCE algorithm
  - Links to original paper (CVPR 2020)
  - Brief explanation of how it works

- **Keyboard Shortcuts**
  - Displays all available keyboard shortcuts

- **About**
  - Application version
  - Credits
  - License information
  - Links to GitHub repository

---

### 2. Display Modes

#### 2.1 Single Enhancement Mode (Default)

Standard two-panel layout for single image enhancement.

### 2.2 Comparison Mode ✅ NEW (Phase 3)

Side-by-side comparison of multiple enhancement methods.

**Features:**
- **Comparison Grid Widget:**
  - Displays multiple enhancement results simultaneously
  - Intelligent grid layout based on cell count:
    - 2 cells: 1×2 grid (50% width each)
    - 3 cells: 1×3 grid (33.3% width each)
    - 4 cells: 2×2 grid (50% width × 50% height)
    - 5-6 cells: 2×3 grid (33.3% width × 50% height)
    - 7+ cells: 4 columns (25% width, multiple rows)
  - Uniform cell dimensions (all cells equal size)
  - Scrollable for many methods
  
- **Comparison Cells:**
  - Method name label at top
  - Enhanced image (scaled to fit, maintains aspect ratio)
  - Status indicator with color-coded borders:
    - Gray (pending), Blue (running), Green (done), Red (error)
  - Timing information (e.g., "2.34s")
  - Click to expand (future feature)
  
- **Input & Reference Cells:**
  - Always shows original input image
  - Optional reference (high-light) image cell
  - Same uniform dimensions as method cells
  
- **Method Selection Dialog:**
  - Checkboxes for all available methods (grouped by category)
  - Deep Learning: Zero-DCE (disabled without model)
  - Classical: AutoContrast, Histogram Eq, CLAHE, Gamma
  - Quick selection buttons: All, None, Classical Only, Fast Methods
  - Optional reference image picker (browse/clear)
  - Validation: at least one method required
  
- **Background Processing:**
  - Methods run sequentially in background thread
  - Progressive UI updates (results appear as completed)
  - Status bar shows progress
  - Graceful error handling (one failure doesn't stop others)

**Toggle:**
- Button: "Compare Methods" in single mode
- Menu: View → Comparison Mode
- Keyboard: `C`

**Workflow:**
1. Load input image
2. Click "Compare Methods" or press `C`
3. Select methods and optional reference in dialog
4. View results in grid as they complete
5. Toggle back to single mode with `C`

### 3. Image Display Panels (Single Mode)

#### 3.1 Left Panel: Input Image

**States:**

**A. Empty State (No Image Loaded)**
- Display: Placeholder graphic with icon
  - Icon: Camera or image file icon (📷 or 🖼️)
  - Text: "Click to open an image" or "Drag & drop an image here"
  - Border: Dashed border to indicate drop zone
  - Background: Light gray (#F5F5F5)

**B. Image Loaded State**
- Display: Input image fitted to panel
  - Maintains aspect ratio
  - Centers image in panel
  - Shows full image without cropping
  - Add subtle shadow/border for visual separation

**Interactions:**
- **Click:** Opens file dialog (same as File → Open)
- **Drag & Drop:** Accepts image files dragged into the panel
- **Right-click Menu:**
  - Open Different Image

**Visual Indicators:**
- Label: "Input Image" (top of panel)
- Image info overlay (bottom corner, toggleable):
  - Filename
  - Dimensions (WxH)
  - File size

#### 3.2 Right Panel: Enhanced Image

**States:**

**A. Empty State (No Enhancement Yet)**
- Display: Placeholder graphic with icon
  - Icon: Sparkles or magic wand icon (✨)
  - Text: "Enhanced image will appear here"
  - Background: Light gray (#F5F5F5)

**B. Processing State (During Enhancement)**
- Display: Loading indicator
  - Animated spinning circle/progress indicator
  - Text: "Enhancing..." with percentage if possible
  - Semi-transparent overlay on placeholder
  - Cancel button (optional, see Section 3.3)

**C. Enhanced Image Loaded State**
- Display: Enhanced image fitted to panel
  - Same display properties as input panel
  - Add subtle highlight to indicate "new" result

**Interactions:**
- **Click (when enhanced image available):** Opens save dialog
- **Right-click Menu:**
  - Save Image
  - Compare Methods… (launch comparison mode and method selection)
  - Quick Split View (toggle per-panel slider, see Section 2.3)

**Visual Indicators:**
- Label: "Enhanced Image" (top of panel)
- Image info overlay (bottom corner, toggleable):
  - Filename
  - Dimensions (WxH)
  - File size
- Processing time indicator (bottom corner): "Enhanced in 2.3s"

#### 2.3 Image Comparison Features

Comparison now supports both single-result inspection and multi-method analysis. Users can switch between modes as needed; all comparison views share the same zoom and navigation controls.

**Multi-Method Grid (Phase 3 focus)**
- Triggered when comparison mode is enabled and multiple enhancement methods are selected.
- Grid layout adapts to available width (2 columns minimum, 3-4 columns on wide screens).
- Each cell shows:
  - Enhanced image rendered via `ImagePanel` derivative. The grid always includes the original low-light input cell; if the user supplies a high-light reference, it appears alongside enhancement results.
  - Method name, execution time, and status badges (pending, running, completed, failed).
  - Inline before/after slider anchored to the original image. Slider defaults to 50/50 split, is draggable horizontally, and reveals the original image on the left side of the bar and the enhanced result on the right.
  - Action icons (expand, save, context menu).
- Cells display live progress indicators until their enhancement completes.
- Double-click or press `Enter` on a focused cell to expand it in a dedicated viewer while preserving the grid state.

**Per-Cell Before/After Slider**
- Slider handle appears on hover/focus; keyboard users can adjust with `Left/Right` arrows.
- Press `Space` to toggle between 0%, 50%, and 100% reveal states quickly.
- Enhancement cells use cached original image data to ensure consistent comparisons across methods, while the reference cell defaults to a static display (toggle shows reference vs original when available).
- Works in both grid view and expanded cell view.

**Side-by-Side View (Single-Result Mode)**
- Remains available when only one enhancement result is present.
- Input on left, output on right with independent zoom/pan.
- Acts as fallback for users who prefer traditional two-panel comparison.

**Split View (Expanded View Option)**
- Vertical slider overlay inside a single panel.
- Accessible from the expanded cell view or via the legacy “Compare with Original” action.
- Useful for detailed inspection of a single method; co-exists with multi-cell comparison.
- Keyboard shortcut: `Shift+C` (toggle when expanded) while `C` toggles global comparison grid mode.

**Zoom Controls (Shared Across Modes)**
- Zoom in/out buttons (+ / -).
- Fit to window (default).
- Actual size (100%).
- Optional synchronized zoom for selected cells.
- Mouse wheel zoom with modifier (`Ctrl` + scroll).
- Keyboard: `Ctrl +`, `Ctrl -`, `Ctrl 0`.

---

### 3. Enhancement Control

#### 3.1 Enhance Button

**Location:** Centered between input and output panels

**Appearance:**
- Shape: Rounded rectangle
- Icon: Right arrow (→)
- Size: 48x48 pixels (or larger for accessibility)
- Color: Primary accent color (e.g., blue #2196F3)
- Hover effect: Lighten color, add shadow
- Active effect: Press animation

**States:**

**A. Disabled State**
- Condition: No input image loaded OR no model loaded
- Appearance: Grayed out, cursor shows "not-allowed"
- Tooltip: "Load an image and model first"

**B. Ready State**
- Condition: Input image and model both loaded
- Appearance: Full color, cursor shows "pointer"
- Tooltip: "Enhance image (Ctrl+E)"

**C. Processing State**
- Condition: Enhancement in progress
- Appearance: Animated (rotating or pulsing)
- Shows loading spinner overlay
- Button text changes to "Cancel" (optional)

**D. Completed State**
- Condition: Enhancement finished successfully
- Appearance: Brief success animation (check mark ✓)
- Returns to Ready state after 1 second

**Interactions:**
- **Click:** Triggers enhancement process
- **Keyboard:** `Ctrl+E` or `Space` (when button focused)

#### 3.2 Enhancement Process

**Workflow:**
1. User clicks Enhance button
2. UI switches to Processing state
3. Progress indicator appears in right panel
4. Backend performs inference:
   - Load image as numpy array
   - Preprocess (resize to 256x256, normalize)
   - Run through Zero-DCE model
   - Post-process output
5. On success:
   - Display enhanced image in right panel
   - Update status bar
   - Enable Save functionality
6. On error:
   - Show error dialog with details
   - Log error for debugging
   - Return to Ready state

**Performance Considerations:**
- Run inference in separate thread (avoid UI freezing)
- Disable UI interactions during processing
- Show estimated time remaining (optional)

---

### 4. Preferences Dialog ✅ IMPLEMENTED

**Access:** Edit → Preferences (Ctrl+,)

**Dialog Features:**
- Modal window (600x450 minimum)
- Tabbed interface for organizing settings
- OK/Cancel/Apply buttons with proper behavior
- Dirty state tracking with unsaved changes warnings
- Escape key, Cancel button, and Alt+F4 all trigger warnings if unsaved
- Settings immediately refresh main window UI

**Categories (Tabbed Interface):**

#### 4.1 General Tab ✅ IMPLEMENTED

**Model Settings:**
- **Weights Directory:** ⚠️ DESIGN CHANGE
  - Directory path (not specific file) for simpler UX
  - Browse button to select directory
  - Auto-load model on startup (checkbox)
  - **Rationale:** Users set directory once, pick specific models from menu

**Display Settings:**
- **Default zoom level:** [Fit to Window | Actual Size (100%)]
- **Keep zoom synchronized between panels** (checkbox)
- **Show image info overlay** (checkbox)

**Performance Settings:**
- **GPU Acceleration:** [Auto (Recommended) | Enable | Disable (CPU Only)]
  - Live GPU status indicator showing detected GPUs
- **Max image dimension:** [2048 | 4096 (Recommended) | 8192 | Unlimited]
  - Warning shown if Unlimited selected

#### 4.2 Advanced Tab (Phase 3)
**Status:** Deferred to Phase 3

Planned features:
- Model enhancement iterations
- Output format settings
- Cache management
- Debug logging

---

### 5. Status Bar ✅ IMPLEMENTED

**Location:** Bottom of window

**Components:**

**A. Left Section:** ✅ IMPLEMENTED
- Status messages with timing:
  - "Ready" (idle)
  - "Loading model..." (during model load)
  - "Enhancing image..." (during processing)
  - "Enhanced successfully in 2.34s" (after completion with timing)
  - "Preferences saved and applied" (after settings change)
  - Error messages when operations fail

**B. Right Section (Permanent):** ✅ IMPLEMENTED
- Current model info:
  - Format: "Model: <filename>" (when loaded)
  - Format: "No model loaded" (when not loaded)
  - Updates immediately when model loaded or settings changed

---

## Technical Specifications

### 6. Image Handling

#### 6.1 Input Processing
- **Supported Formats:**
  - Read: JPEG, PNG
  - Write: JPEG, PNG

- **Size Limits:**
  - Minimum: 64x64 pixels
  - Maximum: 8192x8192 pixels (configurable)
  - Recommended: 256x256 to 2048x2048

- **Preprocessing Pipeline:**
  1. Load image using PIL
  2. Convert to RGB (if grayscale or other mode)
  3. Resize if exceeds max dimension (preserve aspect ratio)
  4. Normalize to [0, 1] range
  5. Prepare tensor for model input

#### 6.2 Output Processing
- **Post-processing Pipeline:**
  1. Extract tensor from model output
  2. Denormalize to [0, 255]
  3. Convert to uint8
  4. Create PIL Image
  5. Optionally resize to original dimensions

- **Save Options:**
  - PNG: Lossless, preserves quality
  - JPEG: Adjustable quality (default 95)
  - Preserve EXIF metadata (checkbox in save dialog)

### 7. Model Integration

#### 7.1 Model Loading
- **Initialization:**
  - Load default weights on startup (if configured)
  - Lazy loading: Only load when needed
  - Validate model architecture compatibility

- **Weight Files:**
  - Format: `.weights.h5` (Keras)
  - Location: `./weights/` directory by default
  - Fallback: Prompt user if default not found

- **Error Handling:**
  - Invalid file format → Error dialog
  - Corrupted weights → Error dialog with recovery options
  - Missing dependencies → Guide user to install

#### 7.2 Inference Pipeline
```python
# Pseudocode
def enhance_image(input_image: PIL.Image) -> PIL.Image:
    # 1. Preprocess
    tensor = preprocess(input_image)
    
    # 2. Run model
    with torch.no_grad():  # or tf.function for Keras
        enhanced_tensor = model(tensor)
    
    # 3. Post-process
    output_image = postprocess(enhanced_tensor)
    
    return output_image
```

#### 7.3 Performance Optimization
- **GPU Acceleration:**
  - Auto-detect CUDA/Metal availability
  - Fallback to CPU if GPU unavailable
  - Show GPU status in Preferences

- **Threading:**
  - Use QThread for background processing
  - Signals/slots for UI updates
  - Cancel mechanism (optional)

### 8. Error Handling

#### 8.1 Error Types & Messages

**Model Errors:**
- "Failed to load model weights"
  - Solution: Check file path and format
- "Model inference failed"
  - Solution: Check image format and size
- "Out of memory"
  - Solution: Reduce image size or enable CPU mode

**Image Errors:**
- "Failed to open image"
  - Solution: Check file format and corruption
- "Image too large"
  - Solution: Resize image or adjust max dimension
- "Unsupported image format"
  - Solution: Convert to supported format

**File I/O Errors:**
- "Permission denied"
  - Solution: Check folder permissions
- "Disk full"
  - Solution: Free up space
- "File not found"
  - Solution: Check if file was moved/deleted

#### 8.2 Error Dialog Design
- Clear error message (user-friendly language)
- Technical details (expandable section)
- Suggested solutions
- Copy error button (for bug reports)
- OK / Cancel buttons

---

## User Workflows

### 9. Primary Use Cases

#### 9.1 Basic Enhancement Workflow
1. Launch application
2. Click left panel OR File → Open
3. Select low-light image from file dialog
4. Image displays in left panel
5. Click Enhance button (center)
6. Wait for processing (spinner shown)
7. Enhanced image appears in right panel
8. Click right panel OR File → Save to export
9. Select save location and format
10. Done!

**Expected Time:** 30 seconds (first-time user)

#### 9.2 Comparison Workflow
1. Load an input image (steps 1-4 above) and ensure a model is available.
2. Press `C`, use View → Comparison Mode, or click the toolbar toggle to open comparison mode.
3. Method selection dialog appears (unless a previous selection is cached); choose one or more enhancement methods and optionally pick a high-light reference image (import dialog offers “Use paired reference” or “Browse…” when available).
4. UI switches to multi-method grid. The original low-light input appears in the first cell, followed by the optional reference cell, then each selected method rendered into its own cell with progress indicator and inline before/after slider.
5. Drag the slider in any cell to reveal the original image on the left of the handle and the enhanced output on the right. Keyboard users can adjust the slider with arrow keys.
6. Double-click a cell (or use the expand icon) to open the enhanced result in an expanded view. In this view, press `Shift+C` (or use the on-screen toggle) to switch the per-cell slider into full split-view mode for precise inspection.
7. Use grid controls to zoom, pan, and navigate between cells. Press `C` again to exit comparison mode and return to single-result view.

---

## Accessibility Features

### 10. Keyboard Navigation

**Global Shortcuts:**
- `Ctrl+O`: Open image
- `Ctrl+S`: Save enhanced image
- `Ctrl+E`: Enhance (trigger button)
- `C`: Toggle comparison mode ✅ NEW (Phase 3)
- `Ctrl+Q`: Quit application
- `Ctrl+,`: Preferences
- `Ctrl+W`: Close window
- `Ctrl+Z`: Undo (future feature)
- `F1`: Help
- `Space`: Enhance (when button focused)

**Navigation:**
- `Tab`: Move between UI elements
- `Shift+Tab`: Move backward
- `Enter`: Activate focused button
- `Esc`: Close dialog / Cancel operation

### 11. Screen Reader Support
- All UI elements have proper labels
- Images have alt-text descriptions
- Status updates announced to screen readers
- Progress indicators with text descriptions

### 12. Visual Accessibility
- High contrast mode support
- Minimum text size: 12pt
- Adjustable UI scale (80%-150%)
- Colorblind-friendly color scheme
- Clear focus indicators for keyboard navigation

---

## Non-Functional Requirements

### 13. Performance
- **Startup Time:** < 3 seconds
- **Model Loading:** < 2 seconds
- **Enhancement Time:** < 5 seconds (typical image, GPU)
- **UI Responsiveness:** 60 FPS animations, no freezing

### 14. Reliability
- **Stability:** No crashes during normal operation
- **Error Recovery:** Graceful handling of all errors
- **Data Safety:** No data loss on unexpected closure
- **Testing:** Unit tests for core functions, UI tests

### 15. Security
- **File Validation:** Check file signatures, not just extensions
- **Resource Limits:** Prevent memory exhaustion
- **Safe Paths:** Validate user-provided paths
- **No Network Access:** Fully offline application

### 16. Usability
- **Learning Curve:** 5 minutes for basic usage
- **Documentation:** Comprehensive help system
- **Intuitive Design:** Follow platform conventions
- **Feedback:** Clear status messages and progress indicators

---

## Future Enhancements

### 17. Potential Features (Future)

- [x] **Comparison with Other Methods:** Implemented via multi-method comparison grid (Phase 3).

---

## Implementation Notes

### 19. Development Phases

#### Phase 1: Core Functionality (MVP)
- Basic UI layout (input, output, button)
- File open/save dialogs
- Model loading and inference
- Simple error handling
- **Estimated Time:** 2-3 weeks

#### Phase 2: Polish & UX
- Menu bar implementation
- Status bar
- Preferences dialog
- Keyboard shortcuts
- Better error messages
- **Estimated Time:** 1-2 weeks

#### Phase 3: Advanced Features ✅ IN PROGRESS
- ✅ Multi-method comparison grid (Task 3.2 complete)
- ✅ Method selection dialog with quick presets
- ✅ Optional reference image support
- ✅ Comparison mode toggle (keyboard shortcut `C`)
- ✅ Uniform cell dimensions with intelligent grid layout
- ✅ Progressive result updates with status indicators
- ✅ Comparison workflow integrated into main window (Task 3.3 core)
- ⏳ Comparison mode cleanup (state reset + progress counters)
- ⏳ Export comparison results (Task 3.5)
- ⏳ Method preferences (Task 3.4)
- **Estimated Time:** 2-3 weeks

#### Phase 4: Testing & Release
- Cross-platform testing
- Bug fixes
- Documentation
- Packaging (installers)
- **Estimated Time:** 1-2 weeks

**Total Estimated Time:** 6-10 weeks

### 20. Technology Decisions

**PyQt6 Advantages:**
- ✅ Cross-platform (Windows, Linux, macOS)
- ✅ Native look and feel
- ✅ Comprehensive UI widgets
- ✅ Good documentation
- ✅ Commercial-friendly license (GPL/Commercial)

**Alternative Considered:**
- Tkinter: Too basic, limited styling
- Kivy: Better for mobile, overkill here
- Electron: Too heavy, requires web tech
- Qt for Python (PySide6): Similar to PyQt6, acceptable alternative

### 21. File Structure (Proposed)

```
repo/
├── gui/
│   ├── __init__.py
│   ├── main_window.py          # Main window class
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── image_panel.py      # Image display widget
│   │   ├── enhance_button.py   # Custom button widget
│   │   ├── comparison_cell.py  # Comparison cell widget ✅ NEW
│   │   └── comparison_grid.py  # Comparison grid widget ✅ NEW
│   ├── dialogs/
│   │   ├── __init__.py
│   │   ├── preferences_dialog.py      # Preferences dialog
│   │   ├── error_dialog.py            # Error display
│   │   └── method_selection_dialog.py # Method selection ✅ NEW
│   ├── resources/
│   │   ├── icons/              # Application icons
│   │   ├── images/             # Placeholder graphics
│   │   └── styles.qss          # Qt stylesheets
│   └── utils/
│       ├── __init__.py
│       ├── image_processor.py     # Image I/O and processing
│       ├── model_loader.py        # Model management
│       ├── settings.py            # Application settings
│       ├── enhancement_result.py  # Result container
│       ├── enhancement_methods.py # Method registry ✅ NEW
│       └── enhancement_runner.py  # Multi-method runner ✅ NEW
├── gui_app.py                   # Main entry point
└── docs/
    ├── GUI_SPECIFICATION.md     # This file
    └── GUI_USER_GUIDE.md        # End-user documentation
```

---

## Design Mockups

### 22. Visual Design Guidelines

**Color Palette:**
- Primary: #2196F3 (Blue)
- Secondary: #4CAF50 (Green)
- Accent: #FF9800 (Orange)
- Background: #FFFFFF (White)
- Text: #212121 (Dark)

**Typography:**
- UI Font: System default (Segoe UI, San Francisco, Ubuntu)
- Code Font: Monospace (Consolas, Menlo, Monospace)
- Sizes: 10pt (small), 12pt (normal), 14pt (large), 18pt (headers)

**Spacing:**
- Padding: 8px, 16px, 24px
- Margins: 8px, 16px
- Corner radius: 4px (subtle), 8px (buttons)

**Icons:**
- Style: Material Design Icons or similar
- Size: 24x24px (small), 48x48px (large)
- Color: Inherit from text or accent color

---

## Testing Requirements

### 23. Test Scenarios

#### Functional Tests
- [ ] Open various image formats (JPEG, PNG)
- [ ] Handle corrupted image files gracefully
- [ ] Load different model weights
- [ ] Enhancement produces correct output
- [ ] Save enhanced image successfully
- [ ] All menu items work correctly
- [ ] Keyboard shortcuts function properly
- [ ] Preferences persist across sessions

#### UI Tests
- [ ] Window resizes correctly
- [ ] Panels maintain aspect ratios
- [ ] Buttons respond to hover/click
- [ ] Dialogs display properly
- [ ] Status bar updates correctly
- [ ] Loading spinner animates smoothly

#### Cross-Platform Tests
- [ ] Test on Windows 10, 11
- [ ] Test on Ubuntu 20.04, 22.04
- [ ] Test on macOS 11, 12, 13
- [ ] Verify file dialogs work on all platforms
- [ ] Check for platform-specific UI issues

#### Performance Tests
- [ ] Test with large images (8K resolution)
- [ ] Test with small images (64x64)
- [ ] Measure enhancement time (GPU vs CPU)
- [ ] Check memory usage during processing
- [ ] Verify no memory leaks over multiple enhancements

#### Error Handling Tests
- [ ] Missing model weights file
- [ ] Unsupported image format
- [ ] Out of memory situation
- [ ] Disk full when saving
- [ ] Permission denied errors

---

## Appendix

### A. References
- **Zero-DCE Paper:** https://arxiv.org/abs/2001.06826
- **PyQt6 Documentation:** https://doc.qt.io/qtforpython-6/
- **Material Design Guidelines:** https://material.io/design

### B. Glossary
- **Zero-DCE:** Zero-Reference Deep Curve Estimation
- **DCE-Net:** The neural network architecture used by Zero-DCE
- **Enhancement:** Process of improving image quality (brightness, contrast)
- **Inference:** Running an image through the trained model
- **Widget:** A UI component (button, panel, etc.)

### C. Changelog
- **v1.0.0-draft (2025-10-28):** Initial specification document
- **v2.0.0 (2025-10-30):** Phase 2 implementation complete
  - Added implementation status markers (✅)
  - Documented design changes (directory-only model selection)
  - Added Phase 2 completion summary
  - Updated status from "Planning" to "Phase 2 Complete"
- **v2.1.0 (2025-10-31):** Phase 3 Task 3.2 complete
  - Added comparison mode feature documentation
  - Added View menu with Comparison Mode toggle
  - Documented comparison grid, cells, and method selection dialog
  - Updated keyboard shortcuts (added `C` for comparison toggle)
  - Updated file structure with new Phase 3 modules

---

## Phase 2 Implementation Summary

### Completed Features (Phase 2)
1. ✅ **Recent Files Submenu** - 14 tests passing
2. ✅ **Enhanced Model Menu** - Default Weights submenu with epoch extraction
3. ✅ **Preferences Dialog** - Comprehensive settings management
4. ✅ **General Tab** - All basic settings with validation
5. ✅ **Progress Indicators** - Timing display for enhancements
6. ✅ **Keyboard Shortcuts** - Ctrl+, for preferences

### Test Coverage
- **Total Tests:** 215+ automated tests
- **Phase 1-2 Test Files:**
  - `test_gui_recent_files.py` - 14 tests
  - `test_gui_model_menu.py` - 11 tests
  - `test_gui_model_settings.py` - 9 tests
  - `test_gui_dialogs_preferences.py` - 42 tests
  - `test_gui_enhancement_result.py` - 33 tests
  - `test_gui_image_panel_timing.py` - 21 tests
  - `test_gui_main_window_timing.py` - 15 tests
  - Plus existing Phase 1 tests
- **Phase 3 Test Files (NEW):**
  - `test_gui_widgets_comparison_cell.py` - 14 tests ✅
  - `test_gui_widgets_comparison_grid.py` - 23 tests ✅
  - `test_gui_dialogs_method_selection.py` - 23 tests ✅
  - `test_gui_utils_enhancement_methods.py` - Tests for method registry
  - `test_gui_utils_enhancement_runner.py` - Tests for multi-method runner

### Design Improvements Made
1. **Simplified Model Selection:**
   - Original: Select specific file in preferences (confusing)
   - Implemented: Select directory only, pick model from menu (intuitive)
2. **Consistent Close Behavior:**
   - Escape key, Cancel button, Alt+F4 all show unsaved changes warning
3. **Live UI Updates:**
   - Settings changes immediately refresh main window
   - Default Weights menu updates when directory changes

### Known Limitations (Phase 2)
- Advanced tab deferred to Phase 3/4
- Status bar doesn't show image dimensions yet (Phase 3/4)
- ~~Compare mode keyboard shortcut deferred to Phase 3~~ ✅ IMPLEMENTED

---

## Sign-off

This specification has been validated through Phase 2 and partial Phase 3 implementation. All core features are working and tested.

**Current Status:** Phase 3 In Progress (Task 3.2 Complete) ✅  
**Completed:** Comparison view widget with method selection  
**Next Tasks:** Task 3.3 (Further integration), Task 3.4 (Preferences), Task 3.5 (Export)
