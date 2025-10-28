# Zero-DCE GUI Application - Feature Specification

> **Version:** 1.0.0-draft  
> **Last Updated:** 2025-10-28  
> **Status:** Planning Phase

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

- **Recent Files**
  - Shows last 10 opened images
  - Clicking reopens the image
  - Stored in application settings

- **Exit (Ctrl+Q)**
  - Closes the application
  - Prompts to save if enhanced image hasn't been saved

#### 1.2 Edit Menu
- **Clear Input (Ctrl+Del)**
  - Clears the input image
  - Resets both panels to placeholder state

- **Preferences... (Ctrl+,)**
  - Opens preferences dialog (see Section 4)

#### 1.3 Model Menu
- **Load Model Weights...**
  - Opens file dialog to select `.weights.h5` weights file
  - Default location: `./weights/` directory
  - Validates model compatibility before loading

- **Default Weights**
  - Submenu listing available weights in `./weights/` directory
  - Quick selection of pre-trained models
  - Shows training epoch in name (e.g., "Epoch 200")

- **Model Info**
  - Shows current loaded model details:
    - File path
    - File size
    - Training epoch (if available)
    - Load timestamp

#### 1.4 Help Menu
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

### 2. Image Display Panels

#### 2.1 Left Panel: Input Image

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

#### 2.2 Right Panel: Enhanced Image

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
  - Compare with Original (toggle split view, see Section 2.3)

**Visual Indicators:**
- Label: "Enhanced Image" (top of panel)
- Image info overlay (bottom corner, toggleable):
  - Filename
  - Dimensions (WxH)
  - File size
- Processing time indicator (bottom corner): "Enhanced in 2.3s"

#### 2.3 Image Comparison Features

**Side-by-Side View (Default)**
- Input on left, output on right
- Independent zoom/pan (optional)

**Split View (Toggle)**
- Single panel with vertical slider
- Drag slider to reveal input/output
- Useful for before/after comparison
- Keyboard shortcut: `C` (Compare)

**Zoom Controls**
- Zoom in/out buttons (+ / -)
- Fit to window (default)
- Actual size (100%)
- Synchronized zoom for both panels (optional)
- Mouse wheel zoom
- Keyboard: `Ctrl +` / `Ctrl -` / `Ctrl 0`

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

### 4. Preferences Dialog

**Access:** Edit → Preferences (Ctrl+,)

**Categories (Tabbed Interface):**

#### 4.1 General Tab
- **Default Model Weights:**
  - Path to default `.weights.h5` file
  - Browse button to select file
  - Auto-load on startup (checkbox)

- **Image Display:**
  - Default zoom level: [Fit to Window | Actual Size]
  - Keep zoom synchronized between panels (checkbox)
  - Show image info overlay (checkbox)

- **Performance:**
  - GPU Acceleration: [Auto | Enable | Disable]
  - Max image dimension (for large files): [2048 | 4096 | Unlimited]

#### 4.2 Advanced Tab
- **Model Settings:**
  - Number of enhancement iterations (default: 8)
  - Output image format: [PNG | JPEG]
  - JPEG quality: Slider (0-100, default 95)

- **Temporary Files:**
  - Cache directory location
  - Clear cache button
  - Auto-clear on exit (checkbox)

- **Logging:**
  - Enable debug logging (checkbox)
  - Log file location
  - Open log folder button

---

### 5. Status Bar

**Location:** Bottom of window

**Components:**

**A. Left Section:**
- Status message:
  - "Ready" (idle)
  - "Loading model..." (during model load)
  - "Enhancing image..." (during processing)
  - "Enhanced successfully" (after completion)
  - "Error: <message>" (on error)

**B. Center Section:**
- Current model info:
  - Format: "Model: <filename>"
  - Tooltip shows full path
  - Click to change model

**C. Right Section:**
- Image dimensions: "1920x1080 px"
- Processing time: "2.3s"
- Memory usage (optional): "Memory: 512 MB"

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
1. Load and enhance image (steps 1-7 above)
2. Press `C` or right-click → Compare
3. UI switches to split-view mode
4. Drag slider to compare before/after
5. Zoom in on specific areas
6. Toggle back to side-by-side with `C`

---

## Accessibility Features

### 10. Keyboard Navigation

**Global Shortcuts:**
- `Ctrl+O`: Open image
- `Ctrl+S`: Save enhanced image
- `Ctrl+E`: Enhance (trigger button)
- `Ctrl+Q`: Quit application
- `Ctrl+,`: Preferences
- `Ctrl+W`: Close window
- `Ctrl+Z`: Undo (future feature)
- `F1`: Help
- `C`: Toggle compare mode
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

- [ ] **Comparison with Other Methods:** CLAHE, Histogram Eq, etc.

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

#### Phase 3: Advanced Features
- Split-view comparison
- Zoom/pan controls
- Recent files
- Model management
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
│   │   └── loading_overlay.py  # Loading spinner
│   ├── dialogs/
│   │   ├── __init__.py
│   │   ├── preferences.py      # Preferences dialog
│   │   ├── about.py            # About dialog
│   │   └── error_dialog.py     # Error display
│   ├── resources/
│   │   ├── icons/              # Application icons
│   │   ├── images/             # Placeholder graphics
│   │   └── styles.qss          # Qt stylesheets
│   └── utils/
│       ├── __init__.py
│       ├── image_processor.py  # Image I/O and processing
│       ├── model_loader.py     # Model management
│       └── settings.py         # Application settings
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

---

## Sign-off

This specification document serves as the blueprint for developing the Zero-DCE GUI application. It will be iteratively refined based on:
- Student feedback and requirements
- Technical feasibility during implementation
- User testing and feedback

**Next Steps:**
1. Review and revise this specification
2. Create detailed mockups/wireframes
3. Set up PyQt6 development environment
4. Begin Phase 1 implementation
5. Iterate and refine based on progress
