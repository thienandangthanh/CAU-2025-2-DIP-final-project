# Zero-DCE GUI Application - Feature Specification

> **Version:** 1.1.0  
> **Last Updated:** 2025-11-28  
> **Status:** Comparison grid and histogram overlay shipped; export tooling, advanced preferences, and expanded comparison views remain in backlog.

## Overview

The Zero-DCE GUI is a cross-platform desktop client that exposes the Zero-Reference Deep Curve Estimation model through an approachable interface. It supports both single-image enhancement and multi-method comparisons against classical baselines while keeping the underlying TensorFlow/Keras stack hidden from end users. The application is currently focused on reliability and clarity for graduate-level coursework; future iterations will add richer comparison visualizations and export tooling.

### Target Platforms

- Windows 10 and 11
- Linux distributions with Qt 6 support (Ubuntu 20.04+, Arch 6.17 verified)
- macOS 10.15 Catalina or newer

### Technology Stack

- **Framework:** PyQt6 with QThread-based background work
- **Model Runtime:** TensorFlow/Keras 3 (backend forced to TensorFlow)
- **Image Processing:** Pillow, NumPy, OpenCV, and a Retinex helper for MSRCR
- **Classical Baselines:** `classical_methods.py` (AutoContrast, Histogram Equalization, CLAHE, Gamma Correction, MSRCR)
- **Settings Persistence:** Qt `QSettings`
- **Entry Point:** `gui_app.py`

---

## User Interface Design

### Overall Layout

```
+---------------------------------------------------------------------+
| File  Edit  View  Model  Help                             [_][ ][X] |
+---------------------------------------------------------------------+
|                                                                     |
|  +---------------+      +-----+      +---------------+              |
|  |               |      |  -> |      |               |              |
|  | Input Image   |      +-----+      | Enhanced      |              |
|  | (click/drag)  |   Compare Methods | Image         |              |
|  |               |                   | (click to save)|             |
|  +---------------+                   +---------------+              |
|                                                                     |
|  Status: Ready                                                      |
+---------------------------------------------------------------------+
| Model: zero_dce.weights.h5                               | Ready    |
+---------------------------------------------------------------------+
```

The main window stores and restores its geometry via `AppSettings`. Toggling comparison mode hides the single-mode widget and shows the comparison grid beneath the same menu bar and status bar.

---

## Core Features

### Menu Bar

#### File Menu
- **Open Image... (`Ctrl+O`)** launches a QFileDialog that filters `*.jpg`, `*.jpeg`, `*.png`, and `*.bmp`. Files are validated via `ImageProcessor.validate_image_file`, including size limits (64 px minimum, 8192 px maximum per side).
- **Recent Files** lists up to 10 entries from QSettings. Missing files stay listed but are disabled and annotated with `(not found)`. A `Clear Recent Files` action wipes the list.
- **Save Enhanced Image... (`Ctrl+S`)** is enabled only when an enhancement result exists. Default filenames follow `<stem>_enhanced.<ext>`. PNG and JPEG are supported; JPEG saves at quality 95 (current default stored in settings).
- **Exit (`Ctrl+Q`)** cleanly shuts down the app, persisting window geometry and unloading the model.

#### Edit Menu
- **Clear Input** clears both panels and resets in-memory images. This action also terminates comparison state and disables enhancement until a new image loads.
- **Preferences... (`Ctrl+,`)** opens the modal preferences dialog (see Section "Preferences Dialog").

#### View Menu
- **Comparison Mode (`C`)** is checkable and keeps the action state in sync with the visible layout. Enabling the mode invokes the Method Selection dialog; disabling it returns to single-mode widgets.
- **Show Histogram Overlay (`H`)** toggles histogram overlays across both single-mode panels and every comparison cell. The choice is persisted via `AppSettings`.
- **Histogram Type** submenu offers `Grayscale` and `RGB`. The actions are radio buttons, and `Shift+H` cycles between the two for quick access.

#### Model Menu
- **Load Model Weights...** prompts for `.h5` or `.weights.h5` files and validates them before loading through `ModelLoader`. Errors surface via `ErrorDialog`.
- **Default Weights** dynamically enumerates files in the configured weights directory. Entries show epochs when they can be parsed from filenames, and the currently loaded model is check-marked. Selecting an entry loads it and updates the default directory in settings.
- **Model Info** displays filename, full path, and file size for the currently loaded weights. If auto-load-on-startup is enabled, `MainWindow` attempts to load `AppSettings.get_full_model_path()` when the window is created and reports missing files with an error dialog.

#### Help Menu
- **About** uses `QMessageBox.about` to show application version `1.1.0`, the Zero-DCE paper citation, and course authorship. Dedicated help pages and keyboard shortcut summaries are tracked for future work.

### Display Modes

#### Single Enhancement Mode
- **Input Panel:** Accepts clicks to open images and drag-and-drop events for supported files. Displays placeholders when empty and shows a file info overlay (name, dimensions, size) when loaded.
- **Enhance Button:** A custom circular button that reflects disabled, ready, processing (hourglass), and completed (checkmark) states. It emits `enhance_clicked` only when both an image and a model are available.
- **Output Panel:** Mirrors the input panel styling, adds an "Enhanced in Xs" overlay once results are ready, and responds to clicks by opening the save dialog.
- **Compare Methods Button:** Lives below the enhance button, shares the same enablement state as the input panel, and is the quickest entry point to comparison mode.

#### Comparison Mode
- **Activation Paths:** View menu toggle, the `C` key, or the `Compare Methods` button. A warning dialog appears if no input image is loaded.
- **Persisted Selection:** The application remembers the last set of selected methods and the optional reference image path. When a new input image is loaded, results are cleared but the selection remains so the user can re-run quickly.
- **Automatic Re-run:** If comparison mode is active and the user loads another image, `_run_comparison` restarts immediately with the remembered methods.

##### Method Selection Dialog
- Grouped checkboxes for **Deep Learning** (`zero-dce`) and **Classical** methods (AutoContrast, Histogram Eq, CLAHE, Gamma, MSRCR) using metadata from `EnhancementMethodRegistry`.
- Disabled state for Zero-DCE when no model is loaded, with tooltips explaining the requirement.
- Quick selection buttons: Select All, Select None, Classical Only, and Fast Methods (leveraging the registry's `ExecutionSpeed` values, currently the classical set).
- An optional reference image picker with Browse and Clear buttons; the chosen path is echoed in a read-only line edit.
- Validation enforces at least one method before closing with OK.

##### Comparison Grid Behavior
- The grid lives inside a scroll area and dynamically chooses 1 to 4 columns based on the total number of cells (input, optional reference, and each method).
- Each `ComparisonCell` shows the method name, the current pixmap, status text, optional timing, and a color-coded border (`gray` pending, `blue` running, `green` done, `red` error).
- Input and reference cells share the same widget class but mark themselves as reference-only, showing "Original" or "Reference" labels.
- Histogram overlays propagate to every cell when enabled in the View menu or preferences. Users can drag the overlay to reposition it per cell.
- Cells emit a `clicked` signal; the main window currently uses it to show timing in the status bar, laying groundwork for a future expanded view.
- Background processing: `EnhancementRunnerThread` runs methods sequentially, emits start/complete/failure signals, and ensures that one failure does not cancel the remaining queue. Status bar messages track `Running {method} (X/Y)` and `Completed {method} (X/Y)`.
- Results are cached in `_enhancement_results` for possible future reuse and to power the status messages.

---

## Image Display Panels (Single Mode)

### Input Panel
- **Empty State:** Shows centered placeholder text, dashed border, and accepts drag-and-drop.
- **Loaded State:** Displays the pixmap scaled with aspect ratio preservation, updates border styling, and shows the info overlay if enabled.
- **Interactions:** Left-click triggers the open dialog; right-click offers "Open Different Image" and "Clear". Clearing fires a signal so the main window can reset state.

### Output Panel
- **Empty State:** Placeholder prompting the user to run enhancement.
- **Processing State:** Displays "Enhancing..." with accent-colored border and disables click actions.
- **Enhanced State:** Shows the result, displays the enhancement time, and enables the info overlay even when the image only exists in memory.
- **Interactions:** Left-click opens the save dialog; right-click offers "Save Image" and "Clear".

Both panels expose histogram overlays and info overlays controlled by preferences.

---

## Histogram Overlay

- Controlled through both the View menu (`Show Histogram Overlay`, `Histogram Type`) and preferences. Keyboard: `H` toggles visibility, `Shift+H` cycles the type.
- Implemented via the shared `HistogramOverlay` widget. It renders either a normalized grayscale curve or RGB channel curves, is draggable within the image label, and remembers custom positions per widget until the image changes.
- Available on input panel, output panel, and every comparison cell. When disabled globally, overlays hide themselves automatically.

---

## Enhancement Controls & Workflow

1. **Pre-flight checks:** `_update_ui_state` ensures the enhance button is only ready when an image and model are both loaded. The compare button only requires an image.
2. **Trigger:** Clicking the enhance button or pressing `Ctrl+E` spawns an `EnhancementWorker` QThread. The worker keeps a reference to the PIL image, the currently loaded `ZeroDCE` instance, and the original size.
3. **Processing:** `ImageProcessor.enhance_image` performs preprocessing (normalization), executes `model.call`, post-processes back to a PIL image, and resizes to the original dimensions if needed.
4. **Completion:** The worker emits a PIL image back to the main thread. The result is stored, converted to a QPixmap, displayed in the output panel, and wrapped in an `EnhancementResult` to capture timing. The status bar reports `Enhanced successfully in {time}`.
5. **Error handling:** Failures route through `ErrorDialog.show_error` with suggestions. The UI reverts to the ready state, and status text shows `Enhancement failed`.
6. **Save:** The save action and output panel click share `_save_enhanced_image`, which respects the user's preferred JPEG quality from settings (currently default 95).

---

## Preferences Dialog

- **General Tab:** 
  - *Model settings:* weights directory selector and an auto-load checkbox.
  - *Display settings:* default zoom (Fit or Actual), synchronized zoom toggle (future use), info overlay visibility, histogram overlay toggle, and histogram type picker.
  - *Performance settings:* GPU mode (Auto/Enable/Disable) with live TensorFlow device detection feedback, and maximum image dimension options (2048, 4096 default, 8192, Unlimited). Choosing Unlimited shows a warning label.
- **Buttons:** OK saves and closes, Apply saves without closing and confirms via an information dialog, Cancel/Esc respects unsaved changes prompts, and Alt+F4 is intercepted to provide the same warning.
- **Settings propagation:** Saving emits `settings_changed`, prompting `MainWindow` to reload settings, refresh the default weights menu, and reapply histogram choices instantly.
- **Advanced Tab:** Still planned. Items such as iteration counts, cache management, export format defaults, and debug logging remain in the backlog.

---

## Status Bar

- **Left side:** transient messages such as `Ready`, `Loaded: image.jpg`, `Enhancing image...`, `Comparing 4 methods... (0/4)`, `Comparison complete: 3/4 methods succeeded`, and error hints.
- **Right side:** persistent label showing `Model: <filename>` or `No model loaded`. Updates occur on every model load/unload and after preference changes.

---

## Technical Specifications

### Image Handling
- Loading uses Pillow with enforced conversion to RGB. Min dimension 64 px, max 8192 px (hard validation) plus an optional downscale step to honor the user's `max_image_dimension` preference (default 4096 px).
- Save supports PNG (lossless) and JPEG with adjustable quality. EXIF preservation logic exists in `ImageProcessor.save_image` but is not yet exposed in the UI.
- Info overlays show filename, resolution, file size, and enhancement time when available.

### Model Integration
- `ModelLoader` enforces `.h5` or `.weights.h5` extensions, resolves absolute paths, and instantiates `ZeroDCE` before loading weights. Auto-load on startup uses the configured weights directory plus default filename.
- Models are unloaded on exit to free GPU memory and clear the Keras backend session.
- Inference for single runs and Zero-DCE comparison methods shares the same preprocessing and post-processing pipeline to guarantee consistent results.

### Comparison Infrastructure
- `EnhancementMethodRegistry` registers `zero-dce` plus the classical methods defined in `classical_methods.py`. Each entry declares whether it needs a model and its qualitative speed.
- `EnhancementRunner` executes methods sequentially, timing each and capturing results via `EnhancementResult`. `EnhancementRunnerThread` wraps the runner in a QThread so UI updates happen through Qt signals.
- Classical methods run entirely on CPU using Pillow and OpenCV; Zero-DCE uses the loaded Keras model.
- Reference images are loaded with Pillow; failures prompt a warning but do not cancel the comparison run.

### Settings Persistence
- `AppSettings` uses the organization/application ID `ZeroDCE/GUI` and stores: weights directory, default model filename, auto load flag, recent files, window geometry, zoom defaults, info overlay preference, histogram visibility/type, GPU mode, max image dimension, output preferences (format, JPEG quality), cache settings (future), and logging toggles (future).
- Changes are synced immediately when preferences are saved or when view menu toggles write back to settings.

### Error Handling & Messaging
- `ErrorDialog` centralizes message formatting, with dedicated helpers for model, image, file, and memory errors. Dialogs show a friendly summary, optional solution, and expandable technical details with a copy-to-clipboard button.
- Model auto-load failures during startup produce a blocking error dialog that explains the expected path and how to fix the configuration.
- Reference image load errors use `QMessageBox.warning` so the comparison run can continue with the remaining cells.

---

## User Workflows

### Basic Enhancement Workflow
1. Launch `gui_app.py`.
2. Load an image via the left panel or File -> Open.
3. Load weights if auto-load did not find the model (Model -> Load Model Weights).
4. Click the enhance button or press `Ctrl+E`.
5. Wait for the output panel to show the enhanced image and time overlay.
6. Save the result by clicking the output panel or choosing File -> Save Enhanced Image.

### Multi-Method Comparison Workflow
1. Load an input image as above.
2. Ensure a Zero-DCE model is loaded if that method should be included.
3. Enter comparison mode via the View menu, the `C` key, or the Compare Methods button.
4. Pick the desired methods and optional reference image in the dialog.
5. Watch the comparison grid fill in; histogram overlays can be toggled at any time with `H`.
6. Review timing and visual quality per cell. Click cells to surface their timing in the status bar.
7. Press `C` again or uncheck View -> Comparison Mode to return to single mode. Results persist until you clear the input or start a new comparison.

---

## Keyboard & Accessibility

- `Ctrl+O`: open image dialog.
- `Ctrl+S`: save enhanced image.
- `Ctrl+E`: trigger enhancement when ready.
- `Ctrl+Q`: exit.
- `Ctrl+,`: open preferences.
- `C`: toggle comparison mode.
- `H`: toggle histogram overlay; `Shift+H`: cycle histogram type.
- `Space`: activate the enhance button when focused.
- Standard Qt navigation (Tab/Shift+Tab) works across controls. Dialogs respect `Esc` to cancel (with unsaved prompts) and `Enter` to activate default buttons.
- High DPI support is enabled via `QApplication.setHighDpiScaleFactorRoundingPolicy`. Screen reader support relies on Qt's default accessibility metadata; additional labeling work is planned for future releases.

---

## Non-Functional Requirements

- **Startup:** <3 seconds on a typical development laptop once dependencies are installed.
- **Model Load:** <2 seconds for standard `.weights.h5` files residing on SSD storage.
- **Enhancement Latency:** <5 seconds for 1080p images on GPU, <12 seconds on CPU (Zero-DCE).
- **Responsiveness:** All heavy work happens in background threads, so the UI remains reactive during enhancement and comparison runs.
- **Stability:** The app should gracefully handle missing files, invalid images, and long comparisons without crashes.

---

## Testing Notes

Automated tests focus on unit-level behavior inside `tests/`:

- `tests/test_gui_recent_files.py` verifies menu population and clearing logic.
- `tests/test_gui_model_menu.py` covers default weights listing and checkmark behavior.
- `tests/test_gui_model_settings.py` validates AppSettings defaults and setters.
- `tests/test_gui_dialogs_preferences.py` exercises the preferences dialog interactions and unsaved changes prompts.
- `tests/test_gui_widgets_comparison_cell.py` and `tests/test_gui_widgets_comparison_grid.py` assert layout, status transitions, and timing labels for comparison widgets.
- `tests/test_gui_dialogs_method_selection.py` checks quick selection buttons, disabled states, and validation rules.
- `tests/test_gui_utils_enhancement_methods.py` ensures the registry metadata and execution helpers behave as expected.
- `tests/test_gui_utils_enhancement_runner.py` covers sequential execution, timing capture, and error propagation.
- `tests/test_gui_image_panel_timing.py` validates info overlay timing updates.

Run the suite with `source .venv/bin/activate && pytest -m gui` or run individual files as needed. Manual smoke tests should cover the full enhancement workflow, comparison runs with and without Zero-DCE, histogram toggling, and preference persistence.

---

## File Structure

```
repo/
├── gui_app.py
├── gui/
│   ├── main_window.py
│   ├── dialogs/
│   │   ├── error_dialog.py
│   │   ├── method_selection_dialog.py
│   │   └── preferences_dialog.py
│   ├── widgets/
│   │   ├── image_panel.py
│   │   ├── enhance_button.py
│   │   ├── comparison_cell.py
│   │   ├── comparison_grid.py
│   │   └── histogram_overlay.py
│   └── utils/
│       ├── settings.py
│       ├── model_loader.py
│       ├── image_processor.py
│       ├── enhancement_result.py
│       ├── enhancement_methods.py
│       └── enhancement_runner.py
├── docs/
│   ├── GUI_SPECIFICATION.md   (this document)
│   └── GUI_USER_GUIDE.md
├── classical_methods.py
├── retinex.py
├── zero_dce.py
└── tests/
    └── test_gui_*.py (see Testing Notes)
```

---

## Future Enhancements

1. **Export comparison grid results** (Task 3.5) including stitched images and CSV timing summaries.
2. **Method presets and preferences** (Task 3.4) so users can pin favorite combinations and default references.
3. **Expanded comparison views**, including per-cell before/after sliders and zoom-synchronized inspectors.
4. **Advanced preferences tab** for caching, iteration counts, default output format, and logging toggles.
5. **Cancelable background work** so long comparison runs can be interrupted mid-queue.
6. **Help menu additions** for keyboard shortcuts and Zero-DCE background material.
7. **Accessibility polishing**: explicit focus indicators, larger default fonts, and screen reader descriptions.

---

## Appendix

### References

- Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (CVPR 2020) (https://arxiv.org/abs/2001.06826)
- PyQt6 Documentation (https://doc.qt.io/qtforpython-6/)
- Keras Zero-DCE example (https://keras.io/examples/vision/zero_dce/)

### Glossary

- **Zero-DCE:** Zero-Reference Deep Curve Estimation model implemented in `model.py`.
- **DCE-Net:** The convolutional backbone producing per-pixel curve parameters.
- **Enhancement Result:** Container object storing an enhanced PIL image plus metadata.
- **Comparison Cell:** GUI widget representing one method's output within the grid.

### Changelog

- **v1.1.0 (2025-11-28):** Document realigned with current implementation (comparison grid, histogram overlay, method selection dialog, tested modules) and removed references to unimplemented sliders/export features.
- **v1.0.0 (2025-10-30):** Phase 2 polish complete (menus, preferences dialog, status bar, keyboard shortcuts).
- **v0.9.0 (2025-10-28):** Initial specification drafted for the Phase 1 MVP.
