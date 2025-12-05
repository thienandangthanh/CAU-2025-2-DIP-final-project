# Image Comparison Usage Guide

## Quick Start

The `compare.py` script now automatically arranges images in a multi-row grid layout for better readability in reports and presentations.

## Basic Usage

### Compare All Methods (Recommended for Reports)

```bash
python compare.py \
  -i ./lol_dataset/eval15/low/1.png \
  -w ./weights/zero_dce.weights.h5 \
  -r ./lol_dataset/eval15/high/1.png \
  -o comparison_all_methods.png
```

**Output:** 2x4 grid with 8 images (auto-calculated):
- Row 1: Original, Reference, Zero-DCE, AutoContrast
- Row 2: Histogram Eq, CLAHE, Gamma Correction, MSRCR

**File Size:** ~4.8 MB (high quality, 150 DPI)

### Compare with Custom Grid Layout

```bash
# Use 3 columns instead of auto-calculated 4
python compare.py \
  -i ./lol_dataset/eval15/low/1.png \
  -w ./weights/zero_dce.weights.h5 \
  -r ./lol_dataset/eval15/high/1.png \
  --columns 3 \
  -o comparison_3cols.png
```

**Output:** 3x3 grid (3 rows, 3 columns) with 8 images:
- Row 1: Original, Reference, Zero-DCE
- Row 2: AutoContrast, Histogram Eq, CLAHE
- Row 3: Gamma Correction, MSRCR, (empty slot)

### Compare with Narrow Layout

```bash
# Use 2 columns for a narrower, taller layout
python compare.py \
  -i ./lol_dataset/eval15/low/1.png \
  -w ./weights/zero_dce.weights.h5 \
  -r ./lol_dataset/eval15/high/1.png \
  --columns 2 \
  -o comparison_2cols.png
```

**Output:** 4x2 grid (4 rows, 2 columns) with 8 images - ideal for portrait-oriented reports

### Compare Specific Methods

```bash
python compare.py \
  -i ./lol_dataset/eval15/low/1.png \
  -w ./weights/zero_dce.weights.h5 \
  --methods zero-dce autocontrast clahe \
  -o comparison_selected.png
```

**Output:** 2x2 grid with 4 images:
- Row 1: Original, Zero-DCE
- Row 2: AutoContrast, CLAHE

### Without Reference Image

```bash
python compare.py \
  -i ./lol_dataset/eval15/low/1.png \
  -w ./weights/zero_dce.weights.h5 \
  -o comparison_no_ref.png
```

**Output:** 2x4 grid with 7 images (Original + 6 methods)

### Save Individual Enhanced Images

```bash
python compare.py \
  -i ./lol_dataset/eval15/low/1.png \
  -w ./weights/zero_dce.weights.h5 \
  -o comparison.png \
  --save-individual
```

**Output:**
- `comparison.png` - Grid comparison image
- `individual/` directory with separate enhanced images:
  - `1_zero_dce.png`
  - `1_autocontrast.png`
  - `1_histogram_eq.png`
  - `1_clahe.png`
  - `1_gamma_correction.png`
  - `1_msrcr.png`

## Grid Layout Examples

### Auto-Calculated Layouts (Default)

The script automatically determines the optimal grid layout based on the number of images:

| Images | Grid Layout | Example |
|--------|-------------|---------|
| 2 | 1x2 | Original + Zero-DCE |
| 3 | 1x3 | Original + 2 methods |
| 4 | 2x2 | Original + 3 methods |
| 5 | 2x3 | Original + 4 methods |
| 6 | 2x3 | Original + 5 methods |
| 7 | 2x4 | Original + 6 methods |
| 8 | 2x4 | Original + Reference + 6 methods |

**Maximum Columns:** 4 (prevents images from being too small)

### Custom Column Layouts

You can override the auto-calculation with `--columns`:

| Command | Images | Result | Use Case |
|---------|--------|--------|----------|
| `--columns 2` | 8 | 4x2 grid | Narrow layout for portrait reports |
| `--columns 3` | 8 | 3x3 grid | Balanced square layout |
| `--columns 4` | 8 | 2x4 grid | Wide layout (same as auto) |
| `--columns 1` | 4 | 4x1 grid | Vertical stack for detailed viewing |
| `--columns 5` | 8 | 2x5 grid | Very wide layout (use with caution) |

**Tip:** Rows grow automatically based on: `rows = ⌈images / columns⌉`

## Available Enhancement Methods

Use these keys with the `--methods` flag:

| Method Key | Full Name | Description |
|------------|-----------|-------------|
| `zero-dce` | Zero-DCE | Deep learning-based curve estimation |
| `autocontrast` | AutoContrast | Global contrast normalization |
| `histogram-eq` | Histogram Eq | Histogram equalization on YUV |
| `clahe` | CLAHE | Contrast Limited Adaptive Histogram Eq |
| `gamma` | Gamma Correction | Power-law transformation (gamma=2.2) |
| `msrcr` | MSRCR | Multi-Scale Retinex with Color Restoration |

## CLI Options

```
-i, --input PATH          Input low-light image (required)
-w, --weights PATH        Zero-DCE model weights (required)
-o, --output PATH         Output comparison image path
-r, --reference PATH      Reference (well-exposed) image for comparison
--methods METHOD [...]    Enhancement methods to compare (default: all)
-c, --columns INT         Number of columns in grid (default: auto-calculate)
--save-individual         Save individual enhanced images
```

## When to Use Custom Columns

### Use Auto-Calculate (Default) When:
- ✅ You want the most balanced, readable layout
- ✅ You're generating comparisons quickly
- ✅ You trust the algorithm to optimize for you
- ✅ Most common use case

### Use Custom Columns When:
- 📐 **Paper Size Constraints**: Need to fit specific page dimensions
  - Portrait orientation → Use `--columns 2` or `--columns 3`
  - Landscape orientation → Use `--columns 4` or `--columns 5`
- 📊 **Presentation Requirements**: Match your slide format
  - Square slides → Use `--columns 3` for balanced appearance
  - Wide slides → Use `--columns 4` for maximum width usage
- 🎯 **Specific Comparisons**: Emphasize certain groupings
  - Compare pairs → Use `--columns 2`
  - Compare triads → Use `--columns 3`
- 📱 **Different Viewing Contexts**:
  - Mobile/tablet viewing → Use `--columns 2` (narrower)
  - Desktop/projector → Use `--columns 4` (wider)

### Examples by Use Case

**Academic Paper (2-column format)**
```bash
# Narrow layout fits better in 2-column papers
python compare.py -i input.png -w weights.h5 --columns 2 -o fig1.png
```

**Presentation Slide (16:9 aspect ratio)**
```bash
# Wide layout maximizes slide space
python compare.py -i input.png -w weights.h5 --columns 4 -o slide1.png
```

**Technical Report (single column)**
```bash
# Balanced square layout
python compare.py -i input.png -w weights.h5 --columns 3 -o report_fig.png
```

**Detailed Analysis (vertical comparison)**
```bash
# Stack vertically for detailed side-by-side viewing
python compare.py -i input.png -w weights.h5 --columns 1 -o detailed.png
```

## Tips for Report Figures

### High-Quality Output
- Default DPI: 150 (publication quality)
- Figure size: 5 inches per image cell
- Format: PNG (lossless compression)

### For Academic Papers

1. **Full Comparison (All Methods)**
   ```bash
   python compare.py -i input.png -w weights.h5 -r reference.png -o fig1_comparison.png
   ```
   Use for: Main results figure showing all enhancement approaches

2. **Selected Methods (Key Comparisons)**
   ```bash
   python compare.py -i input.png -w weights.h5 \
     --methods zero-dce autocontrast clahe msrcr \
     -o fig2_selected.png
   ```
   Use for: Focused comparison with specific baselines

3. **With Individual Images**
   ```bash
   python compare.py -i input.png -w weights.h5 -o comparison.png --save-individual
   ```
   Use for: When you need separate high-res images for detailed analysis

### For Presentations

Use fewer methods for clarity:
```bash
python compare.py -i input.png -w weights.h5 \
  --methods zero-dce autocontrast clahe \
  -o slide_comparison.png
```

**Result:** Clean 2x2 grid that's easy to read on slides

## Layout vs. Old Version

### Before (Old Single-Row Layout)
```
❌ Problem: Too wide horizontally
[Img1] [Img2] [Img3] [Img4] [Img5] [Img6] [Img7] [Img8]
Width: 40 inches (8 × 5 inches) - doesn't fit on standard paper
```

### After (New Multi-Row Grid)
```
✅ Solution: Balanced grid layout
[Img1] [Img2] [Img3] [Img4]
[Img5] [Img6] [Img7] [Img8]
Size: 20 inches wide × 10 inches tall - fits on A4/Letter paper
```

## Example Workflow for Final Report

```bash
# Step 1: Generate comparison for multiple test images
for i in 1 2 3 4 5; do
    python compare.py \
        -i ./lol_dataset/eval15/low/$i.png \
        -w ./weights/zero_dce.weights.h5 \
        -r ./lol_dataset/eval15/high/$i.png \
        -o ./report_figures/comparison_$i.png
done

# Step 2: Generate selected method comparison
python compare.py \
    -i ./lol_dataset/eval15/low/1.png \
    -w ./weights/zero_dce.weights.h5 \
    --methods zero-dce msrcr \
    -o ./report_figures/comparison_best_methods.png

# Step 3: Save individual images for detailed analysis
python compare.py \
    -i ./lol_dataset/eval15/low/1.png \
    -w ./weights/zero_dce.weights.h5 \
    -o ./report_figures/comparison_with_individuals.png \
    --save-individual
```

## GUI Comparison Mode

The GUI application uses the same grid layout logic:

1. Open an image in the GUI
2. Click "Comparison Mode" button
3. Select enhancement methods via the dialog
4. Grid automatically adjusts based on number of methods

**Consistency:** CLI and GUI use identical layout calculations from `layout_utils.py`

## Troubleshooting

### Images are too small
Try comparing fewer methods at once:
```bash
python compare.py -i input.png -w weights.h5 --methods zero-dce autocontrast -o output.png
```

### Need custom layout
The grid layout is optimized for readability. If you need a different arrangement, you can:
1. Save individual images with `--save-individual`
2. Arrange them manually using image editing software

### File size too large
The default 150 DPI produces high-quality images (~5MB). This is intentional for publication quality.

## Technical Details

### Grid Layout Algorithm

Implemented in `layout_utils.py`:

```python
from layout_utils import calculate_optimal_grid_layout

# Calculate layout for 8 images
rows, cols = calculate_optimal_grid_layout(8)
# Returns: (2, 4) -> 2 rows, 4 columns
```

### Matplotlib Figure Size

```python
fig_width = ncols * 5   # 5 inches per image
fig_height = nrows * 5  # 5 inches per image
```

For 8 images (2x4 grid):
- Width: 4 × 5 = 20 inches
- Height: 2 × 5 = 10 inches

### DPI and Resolution

- Default DPI: 150
- Pixel dimensions (8 images): 3000 × 1500 pixels
- File size: ~4-5 MB (PNG, lossless)

## Related Documentation

- `LAYOUT_REFACTOR_SUMMARY.md` - Technical details of the grid layout refactoring
- `README.md` - Project overview and setup
- `AGENTS.md` - Development guidelines

## Getting Help

```bash
# Show all CLI options
python compare.py --help

# Test the script
python test_compare.py
```

## Citation

If you use this comparison tool in your research, please cite the Zero-DCE paper:

```bibtex
@inproceedings{guo2020zero,
  title={Zero-reference deep curve estimation for low-light image enhancement},
  author={Guo, Chunle and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1780--1789},
  year={2020}
}
```
