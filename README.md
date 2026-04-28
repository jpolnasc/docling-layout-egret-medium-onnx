# docling-layout-egret-medium — ONNX Inference

Document layout detection using the [docling-layout-egret-medium](https://huggingface.co/docling-project/docling-layout-egret-medium) model exported to ONNX.

Detects 17 layout classes: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title, Document Index, Code, Checkbox-Selected, Checkbox-Unselected, Form, Key-Value Region.

No PyTorch or Transformers required at runtime.

---

## Repository contents

```
infer_image.py                                    # inference script
requirements.txt                                  # runtime dependencies (no torch)
onnx_output/
    docling_layout_egret_medium_simplified.onnx   # exported model (~75MB)
    id2label.json                                 # label index -> class name
    processor/
        preprocessor_config.json                  # processor metadata (for reference)
```

To export the model yourself from the original HuggingFace weights, see [EXPORT.md](EXPORT.md).

---

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running inference

### Score-based detection

```bash
python infer_image.py document.png --threshold 0.4 --vis result.png
```

Save detections as JSON:

```bash
python infer_image.py document.png --threshold 0.4 --output detections.json
```

### Density-based detection (Picture + Table)

Uses query density heatmaps to detect figures and tables even when individual query scores are low. Results are saved automatically to `density_output/<stem>_density.png`.

```bash
python infer_image.py document.png --density-vis
```

Tune sensitivity with `--density-min-peak` (default 1.0 — lower catches more, higher is stricter):

```bash
python infer_image.py document.png --density-vis --density-min-peak 0.8
```

### Non-text mask

Paints detected Picture and Table regions in orange. Useful for pipelines that pass the image to Tesseract or similar OCR — mask out non-text first, then run OCR on the rest.
Output is saved to `density_output/<stem>_masked.png`.

```bash
python infer_image.py document.png --mask-vis
```

### All options at once

```bash
python infer_image.py document.png \
    --threshold 0.4 \
    --vis result.png \
    --output detections.json \
    --density-vis \
    --mask-vis \
    --density-min-peak 1.0
```

---

## Image sizing

Images are automatically resized so the longest side falls between `MIN_SIDE` (default 1024) and `MAX_SIDE` (default 1600), aligned to the nearest multiple of 32 required by the encoder. Bounding boxes are always returned in the original image coordinate space.

Configure via a `.env` file:

```
MIN_SIDE=1024
MAX_SIDE=1600
SCORE_THRESHOLD=0.6
DEVICE=cpu
ONNX_OUTPUT_DIR=./onnx_output
```

---

## Using as a library

```python
from PIL import Image
from infer_image import load_session, infer, filter_results, density_detect, mask_non_text

sess = load_session("onnx_output/docling_layout_egret_medium_simplified.onnx")
image = Image.open("document.png").convert("RGB")

# All 300 decoder queries, unfiltered
all_results = infer(sess, image)

# Filter by score
detections = filter_results(all_results, threshold=0.4)

# Density-based figure/table detection (label 6 = Picture, 8 = Table)
figures = density_detect(all_results, label_id=6, img_w=image.width, img_h=image.height)
figures = [d for d in figures if d["density_peak"] >= 1.0]

# Non-text mask overlay
masked = mask_non_text(image, all_results, min_peak=1.0)
masked.save("masked.png")
```
