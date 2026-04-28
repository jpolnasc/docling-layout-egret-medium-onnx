"""
Run layout detection on a real image using the exported ONNX model.

Usage:
    python infer_image.py <image_path> [--threshold 0.6] [--output detections.json]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

ONNX_OUTPUT_DIR = Path(os.getenv("ONNX_OUTPUT_DIR", "./onnx_output"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.6))
DEVICE = os.getenv("DEVICE", "cpu")

# Resize bounds (applied to the longest side, aspect ratio preserved)
MIN_SIDE = int(os.getenv("MIN_SIDE", 1024))
MAX_SIDE = int(os.getenv("MAX_SIDE", 1600))


_simplified = ONNX_OUTPUT_DIR / "docling_layout_egret_medium_simplified.onnx"
_base = ONNX_OUTPUT_DIR / "docling_layout_egret_medium.onnx"
DEFAULT_ONNX = _simplified if _simplified.exists() else _base


def load_session(onnx_path):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if DEVICE == "cuda"
        else ["CPUExecutionProvider"]
    )
    return ort.InferenceSession(str(onnx_path), providers=providers)


def preprocess(image: Image.Image) -> np.ndarray:
    """PIL Image -> (1, 3, H, W) float32 ready for ONNX. Matches RTDetrImageProcessor (rescale only, no normalize)."""
    arr = np.array(image, dtype=np.float32) / 255.0   # HWC [0,1]
    arr = arr.transpose(2, 0, 1)[np.newaxis]           # -> (1,3,H,W)
    return arr


def postprocess(logits: np.ndarray, pred_boxes: np.ndarray,
                img_h: int, img_w: int) -> dict:
    """
    logits:     (1, num_queries, num_classes)
    pred_boxes: (1, num_queries, 4)  cxcywh normalised [0,1]
    Returns dict with keys scores, labels, boxes (all numpy arrays).
    Uses focal-loss sigmoid + topk, matching RTDetrImageProcessor behaviour.
    """
    logits    = logits[0]       # (Q, C)
    pred_boxes = pred_boxes[0]  # (Q, 4)

    # cxcywh -> xyxy, then scale to pixel coords
    cx, cy, w, h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes *= np.array([img_w, img_h, img_w, img_h], dtype=np.float32)

    # sigmoid scores, topk
    scores_all = 1.0 / (1.0 + np.exp(-logits))               # (Q, C)
    num_queries, num_classes = scores_all.shape
    flat = scores_all.ravel()                                  # Q*C
    topk_idx = np.argpartition(flat, -num_queries)[-num_queries:]
    topk_idx = topk_idx[np.argsort(flat[topk_idx])[::-1]]

    scores = flat[topk_idx]
    labels = topk_idx % num_classes
    query_idx = topk_idx // num_classes
    boxes = boxes[query_idx]

    return {"scores": scores, "labels": labels, "boxes": boxes}


def normalize_size(image: Image.Image, min_side: int = MIN_SIDE, max_side: int = MAX_SIDE) -> Image.Image:
    """Resize so the longest side is within [min_side, max_side] and both dims are multiples of 32."""
    w, h = image.size
    longest = max(w, h)

    if longest < min_side:
        scale = min_side / longest
    elif longest > max_side:
        scale = max_side / longest
    else:
        scale = 1.0

    # Round to nearest multiple of 32 — required by the encoder's multi-scale concat
    new_w = max(32, round(w * scale / 32) * 32)
    new_h = max(32, round(h * scale / 32) * 32)

    if (new_w, new_h) != (w, h):
        print(f"  Resizing image: {w}x{h} -> {new_w}x{new_h} (scale={scale:.3f}, aligned to 32)")
    return image.resize((new_w, new_h), Image.LANCZOS)


def infer(sess, image: Image.Image) -> dict:
    """Run inference and return ALL detections (no threshold applied)."""
    resized = normalize_size(image)
    pixel_values = preprocess(resized)

    input_name = sess.get_inputs()[0].name
    logits_np, boxes_np = sess.run(None, {input_name: pixel_values})

    results = postprocess(logits_np, boxes_np, resized.height, resized.width)

    # Scale boxes from resized space back to original image coordinates
    orig_w, orig_h = image.size
    scale_x = orig_w / resized.width
    scale_y = orig_h / resized.height
    if len(results["boxes"]) > 0:
        results["boxes"][:, [0, 2]] *= scale_x
        results["boxes"][:, [1, 3]] *= scale_y

    return results


def filter_results(results: dict, threshold: float) -> dict:
    mask = results["scores"] >= threshold
    return {
        "scores": results["scores"][mask],
        "labels": results["labels"][mask],
        "boxes":  results["boxes"][mask],
    }


def density_detect(results, label_id: int, img_w: int, img_h: int,
                   heatmap_scale: float = 0.1, min_density: float = 0.15) -> list[dict]:
    """
    Detect regions of a given label using query density instead of score threshold.

    Each query votes on the heatmap proportionally to its score. Connected regions
    above min_density (fraction of the peak) are returned as bounding boxes.

    Returns list of dicts: {box, score, density_peak}
    """
    from scipy.ndimage import label as scipy_label, gaussian_filter

    mask = results["labels"] == label_id
    boxes = results["boxes"][mask]
    scores = results["scores"][mask]

    if len(boxes) == 0:
        return []

    # Build heatmap at reduced resolution
    map_w = max(1, int(img_w * heatmap_scale))
    map_h = max(1, int(img_h * heatmap_scale))
    heatmap = np.zeros((map_h, map_w), dtype=np.float32)

    for (x0, y0, x1, y1), score in zip(boxes, scores):
        # Clamp to image bounds
        mx0 = max(0, int(x0 * heatmap_scale))
        my0 = max(0, int(y0 * heatmap_scale))
        mx1 = min(map_w, int(x1 * heatmap_scale))
        my1 = min(map_h, int(y1 * heatmap_scale))
        if mx1 > mx0 and my1 > my0:
            heatmap[my0:my1, mx0:mx1] += score

    # Smooth and threshold
    heatmap = gaussian_filter(heatmap, sigma=2.0)
    threshold_val = heatmap.max() * min_density
    binary = heatmap >= threshold_val

    # Find connected components
    labeled, n_components = scipy_label(binary)
    detections = []
    for comp_id in range(1, n_components + 1):
        region = labeled == comp_id
        ys, xs = np.where(region)
        if len(xs) == 0:
            continue

        # Bounding box in original image coords
        x0_orig = int(xs.min() / heatmap_scale)
        y0_orig = int(ys.min() / heatmap_scale)
        x1_orig = int(xs.max() / heatmap_scale)
        y1_orig = int(ys.max() / heatmap_scale)

        # Best score among queries whose center falls inside this region
        cx = ((boxes[:, 0] + boxes[:, 2]) / 2 * heatmap_scale).astype(int)
        cy = ((boxes[:, 1] + boxes[:, 3]) / 2 * heatmap_scale).astype(int)
        inside = (
            (cx >= xs.min()) & (cx <= xs.max()) &
            (cy >= ys.min()) & (cy <= ys.max())
        )
        best_score = float(scores[inside].max()) if inside.any() else 0.0

        detections.append({
            "box": [x0_orig, y0_orig, x1_orig, y1_orig],
            "score": best_score,
            "density_peak": float(heatmap[region].max()),
        })

    # Sort by best score descending
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def mask_non_text(image: Image.Image, all_results, min_peak: float = 1.0,
                  fill_color: tuple = (255, 165, 0), alpha: float = 0.5) -> Image.Image:
    """
    Returns a copy of the image with non-text regions (Picture + Table) painted over.
    Uses density detection for both labels. Useful for visualising what goes to Tesseract.
    fill_color: RGB tuple for the mask overlay (default orange)
    alpha: opacity of the overlay (0=transparent, 1=opaque)
    """
    img_w, img_h = image.size
    overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    fill_rgba = (*fill_color, int(255 * alpha))

    for label_id in [6, 8]:  # Picture, Table
        regions = density_detect(all_results, label_id, img_w, img_h)
        for d in regions:
            if d["density_peak"] < min_peak:
                continue
            x0, y0, x1, y1 = d["box"]
            draw.rectangle([x0, y0, x1, y1], fill=fill_rgba)

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def draw_detections(image: Image.Image, results, id2label):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        x0, y0, x1, y1 = [int(v) for v in box.tolist()]
        label = id2label.get(int(label_id), str(int(label_id)))
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, max(0, y0 - 16)), f"{label} {score:.2f}", fill="red", font=font)

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--output", default=None, help="Save detections to JSON file")
    parser.add_argument("--vis", default=None, help="Save visualisation to image file")
    parser.add_argument("--density-vis", default=None, help="Save density-based detection to image file (auto-saved to density_output/)")
    parser.add_argument("--density-label", type=int, default=None, help="Label ID for density detection (default: 6=Picture, 8=Table). Use multiple times for both.")
    parser.add_argument("--density-min-peak", type=float, default=1.0, help="Minimum density_peak to keep a detection (default: 1.0)")
    parser.add_argument("--mask-vis", action="store_true", help="Save non-text mask overlay to density_output/<stem>_masked.png")
    args = parser.parse_args()

    with open(ONNX_OUTPUT_DIR / "id2label.json") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    sess = load_session(DEFAULT_ONNX)

    image = Image.open(args.image).convert("RGB")
    all_results = infer(sess, image)
    results = filter_results(all_results, args.threshold)

    detections = []
    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        label = id2label.get(int(label_id), str(int(label_id)))
        det = {
            "label": label,
            "score": round(float(score), 4),
            "box": [round(float(v), 1) for v in box.tolist()],
        }
        detections.append(det)
        print(f"{label:30s}  score={score:.3f}  box={box.tolist()}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(detections, f, indent=2)
        print(f"\nDetections saved to {args.output}")

    if args.vis:
        vis = draw_detections(image.copy(), results, id2label)
        vis.save(args.vis)
        print(f"Visualisation saved to {args.vis}")

    if args.density_vis:
        # Output directory for density results
        density_dir = Path("density_output")
        density_dir.mkdir(exist_ok=True)

        img_stem = Path(args.image).stem
        out_path = density_dir / f"{img_stem}_density.png"

        # Labels to detect: default Picture(6) + Table(8), or user-specified
        LABEL_COLORS = {6: ("blue", "Picture"), 8: ("green", "Table")}
        labels_to_detect = [args.density_label] if args.density_label is not None else [6, 8]

        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except Exception:
            font = ImageFont.load_default()

        vis2 = image.copy()
        draw = ImageDraw.Draw(vis2)

        for lbl_id in labels_to_detect:
            color, lbl_name = LABEL_COLORS.get(lbl_id, ("red", f"label{lbl_id}"))
            detections_d = density_detect(all_results, lbl_id, image.width, image.height)
            filtered_d = [d for d in detections_d if d["density_peak"] >= args.density_min_peak]
            print(f"\nDensity {lbl_name} (peak>={args.density_min_peak}): {len(filtered_d)}/{len(detections_d)}")
            for d in filtered_d:
                x0, y0, x1, y1 = d["box"]
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                draw.text((x0, max(0, y0 - 16)),
                          f"{lbl_name} s={d['score']:.2f} d={d['density_peak']:.2f}",
                          fill=color, font=font)
                print(f"  box={d['box']}  score={d['score']:.3f}  density_peak={d['density_peak']:.3f}")

        vis2.save(out_path)
        print(f"\nDensity visualisation saved to {out_path}")

    if args.mask_vis:
        density_dir = Path("density_output")
        density_dir.mkdir(exist_ok=True)
        img_stem = Path(args.image).stem
        mask_path = density_dir / f"{img_stem}_masked.png"
        masked = mask_non_text(image, all_results, min_peak=args.density_min_peak)
        masked.save(mask_path)
        print(f"Masked image saved to {mask_path}")


if __name__ == "__main__":
    main()
