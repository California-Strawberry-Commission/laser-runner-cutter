""" File: diagnose_confidence.py

Description: Diagnose why a YOLOv8 model (.pt) produces low confidence for an object in an image.

Runs a battery of tests that each change one factor (preprocessing, photometric properties,
geometry, context) and reports how the confidence of the target object responds. Factors
that substantially raise confidence point at the likely cause.

Example:
    python ml_utils/diagnose_confidence.py \
        --weights_file runner_segmentation_model/models/runner3455-yolov8l-seg/weights/best.pt \
        --image_path path/to/image.png \
        --reference_dir runner_segmentation_model/data/prepared/runner3455/images
"""

import argparse
import os
from glob import glob

import cv2
import numpy as np
from natsort import natsorted
from ultralytics import YOLO

# Minimum IoU between a detection and the target box for the detection to count as the target
MATCH_IOU = 0.3
# Confidence change considered meaningful when summarizing results
SIGNIFICANT_DELTA = 0.05
# Confidence gain large enough to attribute a likely cause in the summary
MAJOR_DELTA = 0.1


def tuple_type(arg_string):
    try:
        return tuple(map(int, arg_string.strip("()").split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {arg_string}")


def iou(box, boxes):
    """IoU between one xyxy box and an (N, 4) array of xyxy boxes."""
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def image_stats(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = [image_bgr[..., i].astype(np.float64) for i in range(3)]
    return {
        "mean_b": float(b.mean()),
        "mean_g": float(g.mean()),
        "mean_r": float(r.mean()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(hsv[..., 1].mean()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "pct_saturated_pixels": float(100.0 * np.mean(gray >= 250)),
        "pct_black_pixels": float(100.0 * np.mean(gray <= 5)),
        "channel_spread": float(
            np.mean(np.abs(b - g)) + np.mean(np.abs(g - r)) + np.mean(np.abs(b - r))
        ),
    }


class Diagnoser:
    def __init__(self, weights_file, iou_thresh):
        self.model = YOLO(weights_file)
        self.train_args = {}
        if isinstance(getattr(self.model, "ckpt", None), dict):
            self.train_args = self.model.ckpt.get("train_args", {}) or {}
        self.imgsz = self.train_args.get("imgsz", 640)
        self.iou_thresh = iou_thresh
        self.results = []

    def predict(self, image_bgr):
        """Run prediction with a near-zero threshold so low-confidence candidates are visible."""
        result = self.model.predict(
            image_bgr,
            imgsz=self.imgsz,
            conf=0.001,
            iou=self.iou_thresh,
            max_det=300,
            device="cpu",
            verbose=False,
        )[0]
        boxes = result.boxes
        return {
            "xyxy": boxes.xyxy.cpu().numpy(),
            "conf": boxes.conf.cpu().numpy(),
            "cls": boxes.cls.cpu().numpy().astype(int),
            "result": result,
        }

    @staticmethod
    def match(dets, target_box):
        """Return (conf, iou, cls) of the detection best matching target_box."""
        ious = iou(target_box, dets["xyxy"])
        if len(ious) == 0:
            return 0.0, 0.0, None
        # Among detections overlapping the target, pick the most confident one
        candidates = np.where(ious >= MATCH_IOU)[0]
        if len(candidates) == 0:
            best = int(np.argmax(ious))
            return 0.0, float(ious[best]), None
        best = candidates[np.argmax(dets["conf"][candidates])]
        return float(dets["conf"][best]), float(ious[best]), int(dets["cls"][best])

    def run(self, category, name, image_bgr, target_box):
        dets = self.predict(image_bgr)
        conf, match_iou, cls = self.match(dets, target_box)
        entry = {
            "category": category,
            "name": name,
            "conf": conf,
            "iou": match_iou,
            "cls": cls,
            "max_conf_in_image": float(dets["conf"].max()) if len(dets["conf"]) else 0.0,
        }
        self.results.append(entry)
        return entry, dets


def print_entry(entry, baseline_conf):
    delta = entry["conf"] - baseline_conf
    found = "" if entry["cls"] is not None else f" (no detection, best IoU={entry['iou']:.2f})"
    print(
        f"  {entry['name']:<38} conf={entry['conf']:.3f}  delta={delta:+.3f}{found}"
    )


##################
# Image transforms
##################


def adjust_gamma(image, gamma):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, lut)


def adjust_contrast(image, alpha):
    mean = image.mean()
    return np.clip((image.astype(np.float32) - mean) * alpha + mean, 0, 255).astype(
        np.uint8
    )


def adjust_saturation(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gray_world_white_balance(image):
    img = image.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    img *= means.mean() / (means + 1e-6)
    return np.clip(img, 0, 255).astype(np.uint8)


def sharpen(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def keep_region(image, box, context):
    """Return the integer xyxy region of box expanded by `context` times its size on each side."""
    h, w = image.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1 = int(max(0, box[0] - context * bw))
    y1 = int(max(0, box[1] - context * bh))
    x2 = int(min(w, np.ceil(box[2] + context * bw)))
    y2 = int(min(h, np.ceil(box[3] + context * bh)))
    return x1, y1, x2, y2


#############
# Test suites
#############


def test_preprocessing(d, image, target, baseline_conf):
    # Downscale-then-upscale to simulate a lower-resolution capture
    h, w = image.shape[:2]
    for factor in [0.5, 0.25]:
        small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        entry, _ = d.run("preprocessing", f"resolution loss x{factor}", degraded, target)
        print_entry(entry, baseline_conf)

    # JPEG compression artifacts
    for quality in [90, 50]:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            jpg = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            entry, _ = d.run("preprocessing", f"JPEG quality={quality}", jpg, target)
            print_entry(entry, baseline_conf)


def test_photometric(d, image, target, baseline_conf):
    tests = [
        ("gamma=0.5 (brighter)", lambda im: adjust_gamma(im, 0.5)),
        ("gamma=0.75 (brighter)", lambda im: adjust_gamma(im, 0.75)),
        ("gamma=1.5 (darker)", lambda im: adjust_gamma(im, 1.5)),
        ("gamma=2.0 (darker)", lambda im: adjust_gamma(im, 2.0)),
        ("contrast x0.7", lambda im: adjust_contrast(im, 0.7)),
        ("contrast x1.4", lambda im: adjust_contrast(im, 1.4)),
        ("saturation x0.5", lambda im: adjust_saturation(im, 0.5)),
        ("saturation x1.5", lambda im: adjust_saturation(im, 1.5)),
        ("grayscale", lambda im: cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)),
        ("CLAHE", clahe),
        ("gray-world white balance", gray_world_white_balance),
        ("gaussian blur sigma=2", lambda im: cv2.GaussianBlur(im, (0, 0), 2.0)),
        ("unsharp mask", sharpen),
        (
            "gaussian noise sigma=10",
            lambda im: np.clip(
                im.astype(np.float32) + np.random.default_rng(0).normal(0, 10, im.shape), 0, 255
            ).astype(np.uint8),
        ),
    ]
    for name, fn in tests:
        entry, _ = d.run("photometric", name, fn(image), target)
        print_entry(entry, baseline_conf)


def test_geometric(d, image, target, baseline_conf):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = target
    tests = [
        ("flip left-right", cv2.flip(image, 1), np.array([w - x2, y1, w - x1, y2])),
        ("flip up-down", cv2.flip(image, 0), np.array([x1, h - y2, x2, h - y1])),
        (
            "rotate 90 cw",
            cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
            np.array([h - y2, x1, h - y1, x2]),
        ),
        (
            "rotate 90 ccw",
            cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
            np.array([y1, w - x2, y2, w - x1]),
        ),
    ]
    for name, img, box in tests:
        entry, _ = d.run("geometric", name, img, box)
        print_entry(entry, baseline_conf)

    # Object scale: zoom in/out on the whole image. When run at fixed imgsz, this changes
    # the object's apparent size in pixels the model sees.
    for scale in [0.5, 0.75, 1.5, 2.0]:
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if scale > 1:
            # Crop back to the original size centered on the target to keep input size fixed
            cx, cy = (target[0] + target[2]) / 2 * scale, (target[1] + target[3]) / 2 * scale
            ox = int(np.clip(cx - w / 2, 0, resized.shape[1] - w))
            oy = int(np.clip(cy - h / 2, 0, resized.shape[0] - h))
            resized = resized[oy : oy + h, ox : ox + w]
            box = target * scale - np.array([ox, oy, ox, oy])
        else:
            # Pad back to the original size with gray (like letterbox padding)
            padded = np.full_like(image, 114)
            padded[: resized.shape[0], : resized.shape[1]] = resized
            resized = padded
            box = target * scale
        entry, _ = d.run("geometric", f"object scale x{scale}", resized, box)
        print_entry(entry, baseline_conf)


def test_context(d, image, target, baseline_conf):
    # Keep only the target plus some surrounding context, gray elsewhere. The image size and
    # object position are unchanged so object scale is not a confounding factor.
    for context in [0.0, 0.25, 1.0, 3.0]:
        masked = np.full_like(image, 114)
        x1, y1, x2, y2 = keep_region(image, target, context)
        masked[y1:y2, x1:x2] = image[y1:y2, x1:x2]
        entry, _ = d.run("context", f"keep only {context}x box context", masked, target)
        print_entry(entry, baseline_conf)


def compare_to_reference(d, image, reference_dir, max_images):
    """Compare image statistics and model confidence against mean of reference images."""
    print("\n" + "=" * 78)
    print("STATS COMPARISON VS REFERENCE IMAGES SAMPLE")
    print("=" * 78)
    paths = natsorted(
        glob(os.path.join(reference_dir, "**", "*.jpg"), recursive=True)
        + glob(os.path.join(reference_dir, "**", "*.png"), recursive=True)
    )
    if not paths:
        print("  No reference images found.")
        return None
    if len(paths) > max_images:
        idx = np.linspace(0, len(paths) - 1, max_images).astype(int)
        paths = [paths[i] for i in idx]

    ref_stats = []
    ref_top_confs = []
    ref_sizes = []
    for path in paths:
        ref = cv2.imread(path, cv2.IMREAD_COLOR)
        if ref is None:
            continue
        ref_sizes.append(ref.shape[:2])
        ref_stats.append(image_stats(ref))
        dets = d.predict(ref)
        ref_top_confs.append(float(dets["conf"].max()) if len(dets["conf"]) else 0.0)
    test_stats = image_stats(image)

    print(f"  Compared against {len(ref_stats)} reference images")
    print(f"  {'statistic':<22} {'image':>10} {'ref mean':>10} {'ref std':>10} {'z':>7}")
    flagged = []
    for key in test_stats:
        values = np.array([s[key] for s in ref_stats])
        mean, std = values.mean(), values.std() + 1e-9
        z = (test_stats[key] - mean) / std
        if abs(z) > 2:
            flagged.append((key, z))
        print(f"  {key:<22} {test_stats[key]:>10.2f} {mean:>10.2f} {std:>10.2f} {z:>+7.2f}")

    sizes = {f"{w}x{h}" for h, w in ref_sizes}
    h, w = image.shape[:2]
    if f"{w}x{h}" not in sizes:
        print(f"  Image size {w}x{h} differs from reference sizes {sorted(sizes)[:5]}")
    ref_top_confs = np.array(ref_top_confs)
    print(
        f"  Top detection conf on reference images: median={np.median(ref_top_confs):.3f}, "
        f"p10={np.percentile(ref_top_confs, 10):.3f}, p90={np.percentile(ref_top_confs, 90):.3f}"
    )
    return {
        "flagged_stats": flagged,
        "ref_median_top_conf": float(np.median(ref_top_confs)),
        "ref_p10_top_conf": float(np.percentile(ref_top_confs, 10)),
    }


def summarize(d, baseline_conf, ref_summary):
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Baseline confidence of target: {baseline_conf:.3f}")

    # Print largest increases in confidence
    gains = sorted(
        [r for r in d.results if r["conf"] - baseline_conf >= SIGNIFICANT_DELTA],
        key=lambda r: r["conf"],
        reverse=True,
    )
    if gains:
        print("\nChanges that raised confidence the most:")
        for r in gains[:10]:
            print(f"  [{r['category']}] {r['name']}: {r['conf']:.3f} ({r['conf'] - baseline_conf:+.3f})")

    # Print notable findings
    findings = list()
    if ref_summary is not None:
        if baseline_conf < ref_summary["ref_p10_top_conf"]:
            findings.append(
                f"Target confidence {baseline_conf:.3f} is below the 10th percentile of top "
                f"confidences on reference images ({ref_summary['ref_p10_top_conf']:.3f}). This "
                "image is harder than typical for the model."
            )
        for key, z in ref_summary["flagged_stats"]:
            findings.append(
                f"Image '{key}' is {abs(z):.1f} std {'above' if z > 0 else 'below'} the "
                "reference distribution."
            )

    print("\nNotes:")
    for i, f in enumerate(findings, 1):
        print(f"  {i}. {f}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose low confidence of a YOLOv8 detection on an image"
    )
    parser.add_argument("--weights_file", required=True, help="Path to .pt model file")
    parser.add_argument("--image_path", required=True, help="Path to image file")
    parser.add_argument(
        "--image_size",
        type=tuple_type,
        default="(1024, 768)",
        help="(width, height) tuple the input image is resized to before inference",
    )
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument(
        "--reference_dir",
        help="Dir of reference images (e.g. training/val set) to compare distribution against",
    )
    parser.add_argument("--max_reference_images", type=int, default=100)
    args = parser.parse_args()

    d = Diagnoser(args.weights_file, args.iou)

    raw = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(raw, args.image_size, interpolation=cv2.INTER_LINEAR)

    # Print model info
    print("=" * 78)
    print("MODEL")
    print("=" * 78)
    print(f"  Weights: {args.weights_file}")
    print(f"  Task: {d.model.task}, classes: {d.model.names}")
    keys = ["imgsz", "rect", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
            "fliplr", "flipud", "mosaic", "epochs", "data"]
    for k in keys:
        if k in d.train_args:
            print(f"  train {k}: {d.train_args[k]}")
    print(f"  Inference: imgsz={d.imgsz}, iou={d.iou_thresh}, device=cpu")

    # Print image info
    print("\n" + "=" * 78)
    print("IMAGE")
    print("=" * 78)
    print(f"  Path: {args.image_path}")
    print(f"  Raw shape: {raw.shape}, dtype: {raw.dtype}")
    print(f"  Resized to: {args.image_size[0]}x{args.image_size[1]}")
    for k, v in image_stats(image).items():
        print(f"  {k}: {v:.2f}")

    # Run baseline inference
    dets = d.predict(image)
    print("\n" + "=" * 78)
    print("BASELINE")
    print("=" * 78)
    order = np.argsort(-dets["conf"])
    print(f"  {len(order)} candidates at conf>=0.001. Top 10:")
    for i in order[:10]:
        print(f"    conf={dets['conf'][i]:.3f}  cls={d.model.names[dets['cls'][i]]}  box={np.round(dets['xyxy'][i]).astype(int).tolist()}")

    if len(order) == 0:
        print("No detections at conf=0.001. Nothing to diagnose.")
        return
    
    target = dets["xyxy"][order[0]]
    baseline_conf = float(dets["conf"][order[0]])
    print(
        f"  Using highest conf detection as target: {np.round(target).astype(int).tolist()} "
        f"(conf={baseline_conf:.3f})"
    )

    # Run tests
    print("\n" + "=" * 78)
    print("TEST: PREPROCESSING")
    print("=" * 78)
    test_preprocessing(d, image, target, baseline_conf)

    print("\n" + "=" * 78)
    print("TEST: PHOTOMETRIC")
    print("=" * 78)
    test_photometric(d, image, target, baseline_conf)

    print("\n" + "=" * 78)
    print("TEST: GEOMETRIC")
    print("=" * 78)
    test_geometric(d, image, target, baseline_conf)

    print("\n" + "=" * 78)
    print("TEST: CONTEXT")
    print("=" * 78)
    test_context(d, image, target, baseline_conf)

    ref_summary = None
    if args.reference_dir:
        ref_summary = compare_to_reference(d, image, args.reference_dir, args.max_reference_images)

    summarize(d, baseline_conf, ref_summary)


if __name__ == "__main__":
    main()
