from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "quality_inputs"
OUTPUT_DIR = ROOT / "quality_outputs"

MIN_AFTER_QUALITY = {
    "neutral_scene.jpg": 94,
    "underexposed_warm.jpg": 88,
    "cool_flat.jpg": 92,
    "overexposed_cool.jpg": 88,
    "flat_hazy_landscape.jpg": 87,
    "oversaturated_green.jpg": 94,
    "portrait_warm_under.jpg": 82,
    "portrait_cool_flat.jpg": 84,
    "portrait_realistic_mixed.jpg": 82,
    "portrait_backlit_face.jpg": 82,
    "portrait_indoor_warm.jpg": 82,
    "product_yellow_cast.jpg": 92,
    "product_white_bg_gray.jpg": 90,
    "night_noisy.jpg": 76,
    "overexposed_sky_detail.jpg": 84,
    "raw_flat_gray_landscape.jpg": 86,
}

MAX_PENALTY = {
    "underexposed_warm.jpg": 3.0,
    "overexposed_cool.jpg": 2.0,
    "flat_hazy_landscape.jpg": 1.5,
    "portrait_cool_flat.jpg": 2.0,
    "portrait_realistic_mixed.jpg": 1.5,
    "portrait_indoor_warm.jpg": 2.0,
    "product_white_bg_gray.jpg": 1.5,
    "overexposed_sky_detail.jpg": 2.0,
    "raw_flat_gray_landscape.jpg": 2.0,
}


def write_image(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not ok:
        raise RuntimeError(f"Could not encode {path.name}")
    path.write_bytes(encoded.tobytes())


def read_image(path: Path) -> np.ndarray:
    image = app.decode_image_bytes(path.read_bytes())
    if image is None:
        raise RuntimeError(f"Could not read {path.name}")
    return image


def gradient_scene(width: int = 1400, height: int = 900) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    xx = np.broadcast_to(x, (height, width))
    yy = np.broadcast_to(y, (height, width))
    sky = np.dstack(
        [
            170 + 55 * (1 - yy) + 8 * xx,
            188 + 35 * (1 - yy),
            205 + 25 * (1 - yy),
        ]
    )
    ground = np.dstack(
        [
            38 + 18 * xx,
            92 + 42 * (1 - yy),
            46 + 16 * xx,
        ]
    )
    split = (yy > 0.55).astype(np.float32)[:, :, None]
    img = sky * (1 - split) + ground * split

    cv2.circle(img, (int(width * 0.65), int(height * 0.42)), 115, (58, 72, 68), -1)
    cv2.circle(img, (int(width * 0.63), int(height * 0.38)), 52, (178, 142, 120), -1)
    cv2.rectangle(img, (int(width * 0.56), int(height * 0.48)), (int(width * 0.74), int(height * 0.72)), (74, 58, 150), -1)
    cv2.rectangle(img, (int(width * 0.08), int(height * 0.5)), (int(width * 0.38), int(height * 0.74)), (62, 82, 96), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_cast(bgr: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    return np.clip(bgr.astype(np.float32) * np.array(gains, dtype=np.float32), 0, 255).astype(np.uint8)


def portrait_scene(width: int = 1200, height: int = 900) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    bg = np.dstack(
        [
            np.broadcast_to(92 + 38 * y, (height, width)),
            np.broadcast_to(102 + 32 * y, (height, width)),
            np.broadcast_to(112 + 28 * y, (height, width)),
        ]
    )
    img = bg.copy()
    cx = width // 2
    cv2.rectangle(img, (cx - 210, int(height * 0.55)), (cx + 210, int(height * 0.92)), (92, 72, 152), -1)
    cv2.circle(img, (cx, int(height * 0.38)), 175, (172, 130, 108), -1)
    cv2.circle(img, (cx, int(height * 0.38)), 192, (208, 178, 136), 16)
    cv2.rectangle(img, (cx - 70, int(height * 0.5)), (cx + 70, int(height * 0.62)), (150, 108, 92), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def realistic_portrait_scene(width: int = 1400, height: int = 950) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    xx = np.broadcast_to(x, (height, width))
    yy = np.broadcast_to(y, (height, width))
    bg = np.dstack(
        [
            76 + 42 * yy + 10 * xx,
            86 + 34 * yy,
            96 + 26 * (1 - xx),
        ]
    )
    img = bg.copy()
    cx = width // 2
    cv2.ellipse(img, (cx, int(height * 0.78)), (250, 255), 0, 0, 360, (62, 58, 118), -1)
    cv2.ellipse(img, (cx, int(height * 0.43)), (150, 185), 0, 0, 360, (174, 128, 102), -1)
    cv2.ellipse(img, (cx - 55, int(height * 0.43)), (42, 54), 0, 0, 360, (190, 144, 116), -1)
    cv2.ellipse(img, (cx + 60, int(height * 0.43)), (40, 52), 0, 0, 360, (156, 112, 92), -1)
    cv2.rectangle(img, (cx - 62, int(height * 0.59)), (cx + 68, int(height * 0.69)), (143, 102, 86), -1)
    cv2.circle(img, (cx - 44, int(height * 0.39)), 10, (44, 48, 58), -1)
    cv2.circle(img, (cx + 48, int(height * 0.39)), 10, (44, 48, 58), -1)
    cv2.ellipse(img, (cx, int(height * 0.49)), (45, 18), 0, 0, 180, (116, 72, 78), 4)
    return np.clip(img, 0, 255).astype(np.uint8)


def backlit_portrait_scene(width: int = 1400, height: int = 950) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    xx = np.broadcast_to(x, (height, width))
    yy = np.broadcast_to(y, (height, width))
    bg = np.dstack(
        [
            198 + 36 * (1 - yy),
            208 + 28 * (1 - yy),
            218 + 18 * xx,
        ]
    )
    img = bg.copy()
    cx = width // 2
    cv2.ellipse(img, (cx, int(height * 0.78)), (255, 245), 0, 0, 360, (52, 62, 118), -1)
    cv2.ellipse(img, (cx, int(height * 0.43)), (142, 176), 0, 0, 360, (118, 86, 74), -1)
    cv2.ellipse(img, (cx - 48, int(height * 0.43)), (36, 50), 0, 0, 360, (132, 96, 82), -1)
    cv2.ellipse(img, (cx + 52, int(height * 0.43)), (38, 50), 0, 0, 360, (102, 74, 66), -1)
    cv2.rectangle(img, (cx - 62, int(height * 0.59)), (cx + 68, int(height * 0.69)), (86, 64, 58), -1)
    cv2.circle(img, (cx - 42, int(height * 0.39)), 9, (32, 34, 40), -1)
    cv2.circle(img, (cx + 48, int(height * 0.39)), 9, (32, 34, 40), -1)
    cv2.ellipse(img, (cx, int(height * 0.49)), (42, 16), 0, 0, 180, (82, 48, 56), 4)
    return np.clip(img, 0, 255).astype(np.uint8)


def indoor_portrait_scene(width: int = 1300, height: int = 900) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    xx = np.broadcast_to(x, (height, width))
    yy = np.broadcast_to(y, (height, width))
    bg = np.dstack(
        [
            70 + 35 * yy + 18 * xx,
            82 + 24 * yy,
            96 + 18 * (1 - xx),
        ]
    )
    img = bg.copy()
    cx = int(width * 0.48)
    cv2.ellipse(img, (cx, int(height * 0.78)), (240, 230), 0, 0, 360, (62, 56, 118), -1)
    cv2.ellipse(img, (cx, int(height * 0.43)), (132, 168), 0, 0, 360, (156, 112, 90), -1)
    cv2.circle(img, (cx - 42, int(height * 0.39)), 8, (35, 34, 38), -1)
    cv2.circle(img, (cx + 46, int(height * 0.39)), 8, (35, 34, 38), -1)
    cv2.ellipse(img, (cx, int(height * 0.5)), (38, 14), 0, 0, 180, (96, 52, 58), 3)
    lamp = np.exp(-(((xx - 0.2) ** 2) / 0.025 + ((yy - 0.14) ** 2) / 0.018))[:, :, None]
    img = img + lamp * np.array([10, 28, 62], dtype=np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def product_scene(width: int = 1200, height: int = 900) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    bg = np.dstack(
        [
            np.broadcast_to(212 - 28 * y, (height, width)),
            np.broadcast_to(218 - 24 * y, (height, width)),
            np.broadcast_to(222 - 18 * y, (height, width)),
        ]
    )
    img = bg.copy()
    cv2.rectangle(img, (360, 300), (840, 650), (118, 128, 138), -1)
    cv2.rectangle(img, (420, 230), (780, 315), (176, 184, 194), -1)
    cv2.line(img, (360, 650), (840, 650), (82, 88, 96), 10)
    return np.clip(img, 0, 255).astype(np.uint8)


def white_product_scene(width: int = 1200, height: int = 900) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    bg = np.dstack(
        [
            np.broadcast_to(226 - 12 * y, (height, width)),
            np.broadcast_to(228 - 12 * y, (height, width)),
            np.broadcast_to(230 - 10 * y, (height, width)),
        ]
    )
    img = bg.copy()
    cv2.rectangle(img, (430, 270), (770, 650), (92, 118, 148), -1)
    cv2.rectangle(img, (470, 215), (730, 305), (152, 174, 198), -1)
    cv2.line(img, (430, 650), (770, 650), (58, 68, 82), 10)
    shadow = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(shadow, (600, 690), (260, 42), 0, 0, 360, 1, -1)
    img -= cv2.GaussianBlur(shadow, (0, 0), 20)[:, :, None] * 30
    return np.clip(img, 0, 255).astype(np.uint8)


def overexposed_sky_scene(width: int = 1400, height: int = 900) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    xx = np.broadcast_to(x, (height, width))
    yy = np.broadcast_to(y, (height, width))
    sky = np.dstack(
        [
            224 + 22 * (1 - yy),
            232 + 16 * (1 - yy),
            238 + 14 * (1 - yy),
        ]
    )
    cloud = 16 * np.sin(xx * 18) * np.sin(yy * 9)
    sky[:, :, 0] += cloud
    sky[:, :, 1] += cloud * 0.7
    mountains = np.dstack(
        [
            78 + 22 * xx,
            92 + 26 * (1 - yy),
            84 + 12 * xx,
        ]
    )
    img = sky.copy()
    mask = yy > 0.58 + 0.08 * np.sin(xx * 10)
    img[mask] = mountains[mask]
    return np.clip(img, 0, 255).astype(np.uint8)


def night_scene(width: int = 1400, height: int = 900) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    img = np.dstack(
        [
            np.broadcast_to(22 + 30 * y, (height, width)),
            np.broadcast_to(28 + 26 * y, (height, width)),
            np.broadcast_to(48 + 42 * (1 - y), (height, width)),
        ]
    )
    for px, py, color in [(0.22, 0.42, (40, 130, 240)), (0.55, 0.5, (70, 170, 210)), (0.78, 0.38, (120, 90, 210))]:
        center = (int(width * px), int(height * py))
        cv2.circle(img, center, 80, color, -1)
        cv2.circle(img, center, 170, tuple(int(c * 0.42) for c in color), -1)
    cv2.rectangle(img, (0, int(height * 0.68)), (width, height), (18, 22, 30), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_exposure(bgr: np.ndarray, gain: float, bias: float = 0) -> np.ndarray:
    return np.clip(bgr.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def raw_flat_gray_scene(base: np.ndarray) -> np.ndarray:
    flat = cv2.addWeighted(base, 0.54, np.full_like(base, 132), 0.46, -12)
    return apply_cast(flat, (1.08, 1.0, 0.92))


def make_inputs() -> list[Path]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = gradient_scene()
    portrait = portrait_scene()
    realistic_portrait = realistic_portrait_scene()
    backlit_portrait = backlit_portrait_scene()
    indoor_portrait = indoor_portrait_scene()
    product = product_scene()
    white_product = white_product_scene()
    night = night_scene()
    variants = {
        "neutral_scene.jpg": base,
        "underexposed_warm.jpg": apply_cast((base.astype(np.float32) * 0.58).astype(np.uint8), (0.86, 0.94, 1.22)),
        "cool_flat.jpg": apply_cast(cv2.addWeighted(base, 0.72, np.full_like(base, 128), 0.28, -18), (1.2, 1.0, 0.82)),
        "overexposed_cool.jpg": apply_cast(adjust_exposure(base, 1.28, 18), (1.12, 1.02, 0.9)),
        "flat_hazy_landscape.jpg": cv2.addWeighted(base, 0.48, np.full_like(base, 188), 0.52, 0),
        "oversaturated_green.jpg": cv2.cvtColor(
            np.dstack(
                [
                    cv2.cvtColor(base, cv2.COLOR_BGR2HSV)[:, :, 0],
                    np.clip(cv2.cvtColor(base, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) * 1.55, 0, 255).astype(np.uint8),
                    cv2.cvtColor(base, cv2.COLOR_BGR2HSV)[:, :, 2],
                ]
            ),
            cv2.COLOR_HSV2BGR,
        ),
        "portrait_warm_under.jpg": apply_cast(adjust_exposure(portrait, 0.66, -8), (0.86, 0.96, 1.18)),
        "portrait_cool_flat.jpg": apply_cast(cv2.addWeighted(portrait, 0.62, np.full_like(portrait, 142), 0.38, -6), (1.16, 1.0, 0.86)),
        "portrait_realistic_mixed.jpg": apply_cast(adjust_exposure(realistic_portrait, 0.74, -10), (1.0, 0.98, 0.96)),
        "portrait_backlit_face.jpg": apply_cast(backlit_portrait, (1.04, 1.0, 0.96)),
        "portrait_indoor_warm.jpg": apply_cast(adjust_exposure(indoor_portrait, 0.72, -12), (0.82, 0.94, 1.2)),
        "product_yellow_cast.jpg": apply_cast(product, (0.88, 0.98, 1.18)),
        "product_white_bg_gray.jpg": apply_cast(adjust_exposure(white_product, 0.82, -10), (1.08, 1.0, 0.9)),
        "night_noisy.jpg": np.clip(night.astype(np.float32) + np.random.default_rng(4).normal(0, 8, night.shape), 0, 255).astype(np.uint8),
        "overexposed_sky_detail.jpg": apply_cast(adjust_exposure(overexposed_sky_scene(), 1.08, 12), (1.06, 1.02, 0.96)),
        "raw_flat_gray_landscape.jpg": raw_flat_gray_scene(base),
    }
    paths: list[Path] = []
    for name, image in variants.items():
        path = INPUT_DIR / name
        write_image(path, image)
        paths.append(path)
    return paths


def main() -> None:
    app.ensure_dirs()
    app.initialize_reference_bank()
    rows = []
    failures = []
    for path in make_inputs():
        original = read_image(path)
        output, adjustments = app.polish_image(original)
        output_path = OUTPUT_DIR / path.name
        write_image(output_path, output)
        scene = adjustments["scene"]
        before_quality, after_quality = [int(part.strip()) for part in adjustments["quality"].split("->")]
        penalty = float(adjustments["weirdness"]["penalty"])
        row = {
            "name": path.name,
            "scene": scene,
            "quality": adjustments["quality"],
            "reference": adjustments["reference"],
            "referenceScore": adjustments["referenceScore"],
            "referenceFit": adjustments.get("referenceFit"),
            "referenceIntensity": adjustments.get("referenceIntensity"),
            "penalty": penalty,
            "colorShift": adjustments["weirdness"]["colorShift"],
            "clipRatio": adjustments["weirdness"]["clipRatio"],
            "productNeutrality": adjustments["weirdness"].get("productNeutrality"),
            "subjectReadability": adjustments["weirdness"].get("subjectReadability"),
            "variant": adjustments["variant"],
            "input": adjustments.get("input"),
            "output": str(output_path),
        }
        rows.append(row)
        min_after = MIN_AFTER_QUALITY[path.name]
        max_penalty = MAX_PENALTY.get(path.name, 0.75)
        if after_quality < min_after:
            failures.append(f"{path.name}: quality {before_quality}->{after_quality} below {min_after}")
        if penalty > max_penalty:
            failures.append(f"{path.name}: penalty {penalty} above {max_penalty}")
        if path.name == "product_yellow_cast.jpg" and float(row["productNeutrality"] or 0) < 82:
            failures.append(f"{path.name}: product neutrality {row['productNeutrality']} below 82")
        if path.name == "product_white_bg_gray.jpg" and float(row["productNeutrality"] or 0) < 86:
            failures.append(f"{path.name}: product neutrality {row['productNeutrality']} below 86")
        if path.name.startswith("portrait_") and float(row["colorShift"]) > 5.2:
            failures.append(f"{path.name}: portrait color shift {row['colorShift']} above 5.2")
        if path.name == "portrait_backlit_face.jpg" and float(row["subjectReadability"] or 0) < 68:
            failures.append(f"{path.name}: subject readability {row['subjectReadability']} below 68")
        if row["input"] == "GRAPHIC":
            if float(row["referenceIntensity"] or 0) > 0.05:
                failures.append(f"{path.name}: graphic input used reference intensity {row['referenceIntensity']}")
            if penalty > 0.25:
                failures.append(f"{path.name}: graphic input penalty {penalty} above 0.25")
    for row in rows:
        print(row)
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
