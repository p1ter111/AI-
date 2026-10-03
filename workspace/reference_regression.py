from __future__ import annotations

from pathlib import Path
import sys

import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "references"
OUTPUT_DIR = ROOT / "reference_outputs"
CONTACT_SHEET = ROOT / "reference_outputs_contact.jpg"

EXPECTED_SCENES = {
    "室内生活样片.jpg": "general",
    "稳态夜景样片.jpg": "night",
    "自然人像样片.jpg": "portrait",
    "自然风景样片.jpg": "landscape",
    "通透风景样片.jpg": "landscape",
}


def save_image(path: Path, bgr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not ok:
        raise RuntimeError(f"Could not encode {path.name}")
    path.write_bytes(encoded.tobytes())


def thumb(bgr, size: tuple[int, int] = (360, 240)) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def write_contact_sheet(tiles: list[Image.Image]) -> None:
    if not tiles:
        return
    width, height = tiles[0].size
    sheet = Image.new("RGB", (width, height * len(tiles)), (230, 230, 230))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (0, index * height))
    sheet.save(CONTACT_SHEET, quality=92)


def main() -> None:
    app.ensure_dirs()
    app.initialize_reference_bank()
    rows = []
    failures = []
    tiles = []
    for path in sorted(REFERENCE_DIR.glob("*.jpg")):
        original = app.decode_image_bytes(path.read_bytes())
        if original is None:
            continue
        output, adjustments = app.polish_image(original)
        output_path = OUTPUT_DIR / path.name
        save_image(output_path, output)
        weirdness = adjustments["weirdness"]
        before_quality, after_quality = [int(part.strip()) for part in adjustments["quality"].split("->")]
        penalty = float(weirdness["penalty"])
        row = {
            "name": path.name,
            "scene": adjustments["scene"],
            "quality": adjustments["quality"],
            "referenceScore": adjustments["referenceScore"],
            "referenceFit": adjustments.get("referenceFit"),
            "referenceIntensity": adjustments.get("referenceIntensity"),
            "protected": adjustments.get("highQualityProtected", False),
            "penalty": penalty,
            "colorShift": weirdness["colorShift"],
            "saturationShift": weirdness["saturationShift"],
        }
        rows.append(row)
        expected_scene = EXPECTED_SCENES.get(path.name)
        if expected_scene and adjustments["scene"] != expected_scene:
            failures.append(f"{path.name}: scene {adjustments['scene']} expected {expected_scene}")
        if before_quality >= 90 and after_quality < before_quality:
            failures.append(f"{path.name}: high-quality sample regressed {before_quality}->{after_quality}")
        if after_quality < min(99, before_quality + 2) and before_quality < 90:
            failures.append(f"{path.name}: weak quality improvement {before_quality}->{after_quality}")
        max_penalty = 5.0 if adjustments["scene"] in {"portrait", "night"} else 3.0
        if penalty > max_penalty:
            failures.append(f"{path.name}: penalty {penalty} above {max_penalty}")
        if before_quality >= 93:
            if float(weirdness["colorShift"]) > 6.2:
                failures.append(f"{path.name}: high-quality color shift {weirdness['colorShift']} above 6.2")
            if float(weirdness["saturationShift"]) > 14.5:
                failures.append(f"{path.name}: high-quality saturation shift {weirdness['saturationShift']} above 14.5")

        tile = Image.new("RGB", (720, 290), "white")
        tile.paste(thumb(original), (0, 0))
        tile.paste(thumb(output), (360, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 246), path.name, fill=(0, 0, 0))
        draw.text(
            (368, 246),
            f"{adjustments['scene']} {adjustments['quality']} R{adjustments.get('referenceIntensity')} C{weirdness['colorShift']} S{weirdness['saturationShift']}",
            fill=(0, 0, 0),
        )
        tiles.append(tile)
    for row in rows:
        print(row)
    write_contact_sheet(tiles)
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
