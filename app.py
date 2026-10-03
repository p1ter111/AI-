from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any
from urllib.error import URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import rawpy  # type: ignore
except Exception:
    rawpy = None


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
WORKSPACE_DIR = ROOT / "workspace"
UPLOAD_DIR = WORKSPACE_DIR / "uploads"
OUTPUT_DIR = WORKSPACE_DIR / "outputs"
PREVIEW_DIR = WORKSPACE_DIR / "previews"
REFERENCE_DIR = WORKSPACE_DIR / "references"
REFERENCE_OUTPUT_DIR = WORKSPACE_DIR / "reference_outputs"
LOG_DIR = WORKSPACE_DIR / "logs"
EVENT_LOG = LOG_DIR / "events.jsonl"
JOBS_DB = WORKSPACE_DIR / "jobs.json"

CALIBRATION_SAMPLE_MAP = [
    ("室内生活样片.jpg", "室内生活", "general"),
    ("自然风景样片.jpg", "自然风景", "landscape"),
    ("通透风景样片.jpg", "高质量风景", "landscape"),
]

COMMON_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
RAW_EXTENSIONS = {
    ".3fr",
    ".ari",
    ".arw",
    ".bay",
    ".braw",
    ".cr2",
    ".cr3",
    ".crw",
    ".dcr",
    ".dng",
    ".erf",
    ".fff",
    ".iiq",
    ".k25",
    ".kdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".raf",
    ".raw",
    ".rwl",
    ".rw2",
    ".sr2",
    ".srf",
    ".srw",
    ".x3f",
}

REFERENCE_URLS = [
    ("自然风景样片", "landscape", "https://picsum.photos/id/1011/1200/800"),
    ("通透风景样片", "landscape", "https://picsum.photos/id/1015/1200/800"),
    ("自然人像样片", "portrait", "https://picsum.photos/id/1027/900/1200"),
    ("室内生活样片", "general", "https://picsum.photos/id/1060/1200/800"),
    ("干净产品样片", "product", "https://picsum.photos/id/1062/1200/800"),
    ("稳态夜景样片", "night", "https://picsum.photos/id/1019/1200/800"),
]

REFERENCE_KIND_BY_NAME = {name: kind for name, kind, _ in REFERENCE_URLS}


def ensure_dirs() -> None:
    for path in (UPLOAD_DIR, OUTPUT_DIR, PREVIEW_DIR, REFERENCE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in COMMON_EXTENSIONS | RAW_EXTENSIONS


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    stem = Path(name).stem or "image"
    suffix = Path(name).suffix.lower()
    clean = "".join(ch if ch.isalnum() or ch in "._-()[] " else "_" for ch in stem).strip()
    return f"{clean[:80] or 'image'}{suffix}"


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def now_ms() -> int:
    return int(time.time() * 1000)


def log_event(kind: str, message: str, **data: Any) -> None:
    ensure_dirs()
    event = {"time": now_ms(), "kind": kind, "message": message, **data}
    try:
        with EVENT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def recent_events(limit: int = 30) -> list[dict[str, Any]]:
    if not EVENT_LOG.exists():
        return []
    try:
        lines = EVENT_LOG.read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in lines if line.strip()]
        diagnostics = [
            event
            for event in events
            if event.get("kind") in {"error", "reference", "raw"}
            or event.get("message") in {"处理失败", "RAW 解码彻底失败", "本地参考样片读取失败"}
        ]
        return diagnostics[-limit:]
    except Exception:
        return []


@dataclass
class ReferenceProfile:
    name: str
    kind: str
    source: str
    strength: float
    lab_mean: list[float]
    lab_std: list[float]
    hsv_mean: list[float]
    contrast: float
    preview_path: Path | None = None
    online: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "strength": self.strength,
            "online": self.online,
            "previewUrl": f"/api/reference-preview/{self.name}" if self.preview_path and self.preview_path.exists() else None,
        }


@dataclass
class Job:
    id: str
    original_name: str
    upload_path: Path
    original_preview_path: Path
    output_path: Path
    preview_path: Path
    status: str = "queued"
    progress: int = 0
    message: str = "等待处理"
    created_at: int = field(default_factory=now_ms)
    updated_at: int = field(default_factory=now_ms)
    input_size: int = 0
    output_size: int = 0
    width: int | None = None
    height: int | None = None
    adjustments: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "originalName": self.original_name,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "inputSize": self.input_size,
            "outputSize": self.output_size,
            "width": self.width,
            "height": self.height,
            "adjustments": self.adjustments,
            "error": self.error,
            "originalPreviewUrl": f"/api/original-preview/{self.id}" if self.original_preview_path.exists() else None,
            "previewUrl": f"/api/preview/{self.id}" if self.preview_path.exists() else None,
            "downloadUrl": f"/api/download/{self.id}" if self.output_path.exists() else None,
        }

    def persisted(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "originalName": self.original_name,
            "uploadPath": str(self.upload_path),
            "originalPreviewPath": str(self.original_preview_path),
            "outputPath": str(self.output_path),
            "previewPath": str(self.preview_path),
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "inputSize": self.input_size,
            "outputSize": self.output_size,
            "width": self.width,
            "height": self.height,
            "adjustments": self.adjustments,
            "error": self.error,
        }

    @classmethod
    def from_persisted(cls, payload: dict[str, Any]) -> "Job":
        return cls(
            id=payload["id"],
            original_name=payload["originalName"],
            upload_path=Path(payload["uploadPath"]),
            original_preview_path=Path(payload["originalPreviewPath"]),
            output_path=Path(payload["outputPath"]),
            preview_path=Path(payload["previewPath"]),
            status=payload.get("status", "queued"),
            progress=int(payload.get("progress", 0)),
            message=payload.get("message", "等待处理"),
            created_at=int(payload.get("createdAt", now_ms())),
            updated_at=int(payload.get("updatedAt", now_ms())),
            input_size=int(payload.get("inputSize", 0)),
            output_size=int(payload.get("outputSize", 0)),
            width=payload.get("width"),
            height=payload.get("height"),
            adjustments=payload.get("adjustments", {}),
            error=payload.get("error"),
        )


jobs: dict[str, Job] = {}
jobs_lock = Lock()
job_queue: Queue[str] = Queue()
reference_lock = Lock()
reference_bank: list[ReferenceProfile] = []
reference_status = {
    "mode": "自动参考",
    "message": "内置审美参考库已启用，联网样片会自动补充。",
    "updatedAt": now_ms(),
}


def builtin_reference_bank() -> list[ReferenceProfile]:
    return [
        ReferenceProfile("自然人像", "portrait", "内置审美目标", 0.22, [158, 133, 141], [45, 9, 12], [18, 54, 166], 118),
        ReferenceProfile("清爽风景", "landscape", "内置审美目标", 0.28, [148, 126, 135], [58, 14, 17], [72, 68, 158], 142),
        ReferenceProfile("干净静物", "product", "内置审美目标", 0.20, [166, 128, 130], [42, 7, 8], [36, 42, 172], 104),
        ReferenceProfile("通用自然", "general", "内置审美目标", 0.24, [154, 128, 134], [50, 10, 13], [42, 56, 162], 122),
        ReferenceProfile("夜景稳态", "night", "内置审美目标", 0.18, [104, 130, 134], [54, 10, 12], [78, 50, 112], 118),
    ]


def set_job(job: Job, **changes: Any) -> None:
    with jobs_lock:
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = now_ms()
        save_jobs_locked()


def save_jobs_locked() -> None:
    ensure_dirs()
    payload = [job.persisted() for job in jobs.values()]
    temp_path = JOBS_DB.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(JOBS_DB)


def save_jobs() -> None:
    with jobs_lock:
        save_jobs_locked()


def load_jobs() -> None:
    if not JOBS_DB.exists():
        return
    try:
        payload = json.loads(JOBS_DB.read_text(encoding="utf-8"))
        restored = 0
        for item in payload:
            job = Job.from_persisted(item)
            if not job.upload_path.exists():
                continue
            if job.status in {"queued", "processing"}:
                job.status = "queued"
                job.progress = 0
                job.message = "服务重启后等待继续处理"
            jobs[job.id] = job
            restored += 1
        log_event("system", "已恢复历史任务", count=restored)
    except Exception as exc:
        log_event("error", "恢复历史任务失败", error=str(exc))


def recover_uploads_without_jobs() -> None:
    ensure_dirs()
    with jobs_lock:
        known_uploads = {job.upload_path.resolve() for job in jobs.values()}
        recovered = 0
        for upload_path in UPLOAD_DIR.iterdir():
            if not upload_path.is_file() or upload_path.resolve() in known_uploads:
                continue
            if not is_supported(upload_path.name):
                continue
            job_id = uuid.uuid4().hex[:12]
            raw_name = upload_path.name
            if "_" in raw_name and len(raw_name.split("_", 1)[0]) >= 8:
                original_name = raw_name.split("_", 1)[1]
            else:
                original_name = raw_name
            output_path = OUTPUT_DIR / f"{Path(original_name).stem}_{job_id}_retouched.jpg"
            preview_path = PREVIEW_DIR / f"{job_id}.jpg"
            original_preview_path = PREVIEW_DIR / f"{job_id}_original.jpg"
            job = Job(
                id=job_id,
                original_name=original_name,
                upload_path=upload_path,
                original_preview_path=original_preview_path,
                output_path=output_path,
                preview_path=preview_path,
                input_size=upload_path.stat().st_size,
                message="从磁盘恢复，等待处理",
            )
            jobs[job_id] = job
            recovered += 1
        if recovered:
            save_jobs_locked()
            log_event("system", "已从上传目录恢复任务", count=recovered)


def enqueue_unfinished_jobs() -> None:
    with jobs_lock:
        pending = [job.id for job in jobs.values() if job.status in {"queued", "processing"}]
    for job_id in pending:
        job_queue.put(job_id)


def reference_snapshot() -> tuple[list[ReferenceProfile], dict[str, Any]]:
    with reference_lock:
        return list(reference_bank), dict(reference_status)


def calibration_snapshot() -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    updated_at = 0
    for filename, label, scene in CALIBRATION_SAMPLE_MAP:
        source_path = REFERENCE_DIR / filename
        output_path = REFERENCE_OUTPUT_DIR / filename
        if not source_path.exists() or not output_path.exists():
            continue
        updated_at = max(updated_at, int(output_path.stat().st_mtime_ns // 1_000_000))
        encoded = quote(filename)
        samples.append(
            {
                "name": label,
                "scene": scene,
                "sourceUrl": f"/api/calibration-source/{encoded}",
                "outputUrl": f"/api/calibration-output/{encoded}",
                "updatedAt": updated_at,
            }
        )
    return {"count": len(samples), "updatedAt": updated_at or now_ms(), "samples": samples}


def load_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    suffix = path.suffix.lower()
    metadata: dict[str, Any] = {"source": "standard"}

    if suffix in RAW_EXTENSIONS:
        metadata["source"] = "raw"
        if rawpy is None:
            raise RuntimeError("当前环境还没有安装 rawpy，无法解码 RAW。请运行 install_dependencies.bat 后重试。")
        try:
            with rawpy.imread(str(path)) as raw:
                try:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        no_auto_bright=False,
                        output_bps=8,
                        gamma=(2.222, 4.5),
                        user_flip=-1,
                        half_size=False,
                    )
                    metadata["decode"] = "full"
                    log_event("raw", "RAW 完整质量解码成功", file=path.name)
                except Exception as full_exc:
                    log_event("raw", "RAW 完整解码失败，尝试半尺寸解码", file=path.name, error=str(full_exc))
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        no_auto_bright=False,
                        output_bps=8,
                        gamma=(2.222, 4.5),
                        user_flip=-1,
                        half_size=True,
                    )
                    metadata["decode"] = "half"
                    metadata["halfSize"] = True
                    metadata["fallback"] = "完整 RAW 解码失败，已使用半尺寸高稳定模式继续处理。"
                    log_event("raw", "RAW 半尺寸解码成功", file=path.name)
        except Exception:
            try:
                with rawpy.imread(str(path)) as raw:
                    thumb = raw.extract_thumb()
                    rgb = decode_raw_thumbnail(thumb)
                    metadata["source"] = "raw preview"
                    metadata["fallback"] = "完整 RAW 解码失败，已使用相机内嵌预览图继续处理。"
                    log_event("raw", "RAW 完整解码失败，已使用内嵌预览兜底", file=path.name)
            except Exception as exc:
                try:
                    rgb = extract_embedded_jpeg_from_raw(path)
                    metadata["source"] = "raw embedded jpeg"
                    metadata["fallback"] = "RAW 解码和缩略图接口失败，已从文件中扫描内嵌 JPEG 继续处理。"
                    log_event("raw", "已从 RAW 文件字节扫描内嵌 JPEG 兜底", file=path.name)
                except Exception:
                    log_event("error", "RAW 解码彻底失败", file=path.name, error=str(exc))
                    raise RuntimeError("RAW 解码失败：文件可能不是有效 RAW、已损坏，或当前 LibRaw 版本暂不支持这个相机格式。") from exc
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), metadata

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            rgb = np.array(image)
            metadata["source"] = image.format or "standard"
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), metadata
    except UnidentifiedImageError as exc:
        raise RuntimeError("无法识别这个图片格式。") from exc


def decode_raw_thumbnail(thumb: Any) -> np.ndarray:
    if str(getattr(thumb, "format", "")).lower().endswith("jpeg"):
        from io import BytesIO

        try:
            with Image.open(BytesIO(thumb.data)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                return np.array(image)
        except Exception:
            array = np.frombuffer(thumb.data, dtype=np.uint8)
            bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("RAW 内嵌预览图无法读取。")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    data = np.asarray(thumb.data)
    if data.ndim == 3:
        return data.astype(np.uint8)
    raise RuntimeError("RAW 内嵌预览图格式暂不支持。")


def extract_embedded_jpeg_from_raw(path: Path) -> np.ndarray:
    data = path.read_bytes()
    starts: list[int] = []
    pos = 0
    while True:
        pos = data.find(b"\xff\xd8\xff", pos)
        if pos == -1:
            break
        starts.append(pos)
        pos += 3

    best: bytes | None = None
    for start in starts:
        end = data.find(b"\xff\xd9", start + 4)
        if end == -1:
            continue
        blob = data[start : end + 2]
        if best is None or len(blob) > len(best):
            best = blob

    if not best or len(best) < 64 * 1024:
        raise RuntimeError("没有找到可用的 RAW 内嵌 JPEG。")

    from io import BytesIO

    with Image.open(BytesIO(best)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.array(image)


def decode_image_bytes(content: bytes) -> np.ndarray:
    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    from io import BytesIO

    try:
        with Image.open(BytesIO(content)) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise RuntimeError("参考样片不是可识别的图片。") from exc


def resize_for_analysis(bgr: np.ndarray, max_side: int = 1400) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return bgr
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def resize_for_processing(bgr: np.ndarray, max_megapixels: float = 16.0) -> tuple[np.ndarray, str | None]:
    h, w = bgr.shape[:2]
    megapixels = (h * w) / 1_000_000
    if megapixels <= max_megapixels:
        return bgr, None
    scale = (max_megapixels / megapixels) ** 0.5
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)
    return resized, f"超大图片已按比例优化到约 {max_megapixels:g}MP 输出，避免批量处理卡死。"


def percentile_clip(channel: np.ndarray, low: float, high: float) -> np.ndarray:
    lo, hi = np.percentile(channel, (low, high))
    if hi <= lo:
        return channel
    clipped = np.clip((channel.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
    return clipped.astype(np.uint8)


def gray_world_white_balance(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    sample = resize_for_analysis(bgr, 900).astype(np.float32)
    mask = cv2.cvtColor(sample.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    valid = (mask > np.percentile(mask, 12)) & (mask < np.percentile(mask, 96))
    if not np.any(valid):
        valid = np.ones(mask.shape, dtype=bool)
    avg_b, avg_g, avg_r = [float(np.mean(sample[:, :, index][valid])) for index in range(3)]
    gray = (avg_b + avg_g + avg_r) / 3.0
    gains = np.array([gray / max(avg_b, 1.0), gray / max(avg_g, 1.0), gray / max(avg_r, 1.0)])
    gains = np.clip(gains, 0.8, 1.24)
    balanced = np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    color_cast = float(max(abs(gains[0] - 1), abs(gains[2] - 1)))
    return balanced, color_cast


def neutral_white_balance(bgr: np.ndarray, scene: str = "general") -> tuple[np.ndarray, float]:
    sample = resize_for_analysis(bgr, 900).astype(np.float32)
    sample_u8 = sample.astype(np.uint8)
    gray = cv2.cvtColor(sample_u8, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sample_u8, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(sample_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

    chroma = np.sqrt((lab[:, :, 1] - 128) ** 2 + (lab[:, :, 2] - 128) ** 2)
    valid = (gray > np.percentile(gray, 14)) & (gray < np.percentile(gray, 96)) & (hsv[:, :, 1] < 82) & (chroma < 18)
    if float(np.mean(valid)) < 0.025:
        valid = (gray > np.percentile(gray, 10)) & (gray < np.percentile(gray, 97)) & (hsv[:, :, 1] < 128) & (chroma < 32)
    if not np.any(valid):
        valid = (gray > np.percentile(gray, 12)) & (gray < np.percentile(gray, 96))
    avg_b, avg_g, avg_r = [float(np.mean(sample[:, :, index][valid])) for index in range(3)]
    gray = (avg_b + avg_g + avg_r) / 3.0
    gains = np.array([gray / max(avg_b, 1.0), gray / max(avg_g, 1.0), gray / max(avg_r, 1.0)])

    max_gain = 1.16 if scene == "portrait" else 1.2
    min_gain = 0.86 if scene == "portrait" else 0.82
    gains = np.clip(gains, min_gain, max_gain)
    if scene in {"landscape", "night"}:
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        sky_green_ratio = float(np.mean(((hue > 38) & (hue < 132) & (sat > 45))))
        if sky_green_ratio > 0.32:
            gains = gains * 0.72 + np.ones(3, dtype=np.float32) * 0.28
    balanced = np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    color_cast = float(max(abs(gains[0] - 1), abs(gains[2] - 1)))
    return balanced, color_cast


def strong_neutral_white_balance(bgr: np.ndarray, scene: str = "general") -> tuple[np.ndarray, float]:
    sample = resize_for_analysis(bgr, 900).astype(np.float32)
    sample_u8 = sample.astype(np.uint8)
    gray = cv2.cvtColor(sample_u8, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sample_u8, cv2.COLOR_BGR2HSV)
    valid = (gray > np.percentile(gray, 8)) & (gray < np.percentile(gray, 98)) & (hsv[:, :, 1] < 165)
    if float(np.mean(valid)) < 0.08:
        valid = (gray > np.percentile(gray, 5)) & (gray < np.percentile(gray, 99))
    avg_b, avg_g, avg_r = [float(np.mean(sample[:, :, index][valid])) for index in range(3)]
    gray_mean = (avg_b + avg_g + avg_r) / 3.0
    gains = np.array([gray_mean / max(avg_b, 1.0), gray_mean / max(avg_g, 1.0), gray_mean / max(avg_r, 1.0)])
    if scene == "product":
        gains = np.clip(gains, 0.76, 1.32)
    else:
        gains = np.clip(gains, 0.8, 1.25)
    balanced = np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    color_cast = float(max(abs(gains[0] - 1), abs(gains[2] - 1)))
    return balanced, color_cast


def auto_exposure_lab(bgr: np.ndarray, intensity: float = 1.0, scene: str = "general") -> tuple[np.ndarray, dict[str, Any]]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    mean_l = float(np.mean(l_channel))
    shadow = float(np.percentile(l_channel, 4))
    highlight = float(np.percentile(l_channel, 96))

    gamma = 1.0
    if mean_l < 92:
        gamma = 0.82
    elif mean_l < 118:
        gamma = 0.9
    elif mean_l > 184:
        gamma = 1.14

    normalized = l_channel.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma) * 255.0
    corrected = percentile_clip(corrected.astype(np.uint8), 0.35, 99.65)

    if scene in {"portrait", "product"}:
        corrected_f = corrected.astype(np.float32)
        target_mean = 126.0 if scene == "portrait" else 184.0
        lift = (target_mean - float(np.mean(corrected_f))) * (0.28 if scene == "portrait" else 0.22) * intensity
        mean_corrected = float(np.mean(corrected_f))
        contrast = 1.035 if scene == "portrait" else 1.02
        corrected_f = (corrected_f - mean_corrected) * contrast + mean_corrected + lift
        corrected_f = cv2.GaussianBlur(corrected_f, (0, 0), 0.35)
        corrected = np.clip(corrected_f, 0, 255).astype(np.uint8)
        merged = cv2.merge((corrected, a_channel, b_channel))
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return result, {
            "meanLight": round(mean_l, 1),
            "shadow": round(shadow, 1),
            "highlight": round(highlight, 1),
            "gamma": round(gamma, 2),
            "contrast": "平滑增强",
        }

    if highlight - shadow > 192:
        clip_limit = 1.08
    elif highlight - shadow > 160:
        clip_limit = 1.18
    else:
        clip_limit = 1.48
    if scene == "general":
        clip_limit *= 0.84
    elif scene == "landscape":
        clip_limit *= 0.9
    clip_limit *= intensity
    clip_limit = float(np.clip(clip_limit, 1.1, 2.0))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    corrected = clahe.apply(corrected)

    merged = cv2.merge((corrected, a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result, {
        "meanLight": round(mean_l, 1),
        "shadow": round(shadow, 1),
        "highlight": round(highlight, 1),
        "gamma": round(gamma, 2),
        "contrast": "柔和增强",
    }


def enhance_color(bgr: np.ndarray, scene: str, intensity: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation_mean = float(np.mean(hsv[:, :, 1]))
    if scene == "portrait":
        saturation_gain = 1.04 if saturation_mean > 70 else 1.08
    elif scene == "landscape":
        saturation_gain = 1.08 if saturation_mean > 95 else 1.12
    elif scene == "night":
        saturation_gain = 1.01 if saturation_mean > 96 else 1.04
    else:
        saturation_gain = 1.06 if saturation_mean > 88 else 1.1
    saturation_gain = 1 + (saturation_gain - 1) * intensity
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_gain, 0, 238 if scene == "portrait" else 245)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + 0.01 * intensity), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def product_background_neutrality(bgr: np.ndarray) -> float:
    sample = resize_for_analysis(bgr, 900)
    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).astype(np.float32)
    light = lab[:, :, 0]
    sat = hsv[:, :, 1]
    bg_mask = (light > np.percentile(light, 58)) & (sat < 115)
    if float(np.mean(bg_mask)) < 0.12:
        bg_mask = light > np.percentile(light, 68)
    if not np.any(bg_mask):
        return 50.0
    a_mean = float(np.mean(lab[:, :, 1][bg_mask]))
    b_mean = float(np.mean(lab[:, :, 2][bg_mask]))
    chroma = float(np.linalg.norm([a_mean - 128.0, b_mean - 128.0]))
    return float(np.clip(100.0 - chroma * 4.2, 0.0, 100.0))


def neutralize_product_background(original: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    sample = resize_for_analysis(candidate, 900)
    lab_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).astype(np.float32)
    light = lab_sample[:, :, 0]
    sat = hsv_sample[:, :, 1]
    height_s, width_s = light.shape
    yy, xx = np.mgrid[0:height_s, 0:width_s]
    central = (xx > width_s * 0.24) & (xx < width_s * 0.76) & (yy > height_s * 0.2) & (yy < height_s * 0.82)
    bg_sample = (light > np.percentile(light, 38)) & (sat < 138) & ~central
    if float(np.mean(bg_sample)) < 0.1:
        bg_sample = (light > np.percentile(light, 50)) & ~central
    if not np.any(bg_sample):
        return candidate

    lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV).astype(np.float32)
    full_light = lab[:, :, 0]
    full_sat = hsv[:, :, 1]
    height, width = full_light.shape
    full_y, full_x = np.mgrid[0:height, 0:width]
    subject_zone = (full_x > width * 0.24) & (full_x < width * 0.76) & (full_y > height * 0.2) & (full_y < height * 0.82)
    mask = ((full_light > np.percentile(full_light, 35)) & (full_sat < 145) & ~subject_zone).astype(np.float32)
    if float(np.mean(mask)) < 0.08:
        return candidate
    mask = cv2.GaussianBlur(mask, (0, 0), 3.0)[:, :, None] * 0.58
    a_delta = float(np.mean(lab_sample[:, :, 1][bg_sample]) - 128.0)
    b_delta = float(np.mean(lab_sample[:, :, 2][bg_sample]) - 128.0)
    lab[:, :, 1] = lab[:, :, 1] - a_delta * mask[:, :, 0]
    lab[:, :, 2] = lab[:, :, 2] - b_delta * mask[:, :, 0]
    neutral = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    before_metrics = weirdness_metrics(candidate, original, "product")
    after_metrics = weirdness_metrics(neutral, original, "product")
    if after_metrics["penalty"] > before_metrics["penalty"] + 1.5:
        return candidate
    return neutral


def recover_tone_detail(bgr: np.ndarray, scene: str, intensity: float = 1.0) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    shadow_mask = np.clip((92 - l_channel) / 92, 0, 1)
    highlight_mask = np.clip((l_channel - 178) / 77, 0, 1)
    shadow_lift = (1.0 - np.power(np.clip(l_channel / 255.0, 0, 1), 0.82)) * 28 * intensity
    highlight_pull = np.power(np.clip((l_channel - 160) / 95, 0, 1), 1.2) * 18 * intensity
    if scene == "night":
        shadow_lift *= 0.55
        highlight_pull *= 1.15
    elif scene == "portrait":
        shadow_lift *= 0.82
        highlight_pull *= 0.8
    elif scene == "general":
        shadow_lift *= 0.86
        highlight_pull *= 0.94
    elif scene == "landscape":
        shadow_lift *= 0.9
        highlight_pull *= 0.96
    l_channel = l_channel + shadow_lift * shadow_mask - highlight_pull * highlight_mask

    blur = cv2.GaussianBlur(l_channel, (0, 0), 18)
    if scene == "portrait":
        local_strength = 0.045
        local_limit = 5
    elif scene == "product":
        local_strength = 0.035
        local_limit = 4
    elif scene == "landscape":
        local_strength = 0.068
        local_limit = 5
    else:
        local_strength = 0.056 if scene == "general" else 0.085
        local_limit = 4 if scene == "general" else 6
    local = np.clip((l_channel - blur) * local_strength * intensity, -local_limit, local_limit)
    lab[:, :, 0] = np.clip(l_channel + local, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def vibrance_color(bgr: np.ndarray, scene: str, intensity: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    headroom = np.clip((175 - sat) / 175, 0, 1)
    gain = 1 + headroom * (0.1 if scene == "portrait" else 0.16) * intensity
    if scene == "night":
        gain = 1 + headroom * 0.07 * intensity
    hsv[:, :, 1] = np.clip(sat * gain, 0, 232 if scene == "portrait" else 242)
    hsv[:, :, 2] = np.clip(val * (1 + 0.006 * intensity), 0, 255)
    vibrant = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if scene == "portrait":
        return protect_skin_tone(bgr, vibrant, strength=0.55)
    return vibrant


def protect_extremes(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shadow_mask = np.clip((34 - gray) / 34, 0, 1)
    highlight_mask = np.clip((gray - 226) / 29, 0, 1)
    mask = np.maximum(shadow_mask, highlight_mask)[:, :, None] * 0.48
    mixed = after.astype(np.float32) * (1 - mask) + before.astype(np.float32) * mask
    return np.clip(mixed, 0, 255).astype(np.uint8)


def smooth_low_texture_background(bgr: np.ndarray, scene: str) -> np.ndarray:
    if scene not in {"portrait", "product"}:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    threshold = np.percentile(grad, 48 if scene == "portrait" else 54)
    mask = (grad < max(threshold, 2.0)).astype(np.float32)
    if scene == "portrait":
        mask *= (1.0 - np.clip(skin_mask(bgr)[:, :, 0] * 1.8, 0, 1))
    else:
        mask *= (hsv[:, :, 1] < 155).astype(np.float32)
    if float(np.mean(mask)) < 0.06:
        return bgr
    mask = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (0, 0), 2.4)[:, :, None]
    blurred = cv2.bilateralFilter(bgr, 7, 18 if scene == "portrait" else 24, 9)
    strength = 0.32 if scene == "portrait" else 0.38
    mixed = bgr.astype(np.float32) * (1 - mask * strength) + blurred.astype(np.float32) * (mask * strength)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def protect_smooth_gradients(original: np.ndarray, candidate: np.ndarray, scene: str) -> np.ndarray:
    if scene == "night":
        return candidate
    original_quality = quality_score_for_scene(original, scene)
    original_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(original_hsv[:, :, 1]))
    if scene == "portrait" and original_quality >= 76 and mean_sat < 72.0:
        return candidate
    if scene == "product" and mean_sat < 56.0:
        return candidate

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    base_blur = cv2.GaussianBlur(original_gray, (0, 0), 2.6)
    resid = np.abs(original_gray - base_blur)
    grad_x = cv2.Sobel(original_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(original_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    smooth = (
        (resid <= max(float(np.percentile(resid, 70)), 0.8 if scene == "portrait" else 1.0))
        & (grad_mag <= max(float(np.percentile(grad_mag, 62)), 2.1 if scene == "portrait" else 2.6))
    ).astype(np.float32)
    if scene == "portrait":
        smooth *= 1.0 - np.clip(skin_mask(original)[:, :, 0] * 1.7, 0, 1)
    edge_map = (grad_mag > float(np.percentile(grad_mag, 76))).astype(np.uint8) * 255
    edge_map = cv2.dilate(edge_map, np.ones((5, 5), np.uint8))
    smooth *= 1.0 - edge_map.astype(np.float32) / 255.0
    if float(np.mean(smooth)) < 0.04:
        return candidate

    mask = cv2.GaussianBlur(smooth, (0, 0), 2.8)[:, :, None]
    candidate_blur = cv2.GaussianBlur(candidate.astype(np.float32), (0, 0), 1.4 if scene == "portrait" else 1.8)
    base_strength = 0.18 if scene == "portrait" else 0.24 if scene == "product" else 0.16
    posterization = np.abs(candidate_gray - cv2.GaussianBlur(candidate_gray, (0, 0), 1.2))
    if float(np.mean(posterization * smooth)) > 1.6:
        base_strength += 0.08
    mixed = candidate.astype(np.float32) * (1 - mask * base_strength) + candidate_blur * (mask * base_strength)

    yy, xx = np.mgrid[0:candidate.shape[0], 0:candidate.shape[1]]
    dither = (((xx * 17 + yy * 29) % 7).astype(np.float32) - 3.0) * (1.1 if scene == "product" else 0.8)
    mixed = mixed + mask * dither[:, :, None]
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)
    mixed = protect_bright_neutral_regions(original, mixed, scene)

    candidate_quality = quality_score_for_scene(candidate, scene)
    mixed_quality = quality_score_for_scene(mixed, scene)
    candidate_metrics = weirdness_metrics(candidate, original, scene)
    mixed_metrics = weirdness_metrics(mixed, original, scene)
    if (
        mixed_quality >= candidate_quality - 2
        and mixed_metrics["penalty"] <= candidate_metrics["penalty"] + 0.6
        and mixed_metrics["colorShift"] <= candidate_metrics["colorShift"] + 0.5
    ):
        return mixed
    return candidate


def lift_backlit_portrait_subject(original: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    height, width = original.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    face = (((xx - width * 0.5) / max(width * 0.2, 1)) ** 2 + ((yy - height * 0.43) / max(height * 0.27, 1)) ** 2) <= 1.0
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
    original_face_l = float(np.mean(original_lab[:, :, 0][face]))
    original_bg_l = float(np.mean(original_lab[:, :, 0][~face]))
    candidate_face_l = float(np.mean(candidate_lab[:, :, 0][face]))
    if original_bg_l - original_face_l < 28 or candidate_face_l >= max(original_face_l + 2, 150):
        return candidate
    target_face_l = min(172.0, max(original_face_l + 10.0, 154.0))
    lift = np.clip(target_face_l - candidate_face_l, 0.0, 46.0)
    if lift <= 1:
        return candidate
    mask = face.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), max(3.0, min(width, height) * 0.018))[:, :, None]
    candidate_lab[:, :, 0] = np.clip(candidate_lab[:, :, 0] + lift * mask[:, :, 0] * 0.82, 0, 255)
    lifted = cv2.cvtColor(candidate_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    before_metrics = weirdness_metrics(candidate, original, "portrait")
    after_metrics = weirdness_metrics(lifted, original, "portrait")
    if (
        after_metrics["penalty"] > before_metrics["penalty"] + 1.5
        and after_metrics["colorShift"] > before_metrics["colorShift"] + 0.5
        and after_metrics["addedClipRatio"] > before_metrics["addedClipRatio"] + 0.002
    ):
        return candidate
    return lifted


def portrait_subject_readability(bgr: np.ndarray) -> float:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    face = (((xx - width * 0.5) / max(width * 0.2, 1)) ** 2 + ((yy - height * 0.43) / max(height * 0.27, 1)) ** 2) <= 1.0
    face_l = float(np.mean(lab[:, :, 0][face]))
    bg_l = float(np.mean(lab[:, :, 0][~face]))
    contrast_gap = max(0.0, bg_l - face_l - 34.0)
    face_score = 100.0 - max(0.0, 146.0 - face_l) * 1.2 - contrast_gap * 0.85
    return float(np.clip(face_score, 0.0, 100.0))


def reduce_noise_and_sharpen(bgr: np.ndarray, scene: str = "general") -> np.ndarray:
    h, w = bgr.shape[:2]
    megapixels = (h * w) / 1_000_000
    if scene == "night":
        denoise_strength = 7 if megapixels >= 8 else 6
    else:
        denoise_strength = 3 if megapixels >= 18 else 2
    if megapixels > 22:
        scale = (22 / megapixels) ** 0.5
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small = cv2.fastNlMeansDenoisingColored(small, None, denoise_strength, denoise_strength, 7, 21)
        denoised = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        denoised = cv2.fastNlMeansDenoisingColored(bgr, None, denoise_strength, denoise_strength, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    blur = cv2.GaussianBlur(l_channel, (0, 0), 1.0)
    if scene == "night":
        sharpen_amount = 0.1
    elif scene in {"portrait", "product"}:
        sharpen_amount = 0.08
    elif scene == "general":
        sharpen_amount = 0.055
    elif scene == "landscape":
        sharpen_amount = 0.072
    else:
        sharpen_amount = 0.18
    sharpened_l = cv2.addWeighted(l_channel, 1 + sharpen_amount, blur, -sharpen_amount, 0)
    sharpened = cv2.cvtColor(cv2.merge((np.clip(sharpened_l, 0, 255).astype(np.uint8), a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    return protect_extremes(denoised, sharpened)


def analyze_reference_image(bgr: np.ndarray, name: str, kind: str, source: str, online: bool) -> ReferenceProfile:
    sample = resize_for_analysis(bgr, 1100)
    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).astype(np.float32)
    l_channel = lab[:, :, 0]
    preview_path = REFERENCE_DIR / f"{safe_filename(name)}.jpg"

    rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(rgb)
    preview.thumbnail((900, 900), Image.Resampling.LANCZOS)
    preview.save(preview_path, format="JPEG", quality=92, optimize=True, progressive=True, subsampling=0)

    strength = 0.2 if kind == "portrait" else 0.24
    return ReferenceProfile(
        name=name,
        kind=kind,
        source=source,
        strength=strength,
        lab_mean=[float(np.mean(lab[:, :, index])) for index in range(3)],
        lab_std=[float(max(np.std(lab[:, :, index]), 4.0)) for index in range(3)],
        hsv_mean=[float(np.mean(hsv[:, :, index])) for index in range(3)],
        contrast=float(max(np.percentile(l_channel, 95) - np.percentile(l_channel, 5), 12.0)),
        preview_path=preview_path,
        online=online,
    )


def verified_reference_kind(bgr: np.ndarray, requested_kind: str) -> str:
    if requested_kind != "product":
        return requested_kind
    scene_info = classify_scene(bgr)
    subject = scene_info.get("subjectRegion", {})
    product_confidence = float(subject.get("productConfidence", 0.0)) if isinstance(subject, dict) else 0.0
    skin_region = float(scene_info.get("skinRegionRatio", 0.0))
    if scene_info.get("scene") != "product" and product_confidence < 0.35:
        return "general" if skin_region > 0.08 else str(scene_info.get("scene", "general"))
    return requested_kind


def load_cached_reference_profiles() -> list[ReferenceProfile]:
    profiles: list[ReferenceProfile] = []
    if not REFERENCE_DIR.exists():
        return profiles
    for path in REFERENCE_DIR.glob("*.jpg"):
        kind = REFERENCE_KIND_BY_NAME.get(path.stem)
        if not kind:
            continue
        try:
            bgr = decode_image_bytes(path.read_bytes())
            kind = verified_reference_kind(bgr, kind)
            profiles.append(analyze_reference_image(bgr, path.stem, kind, "本地缓存网络样片", True))
        except Exception as exc:
            log_event("reference", "本地参考样片读取失败", file=path.name, error=str(exc))
    return profiles


def download_reference(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError("参考样片 URL 必须以 http 或 https 开头。")
    request = Request(url, headers={"User-Agent": "AutoRetouch/1.2"})
    try:
        with urlopen(request, timeout=14) as response:
            content = response.read(12 * 1024 * 1024 + 1)
    except URLError as exc:
        raise RuntimeError(f"无法下载参考样片：{exc}") from exc
    if len(content) > 12 * 1024 * 1024:
        raise RuntimeError("参考样片太大。")
    return content


def load_online_references() -> None:
    loaded: list[ReferenceProfile] = []
    for name, kind, url in REFERENCE_URLS:
        try:
            cache_path = REFERENCE_DIR / f"{safe_filename(name)}.jpg"
            content = cache_path.read_bytes() if cache_path.exists() else download_reference(url)
            bgr = decode_image_bytes(content)
            kind = verified_reference_kind(bgr, kind)
            loaded.append(analyze_reference_image(bgr, name, kind, url, True))
        except Exception:
            continue

    with reference_lock:
        existing_names = {profile.name for profile in reference_bank}
        reference_bank.extend([profile for profile in loaded if profile.name not in existing_names])
        reference_status["updatedAt"] = now_ms()
        if loaded:
            reference_status["message"] = f"已启用内置参考，并自动载入 {len(loaded)} 张网络样片。"
        else:
            reference_status["message"] = "已启用内置参考；网络样片暂未载入，离线也可稳定修图。"


def initialize_reference_bank() -> None:
    cached = load_cached_reference_profiles()
    with reference_lock:
        reference_bank.clear()
        reference_bank.extend(builtin_reference_bank())
        existing_names = {profile.name for profile in reference_bank}
        reference_bank.extend([profile for profile in cached if profile.name not in existing_names])
        reference_status["updatedAt"] = now_ms()
        if cached:
            reference_status["message"] = f"内置审美参考库已启用，并已复用 {len(cached)} 张本地网络样片。"
        else:
            reference_status["message"] = "内置审美参考库已启用，正在尝试补充网络样片。"
    Thread(target=load_online_references, daemon=True).start()


def start_workers() -> None:
    worker_count = int(os.environ.get("WORKERS", "1"))
    worker_count = max(1, min(worker_count, 2))
    for index in range(worker_count):
        Thread(target=worker_loop, name=f"retouch-worker-{index + 1}", daemon=True).start()


def detect_face(sample: np.ndarray) -> bool:
    try:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(str(cascade_path))
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(48, 48))
        return len(faces) > 0
    except Exception:
        return False


def image_texture_profile(bgr: np.ndarray) -> dict[str, Any]:
    sample = resize_for_analysis(bgr, 360)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    residual = float(np.mean(np.abs(gray.astype(np.float32) - blur.astype(np.float32))))
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 60, 140)
    edge_ratio = float(np.mean(edges > 0))
    quantized = (sample // 16).reshape(-1, 3)
    unique_colors = int(len(np.unique(quantized, axis=0)))
    mean_saturation = float(np.mean(hsv[:, :, 1]))

    graphic_score = 0.0
    if residual < 0.75:
        graphic_score += 35.0
    elif residual < 1.2:
        graphic_score += 22.0
    elif residual < 1.8:
        graphic_score += 8.0
    if laplacian < 70:
        graphic_score += 22.0
    elif laplacian < 150:
        graphic_score += 10.0
    if edge_ratio < 0.01:
        graphic_score += 22.0
    elif edge_ratio < 0.018:
        graphic_score += 10.0
    if unique_colors < 80:
        graphic_score += 26.0
    elif unique_colors < 130:
        graphic_score += 16.0
    elif unique_colors < 190:
        graphic_score += 6.0

    return {
        "residual": round(residual, 2),
        "laplacian": round(laplacian, 2),
        "edgeRatio": round(edge_ratio, 4),
        "uniqueColors16": unique_colors,
        "meanSaturation": round(mean_saturation, 1),
        "graphicScore": round(float(np.clip(graphic_score, 0.0, 100.0)), 1),
    }


def classify_scene(bgr: np.ndarray) -> dict[str, Any]:
    sample = resize_for_analysis(bgr, 800)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    mean_l = float(np.mean(gray))
    mean_s = float(np.mean(hsv[:, :, 1]))
    hue = hsv[:, :, 0]
    green_ratio = float(np.mean((hue > 35) & (hue < 95) & (hsv[:, :, 1] > 35)))
    blue_ratio = float(np.mean((hue > 90) & (hue < 130) & (hsv[:, :, 1] > 30)))
    skin_ratio = estimate_skin_ratio(sample)
    skin_region_ratio = largest_skin_region_ratio(sample)
    raw_face_detected = detect_face(sample)
    portrait_region = portrait_region_confidence(sample)
    subject_region = central_subject_region(sample)
    texture = image_texture_profile(sample)
    is_graphic = bool(texture["graphicScore"] >= 76.0)
    trusted_face = raw_face_detected and (
        skin_ratio > 0.028
        or skin_region_ratio > 0.02
        or portrait_region["confidence"] > 0.12
        or subject_region["portraitConfidence"] > 0.18
    )
    portrait_like = portrait_region["confidence"] >= 0.62
    centered_portrait_like = subject_region["portraitConfidence"] >= 0.72
    natural_scene_signal = green_ratio + blue_ratio
    low_light_portrait_like = (
        mean_l < 95
        and
        (skin_region_ratio > 0.055 or skin_ratio > 0.42)
        and 0.045 < skin_ratio < 0.72
        and subject_region["public"].get("area", 0.0) > 0.12
        and 0.25 <= subject_region["public"].get("centerX", 0.0) <= 0.75
        and natural_scene_signal < 0.38
    )
    backlit_portrait_like = (
        skin_region_ratio > 0.055
        and 0.04 < skin_ratio < 0.24
        and subject_region["public"].get("area", 0.0) > 0.12
        and 0.46 <= subject_region["public"].get("centerX", 0.0) <= 0.54
        and 0.48 <= subject_region["public"].get("centerY", 0.0) <= 0.78
        and subject_region["public"].get("aspect", 9.0) < 1.05
        and natural_scene_signal < 0.32
    )
    product_like = subject_region["productConfidence"] >= 0.66
    low_saturation_product_like = (
        mean_s < 48
        and skin_ratio < 0.18
        and skin_region_ratio < 0.14
        and natural_scene_signal < 0.2
    )

    broad_skin_fill = (
        skin_ratio > 0.42
        and skin_region_ratio > 0.42
        and subject_region["public"].get("area", 0.0) > 0.12
        and natural_scene_signal < 0.18
    )
    bright_sky_like = (
        mean_l > 165
        and mean_s < 38
        and natural_scene_signal > 0.16
        and subject_region["public"].get("area", 0.0) > 0.28
        and subject_region["public"].get("touchesEdge", False)
    )
    night_landscape_like = (
        not trusted_face
        and mean_l < 78
        and skin_ratio < 0.05
        and skin_region_ratio < 0.025
        and natural_scene_signal > 0.42
        and subject_region["public"].get("touchesEdge", False)
        and subject_region["public"].get("portraitConfidence", 0.0) < 0.18
    )
    indoor_lifestyle_like = (
        not trusted_face
        and not product_like
        and mean_l < 96
        and mean_s < 96
        and 0.3 < natural_scene_signal < 0.78
        and skin_ratio < 0.16
        and skin_region_ratio < 0.05
        and subject_region["public"].get("area", 0.0) > 0.34
        and subject_region["public"].get("touchesEdge", False)
        and texture["edgeRatio"] > 0.07
    )

    portrait_scene_like = (
        trusted_face
        or (portrait_like and 0.045 < skin_ratio < 0.42 and skin_region_ratio > 0.035 and natural_scene_signal < 0.62)
        or centered_portrait_like
        or low_light_portrait_like
        or backlit_portrait_like
        or broad_skin_fill
    )

    product_override_portrait = product_like and not (trusted_face or low_light_portrait_like or backlit_portrait_like)

    if product_override_portrait:
        scene = "product"
    elif night_landscape_like:
        scene = "night"
    elif portrait_scene_like:
        scene = "portrait"
    elif low_saturation_product_like:
        scene = "product"
    elif indoor_lifestyle_like:
        scene = "general"
    elif mean_l < 72:
        scene = "night"
    elif natural_scene_signal > 0.22 or bright_sky_like:
        scene = "landscape"
    else:
        scene = "general"

    if scene == "night" and texture["residual"] > 0.9:
        is_graphic = False

    return {
        "scene": scene,
        "meanLight": round(mean_l, 1),
        "meanSaturation": round(mean_s, 1),
        "skinRatio": round(skin_ratio, 3),
        "skinRegionRatio": round(skin_region_ratio, 3),
        "portraitRegionConfidence": round(portrait_region["confidence"], 3),
        "portraitRegion": portrait_region["public"],
        "subjectRegion": subject_region["public"],
        "faceDetected": trusted_face,
        "rawFaceDetected": raw_face_detected,
        "greenBlueRatio": round(natural_scene_signal, 3),
        "texture": texture,
        "isGraphic": is_graphic,
    }


def estimate_skin_ratio(bgr: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (cr > 136) & (cr < 174) & (cb > 82) & (cb < 128) & (y > 50) & (hsv[:, :, 1] > 16) & (hsv[:, :, 1] < 172)
    return float(np.mean(mask))


def largest_skin_region_ratio(bgr: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((cr > 136) & (cr < 174) & (cb > 82) & (cb < 128) & (y > 50) & (hsv[:, :, 1] > 16) & (hsv[:, :, 1] < 172)).astype(np.uint8)
    if float(np.mean(mask)) < 0.005:
        return 0.0
    opened = cv2.morphologyEx(mask * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    if count <= 1:
        return 0.0
    largest = float(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return largest / float(mask.shape[0] * mask.shape[1])


def portrait_region_confidence(bgr: np.ndarray) -> dict[str, Any]:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((cr > 136) & (cr < 174) & (cb > 82) & (cb < 128) & (y > 50) & (hsv[:, :, 1] > 16) & (hsv[:, :, 1] < 172)).astype(np.uint8)
    if float(np.mean(mask)) < 0.012:
        return {"confidence": 0.0, "public": {"area": 0.0}}

    opened = cv2.morphologyEx(mask * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(opened, 8)
    if count <= 1:
        return {"confidence": 0.0, "public": {"area": 0.0}}

    height, width = mask.shape[:2]
    best_index = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[best_index, cv2.CC_STAT_AREA])
    x = float(stats[best_index, cv2.CC_STAT_LEFT])
    y_pos = float(stats[best_index, cv2.CC_STAT_TOP])
    box_w = float(stats[best_index, cv2.CC_STAT_WIDTH])
    box_h = float(stats[best_index, cv2.CC_STAT_HEIGHT])
    cx, cy = (float(value) for value in centroids[best_index])

    area_ratio = area / float(width * height)
    width_ratio = box_w / float(width)
    height_ratio = box_h / float(height)
    aspect = box_w / max(box_h, 1.0)
    center_x = cx / float(width)
    center_y = cy / float(height)
    touches_edge = x <= 4 or y_pos <= 4 or x + box_w >= width - 4 or y_pos + box_h >= height - 4

    score = 1.0
    if not 0.035 <= area_ratio <= 0.36:
        score *= 0.38
    if not 0.36 <= aspect <= 1.55:
        score *= 0.22
    if not 0.2 <= height_ratio <= 0.84:
        score *= 0.45
    if width_ratio > 0.58:
        score *= 0.35
    if not 0.28 <= center_x <= 0.72:
        score *= 0.45
    if not 0.22 <= center_y <= 0.78:
        score *= 0.65
    if touches_edge:
        score *= 0.28

    public = {
        "area": round(area_ratio, 3),
        "aspect": round(aspect, 2),
        "width": round(width_ratio, 3),
        "height": round(height_ratio, 3),
        "centerX": round(center_x, 3),
        "centerY": round(center_y, 3),
        "touchesEdge": touches_edge,
    }
    return {"confidence": float(np.clip(score, 0.0, 1.0)), "public": public}


def central_subject_region(bgr: np.ndarray) -> dict[str, Any]:
    sample = resize_for_analysis(bgr, 800)
    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = sample.shape[:2]
    border_width = max(8, int(min(height, width) * 0.06))
    border = np.concatenate(
        [
            lab[:border_width].reshape(-1, 3),
            lab[-border_width:].reshape(-1, 3),
            lab[:, :border_width].reshape(-1, 3),
            lab[:, -border_width:].reshape(-1, 3),
        ],
        axis=0,
    )
    border_median = np.median(border, axis=0)
    distance = np.linalg.norm((lab - border_median) / np.array([30, 12, 12], dtype=np.float32), axis=2)
    mask = (distance > 1.05).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        empty = {"area": 0.0, "portraitConfidence": 0.0, "productConfidence": 0.0}
        return {"portraitConfidence": 0.0, "productConfidence": 0.0, "public": empty}

    best_index = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[best_index, cv2.CC_STAT_AREA])
    x = float(stats[best_index, cv2.CC_STAT_LEFT])
    y_pos = float(stats[best_index, cv2.CC_STAT_TOP])
    box_w = float(stats[best_index, cv2.CC_STAT_WIDTH])
    box_h = float(stats[best_index, cv2.CC_STAT_HEIGHT])
    cx, cy = (float(value) for value in centroids[best_index])
    area_ratio = area / float(width * height)
    width_ratio = box_w / float(width)
    height_ratio = box_h / float(height)
    aspect = box_w / max(box_h, 1.0)
    center_x = cx / float(width)
    center_y = cy / float(height)
    touches_edge = x <= 3 or y_pos <= 3 or x + box_w >= width - 3 or y_pos + box_h >= height - 3

    portrait_score = 1.0
    if not 0.12 <= area_ratio <= 0.38:
        portrait_score *= 0.35
    if not 0.42 <= aspect <= 0.98:
        portrait_score *= 0.32
    if not 0.42 <= height_ratio <= 0.84:
        portrait_score *= 0.42
    if not 0.22 <= width_ratio <= 0.5:
        portrait_score *= 0.45
    if not 0.34 <= center_x <= 0.66:
        portrait_score *= 0.36
    if not 0.34 <= center_y <= 0.72:
        portrait_score *= 0.48
    if touches_edge:
        portrait_score *= 0.2

    product_score = 1.0
    if not 0.07 <= area_ratio <= 0.32:
        product_score *= 0.4
    if not 0.72 <= aspect <= 1.85:
        product_score *= 0.45
    if not 0.18 <= height_ratio <= 0.56:
        product_score *= 0.55
    if not 0.22 <= width_ratio <= 0.62:
        product_score *= 0.55
    if not 0.34 <= center_x <= 0.66:
        product_score *= 0.36
    if not 0.34 <= center_y <= 0.72:
        product_score *= 0.55
    if touches_edge:
        product_score *= 0.22

    public = {
        "area": round(area_ratio, 3),
        "aspect": round(aspect, 2),
        "width": round(width_ratio, 3),
        "height": round(height_ratio, 3),
        "centerX": round(center_x, 3),
        "centerY": round(center_y, 3),
        "touchesEdge": touches_edge,
        "portraitConfidence": round(float(np.clip(portrait_score, 0.0, 1.0)), 3),
        "productConfidence": round(float(np.clip(product_score, 0.0, 1.0)), 3),
    }
    return {
        "portraitConfidence": float(np.clip(portrait_score, 0.0, 1.0)),
        "productConfidence": float(np.clip(product_score, 0.0, 1.0)),
        "public": public,
    }


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((cr > 133) & (cr < 178) & (cb > 77) & (cb < 135) & (y > 45)) | (
        (hsv[:, :, 0] < 24) & (hsv[:, :, 1] > 18) & (hsv[:, :, 2] > 55)
    )
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.medianBlur(mask_u8, 5)
    mask_u8 = cv2.GaussianBlur(mask_u8, (0, 0), 2.2)
    return (mask_u8.astype(np.float32) / 255.0)[:, :, None]


def protect_skin_tone(original: np.ndarray, candidate: np.ndarray, strength: float = 0.42) -> np.ndarray:
    mask = skin_mask(original)
    if float(np.mean(mask)) < 0.015:
        return candidate
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
    protected_lab = candidate_lab.copy()
    protected_lab[:, :, 1] = candidate_lab[:, :, 1] * (1 - mask[:, :, 0] * strength) + original_lab[:, :, 1] * (mask[:, :, 0] * strength)
    protected_lab[:, :, 2] = candidate_lab[:, :, 2] * (1 - mask[:, :, 0] * strength) + original_lab[:, :, 2] * (mask[:, :, 0] * strength)
    protected = cv2.cvtColor(np.clip(protected_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return protected


def portrait_face_region_mask(bgr: np.ndarray) -> np.ndarray:
    height, width = bgr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cx = width * 0.5
    cy = height * 0.42
    rx = width * 0.18
    ry = height * 0.24
    ellipse = (((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2) <= 1.0
    mask = ellipse.astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, min(width, height) * 0.008))
    return (mask.astype(np.float32) / 255.0)[:, :, None]


def normalize_portrait_skin_region(original: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    detected_mask = skin_mask(candidate)
    face_mask = portrait_face_region_mask(candidate)
    fallback_mask = float(np.mean(detected_mask)) < 0.035
    mask = detected_mask if not fallback_mask else np.maximum(detected_mask, face_mask * 0.42)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
    weighted = mask[:, :, 0] > 0.08
    if not np.any(weighted):
        return candidate
    a_mean = float(np.average(candidate_lab[:, :, 1][weighted], weights=mask[:, :, 0][weighted]))
    b_mean = float(np.average(candidate_lab[:, :, 2][weighted], weights=mask[:, :, 0][weighted]))
    target_a = 133.0 if fallback_mask else 132.0
    target_b = 145.0 if fallback_mask and b_mean < 132.0 else 139.0
    a_delta = np.clip(target_a - a_mean, -5.0, 9.0 if fallback_mask else 6.5)
    b_delta = np.clip(target_b - b_mean, -5.0, 14.0 if fallback_mask else 8.0)
    if abs(a_delta) + abs(b_delta) < 2.0:
        return candidate
    adjusted_lab = candidate_lab.copy()
    strength = 0.82 if fallback_mask else 0.62
    adjusted_lab[:, :, 1] = adjusted_lab[:, :, 1] + a_delta * mask[:, :, 0] * strength
    adjusted_lab[:, :, 2] = adjusted_lab[:, :, 2] + b_delta * mask[:, :, 0] * strength
    adjusted = cv2.cvtColor(np.clip(adjusted_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    before_metrics = weirdness_metrics(candidate, original, "portrait")
    after_metrics = weirdness_metrics(adjusted, original, "portrait")
    if after_metrics["penalty"] > before_metrics["penalty"] + 1.2:
        return candidate
    return adjusted


def protect_natural_colors(original: np.ndarray, candidate: np.ndarray, scene: str) -> np.ndarray:
    if scene not in {"landscape", "general"}:
        return candidate
    original_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV).astype(np.float32)
    candidate_hsv = cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV).astype(np.float32)
    original_quality = quality_score_for_scene(original, scene)
    hue = original_hsv[:, :, 0]
    sat = original_hsv[:, :, 1]
    val = original_hsv[:, :, 2]
    sky = ((hue > 88) & (hue < 132) & (sat > 25) & (val > 80)).astype(np.float32)
    water = ((hue > 84) & (hue < 136) & (sat > 18) & (val > 34)).astype(np.float32)
    green = ((hue > 35) & (hue < 92) & (sat > 35)).astype(np.float32)
    warm = (((hue < 18) | (hue > 166)) & (sat > 32) & (val > 26)).astype(np.float32)
    mask = np.maximum(np.maximum(sky * 0.34, water * 0.22), np.maximum(green * 0.28, warm * 0.18))
    if float(np.mean(mask)) < 0.01:
        return candidate
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0)[:, :, None]
    mixed_hsv = candidate_hsv.copy()
    hue_strength = 0.76 if original_quality >= 90 else 0.62 if scene == "general" else 0.54
    hue_weight = np.clip(mask[:, :, 0] * hue_strength, 0, 1)
    hue_delta = (original_hsv[:, :, 0] - candidate_hsv[:, :, 0] + 90.0) % 180.0 - 90.0
    mixed_hsv[:, :, 0] = (candidate_hsv[:, :, 0] + hue_delta * hue_weight) % 180.0

    sat_cap = original_hsv[:, :, 1] * (1.1 if scene == "general" else 1.14)
    if original_quality >= 94:
        sat_cap = original_hsv[:, :, 1] * 1.04
    elif original_quality >= 90:
        sat_cap = original_hsv[:, :, 1] * 1.07
    sat_cap = np.maximum(sat_cap, original_hsv[:, :, 1] + 3.0)
    water_mask = cv2.GaussianBlur(np.maximum(sky, water), (0, 0), 2.0)
    warm_mask = cv2.GaussianBlur(warm, (0, 0), 1.8)
    water_cap = np.maximum(original_hsv[:, :, 1] * (1.02 if original_quality >= 90 else 1.06), original_hsv[:, :, 1] + 2.0)
    warm_cap = np.maximum(original_hsv[:, :, 1] * (1.03 if original_quality >= 90 else 1.08), original_hsv[:, :, 1] + 2.0)
    cap = sat_cap * (1 - water_mask) + np.minimum(sat_cap, water_cap) * water_mask
    cap = cap * (1 - warm_mask) + np.minimum(cap, warm_cap) * warm_mask
    mixed_hsv[:, :, 1] = np.minimum(candidate_hsv[:, :, 1], cap)
    return cv2.cvtColor(np.clip(mixed_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def protect_bright_neutral_regions(original: np.ndarray, candidate: np.ndarray, scene: str) -> np.ndarray:
    if scene == "night":
        return candidate
    original_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV).astype(np.float32)
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.linalg.norm(original_lab[:, :, 1:3] - np.array([128.0, 128.0], dtype=np.float32), axis=2)
    bright_neutral = (
        (original_lab[:, :, 0] > 170)
        & (original_hsv[:, :, 1] < 82)
        & (chroma < 30)
    ).astype(np.float32)
    if float(np.mean(bright_neutral)) < 0.012:
        return candidate
    mask = cv2.GaussianBlur(bright_neutral, (0, 0), 2.6)[:, :, None]
    strength = 0.72 if scene in {"landscape", "general"} else 0.58
    protected_lab = candidate_lab.copy()
    protected_lab[:, :, 1] = candidate_lab[:, :, 1] * (1 - mask[:, :, 0] * strength) + original_lab[:, :, 1] * (mask[:, :, 0] * strength)
    protected_lab[:, :, 2] = candidate_lab[:, :, 2] * (1 - mask[:, :, 0] * strength) + original_lab[:, :, 2] * (mask[:, :, 0] * strength)
    protected_lab[:, :, 0] = candidate_lab[:, :, 0] * (1 - mask[:, :, 0] * 0.18) + original_lab[:, :, 0] * (mask[:, :, 0] * 0.18)
    return cv2.cvtColor(np.clip(protected_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def protect_shadow_readability(original: np.ndarray, candidate: np.ndarray, scene: str) -> np.ndarray:
    if scene in {"portrait", "night"}:
        return candidate
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    source_shadow = original_gray < (92 if scene == "portrait" else 104)
    darkened = (candidate_gray + 8) < original_gray
    broad_dark_area = float(np.mean((candidate_gray < 58) & source_shadow))
    if float(np.mean(source_shadow & darkened)) < 0.035 and broad_dark_area < 0.06:
        return candidate

    mask = np.clip((original_gray - candidate_gray - 4) / 34, 0, 1)
    mask *= np.clip((112 - candidate_gray) / 70, 0, 1)
    if scene == "portrait":
        mask *= 0.42
    else:
        mask *= 0.62
    mask = cv2.GaussianBlur(mask, (0, 0), 2.2)[:, :, None]
    mixed = candidate.astype(np.float32) * (1 - mask) + original.astype(np.float32) * mask
    return np.clip(mixed, 0, 255).astype(np.uint8)


def preserve_scene_atmosphere(original: np.ndarray, candidate: np.ndarray, scene: str) -> np.ndarray:
    if scene not in {"general", "landscape"}:
        return candidate

    original_quality = quality_score_for_scene(original, scene)
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
    original_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV).astype(np.float32)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)

    original_temp = float(np.mean(original_lab[:, :, 2]))
    candidate_temp = float(np.mean(candidate_lab[:, :, 2]))
    cool_shift = max(0.0, original_temp - candidate_temp - (1.1 if scene == "general" else 1.5))
    adjusted_lab = candidate_lab.copy()

    if cool_shift > 0.0:
        hue = original_hsv[:, :, 0]
        sat = original_hsv[:, :, 1]
        val = original_hsv[:, :, 2]
        warm = (((hue < 18) | (hue > 166)) & (sat > 24) & (val > 30)).astype(np.float32)
        neutral = ((val > 110) & (sat < 84)).astype(np.float32)
        natural = (
            (((hue > 35) & (hue < 92) & (sat > 34)) | ((hue > 84) & (hue < 136) & (sat > 20) & (val > 46)))
        ).astype(np.float32)
        mood_mask = np.maximum(warm * 0.62, np.maximum(neutral * 0.42, natural * (0.28 if scene == "landscape" else 0.18)))
        if scene == "general" and float(np.mean(original_gray)) < 124:
            mood_mask = np.maximum(mood_mask, (sat < 104).astype(np.float32) * 0.16)
        mood_mask = cv2.GaussianBlur(mood_mask, (0, 0), 2.4)[:, :, None]
        strength = float(np.clip(0.16 + cool_shift * 0.075 + max(0, original_quality - 86) * 0.008, 0.16, 0.44))
        adjusted_lab[:, :, 1] = candidate_lab[:, :, 1] * (1 - mood_mask[:, :, 0] * strength) + original_lab[:, :, 1] * (
            mood_mask[:, :, 0] * strength
        )
        adjusted_lab[:, :, 2] = candidate_lab[:, :, 2] * (1 - mood_mask[:, :, 0] * strength) + original_lab[:, :, 2] * (
            mood_mask[:, :, 0] * strength
        )
        if original_quality >= 88:
            adjusted_lab[:, :, 0] = candidate_lab[:, :, 0] * (1 - mood_mask[:, :, 0] * 0.08) + original_lab[:, :, 0] * (
                mood_mask[:, :, 0] * 0.08
            )

    adjusted = cv2.cvtColor(np.clip(adjusted_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    adjusted_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY).astype(np.float32)
    original_sharpness = float(cv2.Laplacian(original_gray, cv2.CV_32F).var())
    sharpness_gain = float(cv2.Laplacian(adjusted_gray, cv2.CV_32F).var() - original_sharpness)
    sharpness_limit = 36.0 if scene == "general" else 52.0
    if sharpness_gain > sharpness_limit:
        base_blur = cv2.GaussianBlur(original_gray, (0, 0), 2.2)
        resid = np.abs(original_gray - base_blur)
        grad_x = cv2.Sobel(original_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(original_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        smooth = (
            (resid <= max(float(np.percentile(resid, 70)), 0.9))
            & (grad_mag <= max(float(np.percentile(grad_mag, 62)), 2.5))
        ).astype(np.float32)
        if float(np.mean(smooth)) > 0.04:
            mask = cv2.GaussianBlur(smooth, (0, 0), 2.4)[:, :, None]
            softened = cv2.GaussianBlur(adjusted.astype(np.float32), (0, 0), 1.35 if scene == "general" else 1.15)
            strength = float(np.clip((sharpness_gain - sharpness_limit) / 140.0, 0.1, 0.32))
            adjusted = adjusted.astype(np.float32) * (1 - mask * strength) + softened * (mask * strength)
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
            adjusted_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY).astype(np.float32)
            sharpness_gain = float(cv2.Laplacian(adjusted_gray, cv2.CV_32F).var() - original_sharpness)

    if sharpness_gain > sharpness_limit + (18.0 if scene == "general" else 24.0):
        blend_cap = 0.18 if scene == "general" else 0.14
        if scene == "landscape" and original_quality >= 94:
            blend_cap = 0.2
        blend = float(np.clip((sharpness_gain - sharpness_limit) / (220.0 if scene == "landscape" and original_quality >= 94 else 240.0), 0.06, blend_cap))
        mixed = cv2.addWeighted(original, blend, adjusted, 1 - blend, 0)
        adjusted_quality = quality_score_for_scene(adjusted, scene)
        mixed_quality = quality_score_for_scene(mixed, scene)
        adjusted_metrics = weirdness_metrics(adjusted, original, scene)
        mixed_metrics = weirdness_metrics(mixed, original, scene)
        if (
            mixed_metrics["penalty"] <= adjusted_metrics["penalty"] - 0.18
            and mixed_metrics["sharpnessShift"] <= adjusted_metrics["sharpnessShift"] - 8.0
            and mixed_quality >= adjusted_quality - 2
        ):
            adjusted = mixed

    if scene == "landscape" and original_quality >= 94:
        adjusted_metrics = weirdness_metrics(adjusted, original, scene)
        if adjusted_metrics["sharpnessShift"] > 70.0:
            blend = float(np.clip((adjusted_metrics["sharpnessShift"] - 70.0) / 180.0, 0.08, 0.2))
            mixed = cv2.addWeighted(original, blend, adjusted, 1 - blend, 0)
            mixed_metrics = weirdness_metrics(mixed, original, scene)
            mixed_quality = quality_score_for_scene(mixed, scene)
            adjusted_quality = quality_score_for_scene(adjusted, scene)
            if (
                mixed_metrics["penalty"] <= adjusted_metrics["penalty"] - 0.25
                and mixed_metrics["sharpnessShift"] <= adjusted_metrics["sharpnessShift"] - 12.0
                and mixed_quality >= adjusted_quality - 2
            ):
                adjusted = mixed

    original_shadow_threshold = min(116.0 if scene == "general" else 108.0, float(np.percentile(original_gray, 58)))
    original_shadow_mask = (original_gray < original_shadow_threshold).astype(np.float32)
    original_shadow_sat_mask = (original_hsv[:, :, 1] < (98 if scene == "general" else 88)).astype(np.float32)
    original_shadow_weight = cv2.GaussianBlur(original_shadow_mask * original_shadow_sat_mask, (0, 0), 2.4)[:, :, None]
    if float(np.mean(original_shadow_weight)) > 0.025:
        adjusted_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY).astype(np.float32)
        shadow_lift_now = max(
            0.0,
            float(
                np.sum((adjusted_gray - original_gray) * original_shadow_weight[:, :, 0])
                / (float(np.sum(original_shadow_weight[:, :, 0])) + 1e-6)
            )
            - (7.5 if scene == "general" else 6.2)
        )
        if shadow_lift_now > 0.0:
            blend = float(np.clip(shadow_lift_now / 42.0, 0.07, 0.24 if scene == "general" else 0.18))
            adjusted = (
                adjusted.astype(np.float32) * (1 - original_shadow_weight * blend)
                + original.astype(np.float32) * (original_shadow_weight * blend)
            )
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    original_blur = cv2.GaussianBlur(original_gray, (0, 0), 1.6)
    adjusted_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY).astype(np.float32)
    adjusted_blur = cv2.GaussianBlur(adjusted_gray, (0, 0), 1.6)
    shadow_limit = min(118.0 if scene == "general" else 108.0, float(np.percentile(original_gray, 58)))
    shadow_mask = (original_gray < shadow_limit).astype(np.float32)
    low_sat_mask = (original_hsv[:, :, 1] < (92 if scene == "general" else 86)).astype(np.float32)
    atmosphere_mask = cv2.GaussianBlur(shadow_mask * low_sat_mask, (0, 0), 2.6)[:, :, None]
    original_shadow_texture = float(np.mean(np.abs(original_gray - original_blur) * shadow_mask))
    adjusted_shadow_texture = float(np.mean(np.abs(adjusted_gray - adjusted_blur) * shadow_mask))
    excess_shadow_texture = adjusted_shadow_texture - original_shadow_texture
    if excess_shadow_texture > (0.3 if scene == "general" else 0.46) and float(np.mean(atmosphere_mask)) > 0.025:
        blend = float(np.clip(excess_shadow_texture / 6.8, 0.08, 0.24 if scene == "general" else 0.2))
        mixed = adjusted.astype(np.float32) * (1 - atmosphere_mask * blend) + original.astype(np.float32) * (atmosphere_mask * blend)
        adjusted = np.clip(mixed, 0, 255).astype(np.uint8)

    adjusted = protect_natural_colors(original, adjusted, scene)
    adjusted = protect_bright_neutral_regions(original, adjusted, scene)
    return adjusted


def protect_high_quality_original(original: np.ndarray, candidate: np.ndarray, scene: str, before_score: int) -> tuple[np.ndarray, bool]:
    protect_threshold = 84 if scene == "portrait" else 86 if scene in {"product", "general"} else 90 if scene == "landscape" else 88 if scene == "night" else 92
    if before_score < protect_threshold:
        return candidate, False
    original_metrics = weirdness_metrics(original, original, scene)
    candidate_metrics = weirdness_metrics(candidate, original, scene)
    candidate_score = quality_score_for_scene(candidate, scene)
    sat_limit = 7.0 if scene == "portrait" else 8.0 if scene in {"product", "general"} else 11.5 if scene == "night" else 9.0 if scene == "landscape" else 11.0
    color_limit = 4.8 if scene == "portrait" else 5.8 if scene in {"product", "general"} else 6.6 if scene == "night" else 6.0 if scene == "landscape" else 7.0
    light_limit = 14.0 if scene == "night" else 14.0 if scene == "portrait" else 9.0 if scene == "landscape" else 10.0

    def perceptual_excess(metrics: dict[str, float]) -> float:
        return (
            max(0.0, metrics["saturationShift"] - sat_limit) / 18.0
            + max(0.0, metrics["colorShift"] - color_limit) / 12.0
            + max(0.0, metrics["lightShift"] - light_limit) / 18.0
            + max(0.0, metrics.get("coolShift", 0.0) - (1.2 if scene == "general" else 1.8 if scene == "landscape" else 99.0)) / 4.0
            + max(0.0, metrics.get("sharpnessShift", 0.0) - (46.0 if scene == "general" else 62.0 if scene == "landscape" else 999.0)) / 120.0
            + max(0.0, metrics["penalty"] - 4.0) / 18.0
        )

    excess = perceptual_excess(candidate_metrics)
    if excess <= 0:
        return candidate, False

    best = candidate
    best_metrics = candidate_metrics
    best_quality = candidate_score
    best_excess = excess
    for blend in (float(np.clip(excess, 0.14, 0.46)), 0.22, 0.32, 0.42):
        mixed = cv2.addWeighted(original, blend, candidate, 1 - blend, 0)
        mixed = protect_extremes(original, mixed)
        metrics = weirdness_metrics(mixed, original, scene)
        score = quality_score_for_scene(mixed, scene)
        mixed_excess = perceptual_excess(metrics)
        if (
            mixed_excess <= best_excess - 0.04
            and metrics["penalty"] <= max(best_metrics["penalty"] + 0.6, 1.0)
            and score >= max(before_score - 1, best_quality - 3)
        ):
            best, best_metrics, best_quality, best_excess = mixed, metrics, score, mixed_excess

    if best is candidate:
        return candidate, False
    if best_quality < max(before_score - 1, candidate_score - 3):
        return candidate, False
    if best_metrics["penalty"] > max(6.0, original_metrics["penalty"] + 3.0, candidate_metrics["penalty"]):
        return candidate, False
    return best, True


def final_finish(bgr: np.ndarray, scene: str) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_float = l_channel.astype(np.float32)
    highlight = l_float > 236
    if np.any(highlight):
        l_float[highlight] = 236 + (l_float[highlight] - 236) * 0.55
    shadow = l_float < 14
    if np.any(shadow):
        l_float[shadow] = l_float[shadow] * 0.72 + 4
    l_channel = np.clip(l_float, 0, 255).astype(np.uint8)
    finished = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    if scene != "portrait":
        blur_sigma = 2.8 if scene == "general" else 2.65 if scene == "landscape" else 2.4
        detail_boost = 0.012 if scene == "general" else 0.018 if scene == "landscape" else 0.08
        blur = cv2.GaussianBlur(finished, (0, 0), blur_sigma)
        detail = cv2.addWeighted(finished, 1 + detail_boost, blur, -detail_boost, 0)
        finished = protect_extremes(finished, detail)

    return np.clip(finished, 0, 255).astype(np.uint8)


def profile_distance(bgr: np.ndarray, profile: ReferenceProfile) -> float:
    sample = resize_for_analysis(bgr, 900)
    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab_mean = np.array([np.mean(lab[:, :, index]) for index in range(3)])
    hsv_mean = np.array([np.mean(hsv[:, :, index]) for index in range(3)])
    target_lab = np.array(profile.lab_mean)
    target_hsv = np.array(profile.hsv_mean)
    return float(np.linalg.norm((lab_mean - target_lab) / np.array([36, 13, 13])) + abs(hsv_mean[1] - target_hsv[1]) / 35)


def choose_reference_profile(bgr: np.ndarray, scene: str) -> ReferenceProfile:
    bank, _ = reference_snapshot()
    if not bank:
        bank = builtin_reference_bank()
    candidates = [profile for profile in bank if profile.kind == scene]
    if not candidates:
        candidates = [profile for profile in bank if profile.kind == "general"]
    if not candidates:
        candidates = bank
    return min(candidates, key=lambda profile: profile_distance(bgr, profile))


def apply_reference_style(bgr: np.ndarray, profile: ReferenceProfile, scene: str, intensity: float = 1.0) -> tuple[np.ndarray, str]:
    strength = float(np.clip(profile.strength * intensity, 0.06, 0.24))
    if scene == "portrait":
        strength = min(strength, 0.16)
    elif scene == "night":
        strength = min(strength, 0.11)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    result_lab = lab.copy()

    for index in range(3):
        src = lab[:, :, index]
        src_mean = float(np.mean(src))
        src_std = float(max(np.std(src), 4.0))
        target_mean = profile.lab_mean[index]
        target_std = profile.lab_std[index]
        if index == 0:
            channel_strength = strength * 0.5
            max_delta = 16
        else:
            channel_strength = strength * (0.24 if scene == "portrait" else 0.32)
            max_delta = 5 if scene == "portrait" else 8
        matched = (src - src_mean) * (target_std / src_std) + target_mean
        delta = np.clip((matched - src) * channel_strength, -max_delta, max_delta)
        result_lab[:, :, index] = src + delta

    styled = cv2.cvtColor(np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(styled, cv2.COLOR_BGR2HSV).astype(np.float32)
    src_sat = float(np.mean(hsv[:, :, 1]))
    ref_sat = profile.hsv_mean[1] if len(profile.hsv_mean) > 1 else src_sat
    sat_gain = np.clip(
        ref_sat / max(src_sat, 1.0),
        0.94 if scene == "portrait" else 0.92 if scene == "night" else 0.9,
        1.08 if scene == "portrait" else 1.06 if scene == "night" else 1.14,
    )
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + (sat_gain - 1) * strength * 0.8), 0, 238 if scene == "portrait" else 245)
    styled = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    final = cv2.addWeighted(bgr, 1 - strength * 0.42, styled, strength * 0.42, 0)
    guarded = aesthetic_guardrails(bgr, final, scene)
    return guarded, f"自动参考：{profile.name}"


def aesthetic_guardrails(before: np.ndarray, after: np.ndarray, scene: str) -> np.ndarray:
    before_hsv = cv2.cvtColor(before, cv2.COLOR_BGR2HSV).astype(np.float32)
    after_hsv = cv2.cvtColor(after, cv2.COLOR_BGR2HSV).astype(np.float32)
    before_sat = float(np.mean(before_hsv[:, :, 1]))
    after_sat = float(np.mean(after_hsv[:, :, 1]))
    if scene == "product":
        max_sat = max(before_sat * 1.12, 72)
    elif scene == "night":
        max_sat = max(before_sat * 1.06, 76)
    elif scene == "landscape":
        max_sat = max(before_sat * 1.18, 86)
    elif scene == "general":
        max_sat = max(before_sat * 1.14, 82)
    else:
        max_sat = max(before_sat * (1.22 if scene == "portrait" else 1.35), 70 if scene == "portrait" else 95)
    min_sat = before_sat * 0.78
    if after_sat > max_sat:
        after_hsv[:, :, 1] *= max_sat / max(after_sat, 1)
    elif after_sat < min_sat:
        after_hsv[:, :, 1] *= min_sat / max(after_sat, 1)

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    light_shift = float(np.mean(after_gray) - np.mean(before_gray))
    if abs(light_shift) > 24:
        ratio = 24 / abs(light_shift)
        after_hsv[:, :, 2] = before_hsv[:, :, 2] + (after_hsv[:, :, 2] - before_hsv[:, :, 2]) * ratio

    candidate = cv2.cvtColor(np.clip(after_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    before_lab = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab_shift = float(np.linalg.norm(np.mean(candidate_lab[:, :, 1:3], axis=(0, 1)) - np.mean(before_lab[:, :, 1:3], axis=(0, 1))))
    max_shift = 9 if scene == "portrait" else 10 if scene == "product" else 14
    if ab_shift > max_shift:
        blend = max_shift / ab_shift
        candidate = cv2.addWeighted(before, 1 - blend, candidate, blend, 0)
    if scene == "portrait":
        candidate = protect_skin_tone(before, candidate)
        candidate = normalize_portrait_skin_region(before, candidate)
    else:
        candidate = protect_natural_colors(before, candidate, scene)
    candidate = protect_bright_neutral_regions(before, candidate, scene)
    return candidate


def quality_score(bgr: np.ndarray) -> int:
    sample = resize_for_analysis(bgr, 1000)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    contrast = np.percentile(gray, 95) - np.percentile(gray, 5)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    exposure = 100 - abs(float(np.mean(gray)) - 136) * 0.62
    sat = float(np.mean(hsv[:, :, 1]))
    saturation_score = 100 - max(0, sat - 118) * 0.8 - max(0, 34 - sat) * 0.9
    score = exposure * 0.38 + min(contrast, 150) * 0.28 + min(sharpness / 12, 100) * 0.18 + saturation_score * 0.16
    return int(np.clip(score, 1, 99))


def graphic_quality_score(bgr: np.ndarray, scene: str) -> int:
    sample = resize_for_analysis(bgr, 1000)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_light = float(np.mean(gray))
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    sat = float(np.mean(hsv[:, :, 1]))
    texture = image_texture_profile(sample)

    target_light = 130.0 if scene == "portrait" else 136.0 if scene == "landscape" else 178.0 if scene == "product" else 108.0 if scene == "night" else 132.0
    target_sat = 62.0 if scene == "portrait" else 92.0 if scene == "landscape" else 48.0 if scene == "product" else 118.0 if scene == "night" else 70.0
    target_contrast = 92.0 if scene == "portrait" else 112.0 if scene == "landscape" else 96.0 if scene == "product" else 120.0
    if scene == "landscape" and mean_light > 165.0 and sat < 45.0:
        target_light = 176.0
        target_sat = 42.0
        target_contrast = 122.0

    exposure_score = 100.0 - abs(mean_light - target_light) * (0.62 if scene == "product" else 0.72)
    saturation_score = 100.0 - abs(sat - target_sat) * (0.42 if scene == "landscape" else 0.5)
    contrast_score = 100.0 - max(0.0, target_contrast - contrast) * 0.45 - max(0.0, contrast - target_contrast - 42.0) * 0.28
    smooth_score = 100.0 - max(0.0, texture["residual"] - 1.25) * 18.0 - max(0.0, texture["edgeRatio"] - 0.025) * 500.0
    score = exposure_score * 0.34 + contrast_score * 0.24 + saturation_score * 0.22 + smooth_score * 0.2
    if scene == "product":
        score = score * 0.84 + product_background_neutrality(sample) * 0.16
    if scene == "portrait":
        chroma = float(np.linalg.norm(np.mean(lab[:, :, 1:3], axis=(0, 1)) - np.array([128.0, 140.0], dtype=np.float32)))
        score -= max(0.0, chroma - 18.0) * 0.25
    return int(np.clip(score, 1, 99))


def quality_score_for_scene(bgr: np.ndarray, scene: str) -> int:
    texture = image_texture_profile(bgr)
    if texture["graphicScore"] >= 76.0:
        return graphic_quality_score(bgr, scene)
    if scene not in {"product", "portrait", "night"}:
        return quality_score(bgr)
    sample = resize_for_analysis(bgr, 1000)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_light = float(np.mean(gray))
    sat = float(np.mean(hsv[:, :, 1]))

    if scene == "product":
        exposure = 100 - abs(mean_light - 184) * 0.42
        saturation_score = 100 - max(0.0, sat - 92) * 0.65 - max(0.0, 28 - sat) * 0.6
        neutrality = product_background_neutrality(bgr)
        score = exposure * 0.28 + min(contrast, 145) * 0.27 + min(sharpness / 10, 100) * 0.15 + saturation_score * 0.12 + neutrality * 0.18
    elif scene == "portrait":
        exposure = 100 - abs(mean_light - 124) * 0.55
        saturation_score = 100 - max(0.0, sat - 92) * 0.7 - max(0.0, 30 - sat) * 0.8
        score = exposure * 0.34 + min(contrast, 128) * 0.27 + min(sharpness / 14, 100) * 0.12 + saturation_score * 0.27
    else:
        exposure = 100 - abs(mean_light - 108) * 0.5
        saturation_score = 100 - max(0.0, sat - 110) * 0.55 - max(0.0, 24 - sat) * 0.55
        score = exposure * 0.32 + min(contrast, 145) * 0.28 + min(sharpness / 12, 100) * 0.16 + saturation_score * 0.24
    return int(np.clip(score, 1, 99))


def naturalness_score(candidate: np.ndarray, original: np.ndarray, scene: str) -> float:
    metrics = weirdness_metrics(candidate, original, scene)
    if scene == "general":
        return float(
            100.0
            - metrics["penalty"] * 5.2
            - max(0.0, metrics["sharpnessShift"] - 36.0) * 0.45
            - max(0.0, metrics["lightShift"] - 7.0) * 0.95
            - metrics["coolShift"] * 2.1
            - metrics.get("shadowLift", 0.0) * 0.12
            - metrics.get("shadowTextureShift", 0.0) * 0.22
        )
    if scene == "landscape":
        return float(
            100.0
            - metrics["penalty"] * 4.8
            - max(0.0, metrics["sharpnessShift"] - 50.0) * 0.28
            - max(0.0, metrics["lightShift"] - 6.5) * 0.76
            - metrics["coolShift"] * 1.6
            - metrics.get("shadowLift", 0.0) * 0.05
            - metrics.get("shadowTextureShift", 0.0) * 0.14
        )
    return float(np.clip(100.0 - metrics["penalty"] * 5.0, 0.0, 100.0))


def aesthetic_score(candidate: np.ndarray, original: np.ndarray, scene: str) -> float:
    sample = resize_for_analysis(candidate, 900)
    original_sample = resize_for_analysis(original, 900)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB).astype(np.float32)
    original_lab = cv2.cvtColor(original_sample, cv2.COLOR_BGR2LAB).astype(np.float32)

    mean_l = float(np.mean(gray))
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    saturation = float(np.mean(hsv[:, :, 1]))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    original_sharpness = float(cv2.Laplacian(cv2.cvtColor(original_sample, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    ab_shift = float(np.linalg.norm(np.mean(lab[:, :, 1:3], axis=(0, 1)) - np.mean(original_lab[:, :, 1:3], axis=(0, 1))))
    clip_ratio = float(np.mean((gray < 3) | (gray > 252)))

    target_l = 132 if scene == "night" else 142 if scene == "product" else 136
    target_sat = 58 if scene == "portrait" else 76 if scene == "landscape" else 62
    exposure_score = 100 - abs(mean_l - target_l) * 0.75
    contrast_score = 100 - abs(min(contrast, 150) - 112) * 0.35
    saturation_score = 100 - abs(saturation - target_sat) * 0.55
    if scene in {"general", "landscape"}:
        target_sharpness = float(
            np.clip(
                max(original_sharpness * (1.08 if scene == "general" else 1.12), 54.0 if scene == "general" else 64.0),
                0.0,
                96.0 if scene == "general" else 118.0,
            )
        )
        sharp_score = 100.0 - max(0.0, target_sharpness - sharpness) * 0.48 - max(0.0, sharpness - target_sharpness) * (
            0.95 if scene == "general" else 0.72
        )
        sharp_score = float(np.clip(sharp_score, 0.0, 100.0))
    else:
        sharp_score = min(sharpness / 10, 100)
    penalty = clip_ratio * 260 + max(0, ab_shift - (9 if scene == "portrait" else 14)) * 4
    return exposure_score * 0.34 + contrast_score * 0.24 + saturation_score * 0.22 + sharp_score * 0.12 - penalty


def weirdness_metrics(candidate: np.ndarray, original: np.ndarray, scene: str) -> dict[str, float]:
    sample = resize_for_analysis(candidate, 900)
    original_sample = resize_for_analysis(original, 900)
    gray_u8 = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    original_gray_u8 = cv2.cvtColor(original_sample, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)
    original_gray = original_gray_u8.astype(np.float32)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).astype(np.float32)
    original_hsv = cv2.cvtColor(original_sample, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB).astype(np.float32)
    original_lab = cv2.cvtColor(original_sample, cv2.COLOR_BGR2LAB).astype(np.float32)
    sharpness = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
    original_sharpness = float(cv2.Laplacian(original_gray_u8, cv2.CV_64F).var())

    color_shift = float(np.linalg.norm(np.mean(lab[:, :, 1:3], axis=(0, 1)) - np.mean(original_lab[:, :, 1:3], axis=(0, 1))))
    saturation_shift = float(abs(np.mean(hsv[:, :, 1]) - np.mean(original_hsv[:, :, 1])))
    light_shift = float(abs(np.mean(gray) - np.mean(original_gray)))
    clip_ratio = float(np.mean((gray_u8 < 3) | (gray_u8 > 252)))
    original_clip_ratio = float(np.mean((original_gray_u8 < 3) | (original_gray_u8 > 252)))
    added_clip_ratio = max(0.0, clip_ratio - original_clip_ratio - 0.004)
    cool_shift = 0.0
    shadow_lift = 0.0
    shadow_texture_shift = 0.0
    if scene in {"general", "landscape"}:
        temp_allowance = 1.1 if scene == "general" else 1.5
        original_sat_mean = float(np.mean(original_hsv[:, :, 1]))
        if scene == "general" and original_sat_mean > 88:
            temp_allowance += 0.4
        elif scene == "landscape" and original_sat_mean > 92:
            temp_allowance += 0.9
        cool_shift = max(0.0, float(np.mean(original_lab[:, :, 2]) - np.mean(lab[:, :, 2])) - temp_allowance)
        shadow_threshold = min(116.0 if scene == "general" else 108.0, float(np.percentile(original_gray, 58)))
        shadow_mask = (original_gray < shadow_threshold).astype(np.float32)
        shadow_sat_mask = (original_hsv[:, :, 1] < (98 if scene == "general" else 88)).astype(np.float32)
        shadow_weight = shadow_mask * shadow_sat_mask
        shadow_coverage = float(np.mean(shadow_weight))
        if shadow_coverage > 0.02:
            weight_sum = float(np.sum(shadow_weight)) + 1e-6
            shadow_lift = max(0.0, float(np.sum((gray - original_gray) * shadow_weight) / weight_sum) - (12.0 if scene == "general" else 9.0))
            original_shadow_texture = float(
                np.sum(np.abs(original_gray - cv2.GaussianBlur(original_gray, (0, 0), 1.6)) * shadow_weight) / weight_sum
            )
            candidate_shadow_texture = float(np.sum(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.6)) * shadow_weight) / weight_sum)
            shadow_texture_shift = max(0.0, candidate_shadow_texture - original_shadow_texture - (4.0 if scene == "general" else 2.8))
    sharpness_shift = max(0.0, sharpness - original_sharpness)
    max_color_shift = 8 if scene == "portrait" else 15 if scene == "product" else 13
    max_sat_shift = 15 if scene == "portrait" else 34 if scene == "product" else 22 if scene == "night" else 24
    max_light_shift = 18 if scene == "night" else 26 if scene == "portrait" and float(np.mean(original_gray)) < 78 else 22
    max_sharpness_shift = 30 if scene == "general" else 42 if scene == "landscape" else 999
    penalty = (
        max(0.0, color_shift - max_color_shift) * 4.0
        + max(0.0, saturation_shift - max_sat_shift) * 1.4
        + max(0.0, light_shift - max_light_shift) * 1.2
        + cool_shift * (2.4 if scene == "general" else 1.7 if scene == "landscape" else 0.0)
        + shadow_lift * (0.12 if scene == "general" else 0.06 if scene == "landscape" else 0.0)
        + shadow_texture_shift * (0.28 if scene == "general" else 0.18 if scene == "landscape" else 0.0)
        + max(0.0, sharpness_shift - max_sharpness_shift) * (0.085 if scene == "general" else 0.065 if scene == "landscape" else 0.0)
        + added_clip_ratio * 420.0
    )
    return {
        "colorShift": round(color_shift, 2),
        "saturationShift": round(saturation_shift, 2),
        "lightShift": round(light_shift, 2),
        "coolShift": round(cool_shift, 2),
        "shadowLift": round(shadow_lift, 2),
        "shadowTextureShift": round(shadow_texture_shift, 2),
        "sharpnessShift": round(sharpness_shift, 2),
        "clipRatio": round(clip_ratio, 4),
        "addedClipRatio": round(added_clip_ratio, 4),
        "penalty": round(penalty, 2),
    }


def reference_similarity_score(candidate: np.ndarray, scene: str, profile_name: str | None) -> float:
    bank, _ = reference_snapshot()
    profile = next((item for item in bank if item.name == profile_name), None)
    if profile is None:
        profile = choose_reference_profile(candidate, scene)
    distance = profile_distance(candidate, profile)
    return float(np.clip(100 - distance * 18, 0, 100))


def adaptive_reference_intensity(bgr: np.ndarray, scene: str, profile: ReferenceProfile, requested: float) -> tuple[float, float, str]:
    fit = float(np.clip(100 - profile_distance(bgr, profile) * 18, 0, 100))
    effective = requested
    note = ""
    mean_light = float(np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
    mean_sat = float(np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]))
    if fit < 52:
        effective *= 0.2
        note = "参考差异较大，自动降权"
    elif fit < 65:
        effective *= 0.55
        note = "参考略有差异，柔和套用"

    if fit < 45:
        effective *= 0.2
        note = "参考匹配较低，自动降权"
    quality_now = quality_score_for_scene(bgr, scene)
    if scene == "portrait" and quality_now >= 86:
        effective = min(effective, 0.22)
        note = note or "人像原图质量较高，轻量参考防止过亮"
    elif scene == "portrait" and quality_now >= 82:
        effective = min(effective, 0.32)
        note = note or "人像质量较高，降低参考强度"
    elif scene in {"general", "product"} and quality_now >= 90:
        effective = min(effective, 0.28)
        note = note or "原图质量较高，轻量参考保持自然"
    elif scene in {"general", "product"} and quality_now >= 86:
        effective = min(effective, 0.38)
        note = note or "原图已有较好质感，参考强度已下调"
    elif scene == "landscape" and quality_now >= 94:
        effective = min(effective, 0.24)
        note = note or "高质量风景，轻量参考防止偏色"
    elif scene == "landscape" and quality_now >= 90:
        effective = min(effective, 0.34)
        note = note or "风景质量较高，降低参考强度"
    elif scene == "landscape" and quality_now >= 86:
        effective = min(effective, 0.28)
        note = note or "风景已有氛围，参考强度已下调"
    elif quality_now >= 90:
        effective = min(effective, 0.42)
        note = note or "原图质量较高，轻量参考"
    if scene == "general" and quality_now >= 82:
        effective = min(effective, 0.3)
        note = note or "生活场景优先保留原始氛围"
    if scene == "general" and mean_light < 122 and mean_sat < 96:
        effective = min(effective, 0.22)
        note = note or "室内氛围保留，自动弱化参考"
    if scene in {"general", "landscape"} and fit < 76:
        effective = min(effective, 0.24)
        note = note or "参考匹配一般，保持保守"
    if scene == "portrait":
        effective = min(effective, 0.72)
    elif scene == "night":
        effective = min(effective, 0.42 if quality_now < 82 else 0.26)
    return float(np.clip(effective, 0.0, 1.0)), round(fit, 1), note


def score_candidate(candidate: np.ndarray, original: np.ndarray, scene: str, details: dict[str, Any], before_score: int) -> tuple[float, dict[str, float]]:
    aesthetic = aesthetic_score(candidate, original, scene)
    quality_gain = quality_score_for_scene(candidate, scene) - before_score
    reference_score = reference_similarity_score(candidate, scene, details.get("profile"))
    weirdness = weirdness_metrics(candidate, original, scene)
    naturalness = naturalness_score(candidate, original, scene)
    product_neutrality = product_background_neutrality(candidate) if scene == "product" else 0.0
    subject_readability = portrait_subject_readability(candidate) if scene == "portrait" else 0.0
    final_score = aesthetic + quality_gain * 2.2 + reference_score * 0.2 - weirdness["penalty"] * 1.25
    if scene in {"general", "landscape"}:
        final_score += (naturalness - 84.0) * 0.58
    if scene == "product":
        final_score += product_neutrality * 0.18
    elif scene == "portrait":
        final_score += (subject_readability - 72.0) * 0.28
    if reference_score < 55:
        final_score -= (55 - reference_score) * 0.65
    return final_score, {
        **weirdness,
        "referenceScore": round(reference_score, 1),
        "productNeutrality": round(product_neutrality, 1),
        "subjectReadability": round(subject_readability, 1),
        "naturalnessScore": round(naturalness, 1),
        "candidateScore": round(final_score, 1),
    }


def tame_weird_output(original: np.ndarray, candidate: np.ndarray, scene: str) -> tuple[np.ndarray, dict[str, float]]:
    metrics = weirdness_metrics(candidate, original, scene)
    threshold = 3.0 if scene == "general" else 4.2 if scene == "landscape" else 6.0
    if metrics["penalty"] <= threshold:
        return candidate, metrics
    blend = float(
        np.clip(
            metrics["penalty"] / 40
            + max(0.0, metrics.get("sharpnessShift", 0.0) - (40.0 if scene == "general" else 56.0 if scene == "landscape" else 999.0)) / 420.0
            + metrics.get("coolShift", 0.0) / 18.0,
            0.18,
            0.48,
        )
    )
    corrected = cv2.addWeighted(original, blend, candidate, 1 - blend, 0)
    corrected = aesthetic_guardrails(original, corrected, scene)
    return corrected, weirdness_metrics(corrected, original, scene)


def tune_luminance_curve(bgr: np.ndarray, scene: str) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    if scene == "portrait":
        gamma, contrast, lift = 0.98, 1.05, 1.0
    elif scene == "landscape":
        gamma, contrast, lift = 0.99, 1.03, 0.0
    elif scene == "night":
        gamma, contrast, lift = 0.96, 1.02, -1.0
    else:
        gamma, contrast, lift = 0.98, 1.03, 0.5

    curved = np.power(np.clip(l_channel / 255.0, 0, 1), gamma) * 255.0
    mean_l = float(np.mean(curved))
    curved = (curved - mean_l) * contrast + mean_l + lift
    lab[:, :, 0] = np.clip(curved, 0, 255)
    tuned = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return aesthetic_guardrails(bgr, tuned, scene)


def finish_candidate(candidate: np.ndarray, original: np.ndarray, scene: str) -> tuple[np.ndarray, dict[str, float]]:
    polished = reduce_noise_and_sharpen(candidate, scene)
    polished = aesthetic_guardrails(original, polished, scene)
    polished = final_finish(polished, scene)
    polished = smooth_low_texture_background(polished, scene)
    polished = protect_smooth_gradients(original, polished, scene)
    if scene == "portrait":
        polished = normalize_portrait_skin_region(original, polished)
        polished = lift_backlit_portrait_subject(original, polished)
    polished = protect_shadow_readability(original, polished, scene)
    polished = preserve_scene_atmosphere(original, polished, scene)

    tuned = tune_luminance_curve(polished, scene)
    tuned = smooth_low_texture_background(tuned, scene)
    tuned = protect_smooth_gradients(original, tuned, scene)
    if scene == "portrait":
        tuned = normalize_portrait_skin_region(original, tuned)
        tuned = lift_backlit_portrait_subject(original, tuned)
    tuned = protect_shadow_readability(original, tuned, scene)
    tuned = preserve_scene_atmosphere(original, tuned, scene)
    original_score = quality_score_for_scene(original, scene)
    polished_score = quality_score_for_scene(polished, scene)
    tuned_score = quality_score_for_scene(tuned, scene)
    polished_penalty = weirdness_metrics(polished, original, scene)["penalty"]
    tuned_penalty = weirdness_metrics(tuned, original, scene)["penalty"]
    if tuned_score >= polished_score and tuned_penalty <= max(7.0, polished_penalty + 2.5):
        polished = tuned
    elif scene == "portrait" and tuned_score >= original_score + 14 and tuned_penalty <= 1.0:
        polished = tuned

    polished, final_metrics = tame_weird_output(original, polished, scene)
    return polished, final_metrics


def final_candidate_score(
    candidate: np.ndarray,
    original: np.ndarray,
    scene: str,
    details: dict[str, Any],
    before_score: int,
) -> tuple[float, dict[str, float]]:
    finished, final_metrics = finish_candidate(candidate, original, scene)
    after_score = quality_score_for_scene(finished, scene)
    aesthetic = aesthetic_score(finished, original, scene)
    reference_score = reference_similarity_score(finished, scene, details.get("profile"))
    weirdness = weirdness_metrics(finished, original, scene)
    naturalness = naturalness_score(finished, original, scene)
    product_neutrality = product_background_neutrality(finished) if scene == "product" else 0.0
    subject_readability = portrait_subject_readability(finished) if scene == "portrait" else 0.0
    quality_gain = after_score - before_score
    score = aesthetic + quality_gain * 2.35 + reference_score * 0.24 - weirdness["penalty"] * 1.35
    if scene in {"general", "landscape"}:
        score += (naturalness - 84.0) * 0.62
    if scene == "product":
        score += product_neutrality * 0.22
    elif scene == "portrait":
        score += (subject_readability - 72.0) * 0.35
    if reference_score < 55:
        score -= (55 - reference_score) * 0.75
    if quality_gain < 0:
        score += quality_gain * 2.0
    return score, {
        **final_metrics,
        **weirdness,
        "referenceScore": round(reference_score, 1),
        "productNeutrality": round(product_neutrality, 1),
        "subjectReadability": round(subject_readability, 1),
        "naturalnessScore": round(naturalness, 1),
        "candidateScore": round(score, 1),
        "afterQuality": after_score,
        "aestheticScore": round(aesthetic, 1),
    }


def choose_best_finished_candidate(
    candidates: list[tuple[float, np.ndarray, dict[str, Any], dict[str, float]]],
    before_score: int,
    scene: str,
) -> tuple[float, np.ndarray, dict[str, Any], dict[str, float]]:
    best = max(candidates, key=lambda item: item[0])
    if scene == "portrait":
        best_quality = int(best[3].get("afterQuality", before_score))
        best_penalty = float(best[3].get("penalty", 99.0))
        best_color_shift = float(best[3].get("colorShift", 99.0))
        best_reference = float(best[3].get("referenceScore", 0.0))
        portrait_viable: list[tuple[float, np.ndarray, dict[str, Any], dict[str, float]]] = []
        tied_portrait: list[tuple[float, np.ndarray, dict[str, Any], dict[str, float]]] = []
        for candidate in candidates:
            metrics = candidate[3]
            after_quality = int(metrics.get("afterQuality", before_score))
            penalty = float(metrics.get("penalty", 99.0))
            color_shift = float(metrics.get("colorShift", 99.0))
            saturation_shift = float(metrics.get("saturationShift", 99.0))
            reference_score = float(metrics.get("referenceScore", 0.0))
            if (
                penalty <= max(0.5, best_penalty + 0.5)
                and after_quality >= best_quality + 1
                and color_shift <= max(2.0, best_color_shift + 0.4)
                and reference_score >= best_reference - 32
            ):
                portrait_viable.append(candidate)
            elif (
                penalty <= max(0.5, best_penalty + 0.5)
                and after_quality >= best_quality
                and color_shift <= max(1.0, best_color_shift - 0.12)
                and saturation_shift <= float(best[3].get("saturationShift", 99.0)) + 2.0
                and reference_score >= best_reference - 32
            ):
                tied_portrait.append(candidate)
        if portrait_viable:
            return max(
                portrait_viable,
                key=lambda item: (
                    int(item[3].get("afterQuality", before_score)),
                    -float(item[3].get("colorShift", 99.0)),
                    -float(item[3].get("penalty", 99.0)),
                    item[0],
                ),
            )
        if tied_portrait:
            return max(
                tied_portrait,
                key=lambda item: (
                    -float(item[3].get("colorShift", 99.0)),
                    -float(item[3].get("saturationShift", 99.0)),
                    float(item[3].get("referenceScore", 0.0)),
                    item[0],
                ),
            )
        return best
    if scene == "night":
        best_quality = int(best[3].get("afterQuality", before_score))
        best_penalty = float(best[3].get("penalty", 99.0))
        stable_night = [
            candidate
            for candidate in candidates
            if int(candidate[3].get("afterQuality", before_score)) >= best_quality - 1
            and float(candidate[3].get("penalty", 99.0)) <= best_penalty + 0.8
        ]
        if stable_night:
            return max(
                stable_night,
                key=lambda item: (
                    -float(item[3].get("penalty", 99.0)),
                    -float(item[3].get("saturationShift", 99.0)),
                    -float(item[3].get("colorShift", 99.0)),
                    int(item[3].get("afterQuality", before_score)),
                    item[0],
                ),
            )
        return best
    if scene not in {"landscape", "general", "product"}:
        return best

    best_quality = int(best[3].get("afterQuality", before_score))
    best_reference = float(best[3].get("referenceScore", 0.0))
    best_penalty = float(best[3].get("penalty", 99.0))
    best_cool = float(best[3].get("coolShift", 99.0))
    best_sharpness = float(best[3].get("sharpnessShift", 999.0))
    viable: list[tuple[float, np.ndarray, dict[str, Any], dict[str, float]]] = []
    for candidate in candidates:
        metrics = candidate[3]
        after_quality = int(metrics.get("afterQuality", before_score))
        reference_score = float(metrics.get("referenceScore", 0.0))
        penalty = float(metrics.get("penalty", 99.0))
        cool_shift = float(metrics.get("coolShift", 0.0))
        sharpness_shift = float(metrics.get("sharpnessShift", 0.0))
        if after_quality >= max(before_score + 12, best_quality - 4) and penalty <= 3.0 and reference_score >= best_reference + 15:
            viable.append(candidate)
        elif scene in {"product", "general"} and after_quality >= best_quality - 2 and reference_score >= best_reference - 3 and penalty <= max(1.0, best_penalty - 2.5):
            viable.append(candidate)
        elif scene in {"general", "landscape"} and after_quality >= best_quality - 1 and penalty <= best_penalty + 0.5 and (
            cool_shift <= max(0.4, best_cool - 0.5) or sharpness_shift <= max(10.0, best_sharpness - 18.0)
        ):
            viable.append(candidate)

    if not viable:
        return best
    if scene in {"product", "general"}:
        return max(
            viable,
            key=lambda item: (
                -float(item[3].get("penalty", 99.0)),
                -float(item[3].get("coolShift", 99.0)),
                -float(item[3].get("sharpnessShift", 999.0)),
                int(item[3].get("afterQuality", before_score)),
                float(item[3].get("referenceScore", 0.0)),
                item[0],
            ),
        )
    return max(
        viable,
        key=lambda item: (
            int(item[3].get("afterQuality", before_score)),
            -float(item[3].get("coolShift", 99.0)),
            -float(item[3].get("sharpnessShift", 999.0)),
            float(item[3].get("referenceScore", 0.0)),
            item[0],
        ),
    )


def simple_graphic_exposure(bgr: np.ndarray, scene: str, intensity: float) -> tuple[np.ndarray, dict[str, Any]]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_light = float(np.mean(gray))
    shadow = float(np.percentile(gray, 8))
    highlight = float(np.percentile(gray, 94))
    target = 130.0 if scene == "portrait" else 136.0 if scene == "landscape" else 178.0 if scene == "product" else 108.0 if scene == "night" else 132.0
    max_delta = 48.0 if scene == "portrait" else 34.0
    delta = float(np.clip((target - mean_light) * 0.55 * intensity, -14.0, max_delta))
    contrast = 1.0 + float(np.clip((124.0 - (highlight - shadow)) / 190.0, -0.06, 0.12)) * intensity
    mean_l = float(np.mean(l_channel))
    lab[:, :, 0] = np.clip((l_channel - mean_l) * contrast + mean_l + delta, 0, 255)
    adjusted = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return adjusted, {
        "meanLight": round(mean_light, 1),
        "shadow": round(shadow, 1),
        "highlight": round(highlight, 1),
        "gamma": 1.0,
        "contrast": "图形安全微调",
    }


def simple_graphic_color(bgr: np.ndarray, scene: str, intensity: float) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    target_sat = 58.0 if scene == "portrait" else 92.0 if scene == "landscape" else 48.0 if scene == "product" else 70.0
    max_positive = 0.42 if scene == "landscape" and mean_sat < 50 else 0.22 if mean_sat < 58 else 0.14
    sat_gain = 1.0 + float(np.clip((target_sat / max(mean_sat, 1.0) - 1.0) * 0.45 * intensity, -0.16, max_positive))
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_gain, 0, 235)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def neutralize_graphic_product_background(original: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV).astype(np.float32)
    light = lab[:, :, 0]
    sat = hsv[:, :, 1]
    background = (light > np.percentile(light, 35)) & (sat < 165)
    if float(np.mean(background)) < 0.08:
        return candidate

    a_delta = float(np.mean(lab[:, :, 1][background]) - 128.0)
    b_delta = float(np.mean(lab[:, :, 2][background]) - 128.0)
    mask = cv2.GaussianBlur(background.astype(np.float32), (0, 0), 2.4)[:, :, None] * 0.8
    adjusted = lab.copy()
    adjusted[:, :, 1] = adjusted[:, :, 1] - a_delta * mask[:, :, 0]
    adjusted[:, :, 2] = adjusted[:, :, 2] - b_delta * mask[:, :, 0]
    neutral = cv2.cvtColor(np.clip(adjusted, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    before_neutrality = product_background_neutrality(candidate)
    after_neutrality = product_background_neutrality(neutral)
    before_metrics = weirdness_metrics(candidate, original, "product")
    after_metrics = weirdness_metrics(neutral, original, "product")
    if after_neutrality < before_neutrality + 12 and after_metrics["penalty"] > before_metrics["penalty"] + 2.0:
        return candidate
    return neutral


def graphic_aesthetic_guardrails(original: np.ndarray, candidate: np.ndarray, scene: str, before_score: int) -> np.ndarray:
    original_mean = float(np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)))
    if scene == "portrait" and before_score < 76 and original_mean < 72:
        guarded = protect_skin_tone(original, candidate, strength=0.38)
        guarded = normalize_portrait_skin_region(original, guarded)
        guarded = protect_bright_neutral_regions(original, guarded, scene)
    else:
        guarded = aesthetic_guardrails(original, candidate, scene)
    guarded_mean = float(np.mean(cv2.cvtColor(guarded, cv2.COLOR_BGR2GRAY)))
    candidate_mean = float(np.mean(cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)))
    target = 112.0 if scene == "portrait" and original_mean < 72 else 108.0 if scene == "portrait" else 132.0 if scene == "landscape" else 160.0 if scene == "product" else 112.0
    if before_score < 82 and candidate_mean > guarded_mean + 2.0 and guarded_mean < target - 8.0:
        needed = np.clip((target - guarded_mean) / max(candidate_mean - guarded_mean, 1.0), 0.0, 0.65)
        if needed > 0.05:
            lifted = cv2.addWeighted(guarded, 1 - float(needed), candidate, float(needed), 0)
            lifted = protect_extremes(original, lifted)
            metrics = graphic_candidate_score(lifted, original, scene, before_score)[1]
            max_light = 46.0 if original_mean < 80 else 30.0
            if metrics["penalty"] <= 1.5 and metrics["lightShift"] <= max_light and metrics["colorShift"] <= (7.0 if scene == "portrait" else 10.0):
                guarded = lifted

    if scene == "landscape":
        original_sat = float(np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2HSV)[:, :, 1]))
        guarded_sat = float(np.mean(cv2.cvtColor(guarded, cv2.COLOR_BGR2HSV)[:, :, 1]))
        candidate_sat = float(np.mean(cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV)[:, :, 1]))
        if original_sat < 45.0 and candidate_sat > guarded_sat + 3.0 and guarded_sat < 52.0:
            blend = float(np.clip((52.0 - guarded_sat) / max(candidate_sat - guarded_sat, 1.0), 0.0, 0.58))
            richer = cv2.addWeighted(guarded, 1 - blend, candidate, blend, 0)
            metrics = graphic_candidate_score(richer, original, scene, before_score)[1]
            if metrics["penalty"] <= 1.5 and metrics["colorShift"] <= 9.5:
                guarded = richer
    return guarded


def graphic_candidate_score(candidate: np.ndarray, original: np.ndarray, scene: str, before_score: int) -> tuple[float, dict[str, float]]:
    after_quality = quality_score_for_scene(candidate, scene)
    weirdness = weirdness_metrics(candidate, original, scene)
    graphic_max_color = 15.0 if scene == "product" else 6.5 if scene == "portrait" else 8.0
    if scene == "portrait" and before_score < 72:
        graphic_max_sat = 40.0
        graphic_max_light = 48.0
    elif scene == "portrait" and before_score < 76:
        graphic_max_sat = 30.0
        graphic_max_light = 36.0
    else:
        graphic_max_sat = 42.0 if scene == "product" else 24.0 if before_score < 78 else 16.0
        graphic_max_light = 32.0 if before_score < 78 else 22.0
    graphic_penalty = (
        max(0.0, float(weirdness["colorShift"]) - graphic_max_color) * 3.2
        + max(0.0, float(weirdness["saturationShift"]) - graphic_max_sat) * 0.9
        + max(0.0, float(weirdness["lightShift"]) - graphic_max_light) * 0.9
        + float(weirdness["addedClipRatio"]) * 420.0
    )
    weirdness = {**weirdness, "penalty": round(graphic_penalty, 2)}
    texture = image_texture_profile(candidate)
    reference_score = reference_similarity_score(candidate, scene, None)
    product_neutrality = product_background_neutrality(candidate) if scene == "product" else 0.0
    quality_gain = after_quality - before_score
    allowed_light_shift = 44.0 if scene == "portrait" and before_score < 72 else 36.0 if scene == "portrait" and before_score < 76 else 30.0 if before_score < 76 else 18.0
    allowed_saturation_shift = 38.0 if scene == "portrait" and before_score < 72 else 28.0 if scene == "portrait" and before_score < 76 else 20.0 if before_score < 76 else 12.0
    color_penalty = 0.35 if scene == "product" else 1.1
    score = (
        after_quality * 1.3
        + quality_gain * 1.9
        + reference_score * 0.08
        + texture["graphicScore"] * 0.35
        + product_neutrality * (0.38 if scene == "product" else 0.0)
        - weirdness["penalty"] * 3.0
        - weirdness["colorShift"] * color_penalty
        - max(0.0, weirdness["lightShift"] - allowed_light_shift) * 1.1
        - max(0.0, weirdness["saturationShift"] - allowed_saturation_shift) * 0.8
    )
    return score, {
        **weirdness,
        "referenceScore": round(reference_score, 1),
        "productNeutrality": round(product_neutrality, 1),
        "subjectReadability": round(portrait_subject_readability(candidate), 1) if scene == "portrait" else 0.0,
        "candidateScore": round(score, 1),
        "afterQuality": after_quality,
        "aestheticScore": round(aesthetic_score(candidate, original, scene), 1),
        "graphicScore": texture["graphicScore"],
    }


def polish_graphic_image(bgr: np.ndarray, scene_info: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    scene = scene_info["scene"]
    original_h, original_w = bgr.shape[:2]
    before_score = quality_score_for_scene(bgr, scene)
    profile = choose_reference_profile(bgr, scene)
    reference_fit = round(float(np.clip(100 - profile_distance(bgr, profile) * 18, 0, 100)), 1)
    reference_note = f"自动参考：{profile.name}（图形安全，仅用于审美约束）"

    candidate_rows: list[tuple[float, np.ndarray, dict[str, Any], dict[str, float]]] = []
    base_light = {
        "meanLight": round(float(np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))), 1),
        "shadow": round(float(np.percentile(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 8)), 1),
        "highlight": round(float(np.percentile(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 94)), 1),
        "gamma": 1.0,
        "contrast": "保持图形原貌",
    }
    for label, image, light_stats, cast in [("ORIGINAL", bgr, base_light, 0.0)]:
        score, metrics = graphic_candidate_score(image, bgr, scene, before_score)
        details = {
            "cast": cast,
            "light": light_stats,
            "reference": reference_note,
            "profile": profile.name,
            "referenceFit": reference_fit,
            "effectiveReferenceIntensity": 0.0,
            "whiteBalanceMode": "保留图形色彩",
            "whiteBalanceKey": "graphic",
            "variant": label,
        }
        candidate_rows.append((score, image, details, metrics))

    for balance_mode in ("none", "neutral"):
        if balance_mode == "neutral":
            balanced_raw, cast = neutral_white_balance(bgr, scene)
            balanced = cv2.addWeighted(bgr, 0.58, balanced_raw, 0.42, 0)
            white_balance_note = "图形安全中性校正"
        else:
            balanced, cast = bgr, 0.0
            white_balance_note = "保留图形色彩"
        for exposure_intensity, color_intensity in ((0.45, 0.45), (0.7, 0.62), (0.9, 0.74), (1.12, 0.86), (1.32, 0.95)):
            exposed, light_stats = simple_graphic_exposure(balanced, scene, exposure_intensity)
            colored = simple_graphic_color(exposed, scene, color_intensity)
            guarded = graphic_aesthetic_guardrails(bgr, colored, scene, before_score)
            if scene == "product":
                guarded = neutralize_graphic_product_background(bgr, guarded)
            guarded = cv2.bilateralFilter(guarded, 5, 18, 18)
            guarded = protect_smooth_gradients(bgr, guarded, scene)
            score, metrics = graphic_candidate_score(guarded, bgr, scene, before_score)
            details = {
                "cast": cast,
                "light": light_stats,
                "reference": reference_note,
                "profile": profile.name,
                "referenceFit": reference_fit,
                "effectiveReferenceIntensity": 0.0,
                "whiteBalanceMode": white_balance_note,
                "whiteBalanceKey": balance_mode,
                "variant": f"G{exposure_intensity:.2f}/C{color_intensity:.2f}/{balance_mode[:1].upper()}",
            }
            candidate_rows.append((score, guarded, details, metrics))

    best_score, polished, best_details, final_metrics = max(
        candidate_rows,
        key=lambda item: (
            item[0],
            -float(item[3].get("penalty", 99.0)),
            -float(item[3].get("colorShift", 99.0)),
        ),
    )
    polished = protect_smooth_gradients(bgr, polished, scene)
    after_score = quality_score_for_scene(polished, scene)
    reference_score = reference_similarity_score(polished, scene, best_details.get("profile"))
    final_metrics = {
        **final_metrics,
        "referenceScore": round(reference_score, 1),
        "afterQuality": after_score,
    }

    adjustments = {
        "scene": scene,
        "whiteBalance": f"{best_details['whiteBalanceMode']}，{'已校正' if best_details['cast'] > 0.04 else '轻微优化'}",
        "light": best_details["light"],
        "color": "图形安全自然微调",
        "reference": best_details["reference"],
        "guardrails": "已识别为平涂/插画，禁止套用照片纹理和过强参考色",
        "tone": "保留原有渐变与色块边界",
        "noise": "低纹理保护",
        "sharpness": "未额外锐化",
        "quality": f"{before_score} -> {after_score}",
        "aestheticScore": round(best_score, 1),
        "referenceScore": final_metrics.get("referenceScore"),
        "referenceFit": best_details.get("referenceFit"),
        "referenceIntensity": best_details.get("effectiveReferenceIntensity"),
        "highQualityProtected": False,
        "weirdness": final_metrics,
        "variant": best_details["variant"],
        "size": f"{original_w} x {original_h}",
        "sceneStats": scene_info,
        "input": "GRAPHIC",
    }
    return polished, adjustments


def retouch_candidate(
    bgr: np.ndarray,
    scene: str,
    exposure_intensity: float,
    color_intensity: float,
    reference_intensity: float,
    tone_intensity: float = 1.0,
    white_balance_mode: str = "classic",
) -> tuple[np.ndarray, dict[str, Any]]:
    if white_balance_mode == "neutral":
        balanced, cast = neutral_white_balance(bgr, scene)
        white_balance_note = "中性参考"
        white_balance_key = "neutral"
    elif white_balance_mode == "strong-neutral":
        balanced, cast = strong_neutral_white_balance(bgr, scene)
        white_balance_note = "强中性参考"
        white_balance_key = "strong-neutral"
    elif white_balance_mode == "none":
        balanced, cast = bgr, 0.0
        white_balance_note = "保留现场色"
        white_balance_key = "none"
    else:
        balanced, cast = gray_world_white_balance(bgr)
        white_balance_note = "经典灰点"
        white_balance_key = "classic"
    exposed, light_stats = auto_exposure_lab(balanced, exposure_intensity, scene)
    if tone_intensity > 0:
        toned = recover_tone_detail(exposed, scene, exposure_intensity * tone_intensity)
    else:
        toned = exposed
    colored = enhance_color(toned, scene, color_intensity)
    colored = vibrance_color(colored, scene, color_intensity * (0.72 if scene == "portrait" else 1.0))
    if scene == "product":
        colored = neutralize_product_background(bgr, colored)
    profile = choose_reference_profile(colored, scene)
    effective_reference, reference_fit, reference_mode = adaptive_reference_intensity(colored, scene, profile, reference_intensity)
    if effective_reference <= 0.08:
        referenced = aesthetic_guardrails(bgr, colored, scene)
        reference_note = f"自动参考：{profile.name}（保守）"
    else:
        referenced, reference_note = apply_reference_style(colored, profile, scene, effective_reference)
        if reference_mode:
            reference_note = f"{reference_note}（{reference_mode}）"
    polished = aesthetic_guardrails(bgr, referenced, scene)
    if scene == "product":
        polished = neutralize_product_background(bgr, polished)
    return polished, {
        "cast": cast,
        "light": light_stats,
        "reference": reference_note,
        "profile": profile.name,
        "referenceFit": reference_fit,
        "effectiveReferenceIntensity": round(effective_reference, 2),
        "whiteBalanceMode": white_balance_note,
        "whiteBalanceKey": white_balance_key,
        "toneIntensity": tone_intensity,
        "variant": f"E{exposure_intensity:.2f}/C{color_intensity:.2f}/R{reference_intensity:.2f}/T{tone_intensity:.2f}/{white_balance_key[:1].upper()}",
    }


def polish_image(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    original_h, original_w = bgr.shape[:2]
    scene_info = classify_scene(bgr)
    scene = scene_info["scene"]
    if scene_info.get("isGraphic"):
        return polish_graphic_image(bgr, scene_info)
    before_score = quality_score_for_scene(bgr, scene)
    variants = [
        (0.78, 0.72, 0.0, 0.7),
        (0.86, 0.82, 0.7, 0.8),
        (1.0, 0.95, 0.85, 0.9),
        (1.12, 1.05, 1.0, 1.0),
    ]
    if scene == "portrait":
        variants = [
            (0.88, 0.76, 0.55, 0.0),
            (1.0, 0.84, 0.68, 0.0),
            (1.06, 0.9, 0.72, 0.0),
            (0.82, 0.68, 0.0, 0.28),
            (0.94, 0.76, 0.42, 0.22),
            (1.12, 0.98, 0.55, 0.0),
            (1.18, 0.94, 0.0, 0.0),
        ]
        if before_score >= 86:
            variants = [
                (0.82, 0.72, 0.0, 0.0),
                (0.9, 0.76, 0.22, 0.0),
                (0.96, 0.82, 0.32, 0.0),
                (0.9, 0.72, 0.0, 0.14),
            ]
        elif before_score >= 82:
            variants = [
                (0.86, 0.74, 0.0, 0.0),
                (0.94, 0.8, 0.32, 0.0),
                (1.0, 0.86, 0.42, 0.0),
                (0.92, 0.74, 0.22, 0.18),
            ]
    elif scene == "landscape":
        variants = [(0.82, 0.84, 0.0, 0.78), (0.9, 0.92, 0.54, 0.86), (1.0, 0.98, 0.7, 0.94), (1.08, 1.02, 0.82, 1.0)]
        if before_score >= 94:
            variants = [(0.8, 0.84, 0.0, 0.68), (0.88, 0.9, 0.32, 0.76), (0.96, 0.96, 0.46, 0.84)]
        elif before_score >= 90:
            variants = [(0.82, 0.86, 0.0, 0.74), (0.9, 0.92, 0.38, 0.82), (0.98, 0.98, 0.52, 0.9)]
    elif scene == "general":
        variants = [(0.78, 0.76, 0.0, 0.62), (0.86, 0.84, 0.24, 0.72), (0.94, 0.9, 0.34, 0.82)]
        if before_score >= 88:
            variants = [(0.76, 0.74, 0.0, 0.56), (0.84, 0.82, 0.18, 0.66), (0.9, 0.88, 0.26, 0.76)]
    elif scene == "night":
        variants = [(0.76, 0.66, 0.0, 0.35), (0.84, 0.72, 0.32, 0.45), (0.92, 0.8, 0.42, 0.52)]
        if before_score >= 82:
            variants = [(0.72, 0.62, 0.0, 0.28), (0.8, 0.68, 0.18, 0.36), (0.88, 0.74, 0.26, 0.44)]

    white_balance_modes = ["classic", "neutral"]
    if scene in {"portrait", "night"}:
        white_balance_modes.append("none")
    if scene in {"product", "general"}:
        white_balance_modes.append("strong-neutral")

    rough_source = resize_for_analysis(bgr, 900)
    rough_before_score = quality_score_for_scene(rough_source, scene)
    rough_candidates: list[tuple[float, tuple[float, float, float, float, str], dict[str, Any], dict[str, float], bool]] = []
    for exposure_intensity, color_intensity, reference_intensity, tone_intensity in variants:
        for white_balance_mode in white_balance_modes:
            candidate, details = retouch_candidate(
                rough_source,
                scene,
                exposure_intensity,
                color_intensity,
                reference_intensity,
                tone_intensity,
                white_balance_mode,
            )
            rough_score, rough_metrics = score_candidate(candidate, rough_source, scene, details, rough_before_score)
            keep_for_portrait_tone = False
            params = (exposure_intensity, color_intensity, reference_intensity, tone_intensity, white_balance_mode)
            rough_candidates.append((rough_score, params, details, rough_metrics, keep_for_portrait_tone))

    candidates: list[tuple[float, np.ndarray, dict[str, Any], dict[str, float]]] = []
    finalist_count = 3 if scene == "portrait" else 2
    finalists: list[tuple[float, tuple[float, float, float, float, str], dict[str, Any], dict[str, float], bool]] = []
    for rough_candidate in sorted(rough_candidates, key=lambda item: item[0], reverse=True):
        if len(finalists) < finalist_count or rough_candidate[4]:
            finalists.append(rough_candidate)
    if scene == "portrait":
        portrait_tone_options = [item for item in rough_candidates if item[1][3] > 0 and item[1][4] == "classic"]
        if portrait_tone_options:
            best_tone = max(portrait_tone_options, key=lambda item: item[0])
            if not any(item[2]["variant"] == best_tone[2]["variant"] for item in finalists):
                finalists.append(best_tone)
        stable_portrait_options = [
            item
            for item in rough_candidates
            if item[3].get("penalty", 99.0) <= 1.0 and item[1][4] in {"classic", "neutral"} and item[1][3] <= 0.22
        ]
        if before_score < 84:
            forced_portrait_variants = {
                "E1.18/C0.94/R0.00/T0.00/N",
                "E1.18/C0.94/R0.00/T0.00/C",
                "E1.12/C0.98/R0.55/T0.00/C",
                "E1.12/C0.98/R0.55/T0.00/N",
            }
            for forced_option in rough_candidates:
                if forced_option[2]["variant"] in forced_portrait_variants:
                    if not any(item[2]["variant"] == forced_option[2]["variant"] for item in finalists):
                        finalists.append(forced_option)
        for portrait_option in sorted(
            stable_portrait_options,
            key=lambda item: (item[3].get("candidateScore", item[0]), item[3].get("referenceScore", 0.0)),
            reverse=True,
        )[:2]:
            if not any(item[2]["variant"] == portrait_option[2]["variant"] for item in finalists):
                finalists.append(portrait_option)
    neutral_options = [item for item in rough_candidates if item[2].get("whiteBalanceKey") == "neutral"]
    if neutral_options:
        best_neutral = max(neutral_options, key=lambda item: item[0])
        if not any(item[2]["variant"] == best_neutral[2]["variant"] for item in finalists):
            finalists.append(best_neutral)
    if scene in {"product", "general"}:
        stable_options = [item for item in rough_candidates if item[3].get("penalty", 99.0) <= 1.5]
        if stable_options:
            best_stable = max(stable_options, key=lambda item: (item[0], item[3].get("referenceScore", 0.0)))
            if not any(item[2]["variant"] == best_stable[2]["variant"] for item in finalists):
                finalists.append(best_stable)
    for _, params, _, _, _ in finalists:
        exposure_intensity, color_intensity, reference_intensity, tone_intensity, white_balance_mode = params
        candidate, details = retouch_candidate(
            bgr,
            scene,
            exposure_intensity,
            color_intensity,
            reference_intensity,
            tone_intensity,
            white_balance_mode,
        )
        finished, metrics = finish_candidate(candidate, bgr, scene)
        after_quality = quality_score_for_scene(finished, scene)
        aesthetic = aesthetic_score(finished, bgr, scene)
        reference_score = reference_similarity_score(finished, scene, details.get("profile"))
        weirdness = weirdness_metrics(finished, bgr, scene)
        product_neutrality = product_background_neutrality(finished) if scene == "product" else 0.0
        subject_readability = portrait_subject_readability(finished) if scene == "portrait" else 0.0
        quality_gain = after_quality - before_score
        score = aesthetic + quality_gain * 2.35 + reference_score * 0.24 - weirdness["penalty"] * 1.35
        if scene == "product":
            score += product_neutrality * 0.22
        elif scene == "portrait":
            score += (subject_readability - 72.0) * 0.35
        if reference_score < 55:
            score -= (55 - reference_score) * 0.75
        if quality_gain < 0:
            score += quality_gain * 2.0
        metrics = {
            **metrics,
            **weirdness,
            "referenceScore": round(reference_score, 1),
            "productNeutrality": round(product_neutrality, 1),
            "subjectReadability": round(subject_readability, 1),
            "candidateScore": round(score, 1),
            "afterQuality": after_quality,
            "aestheticScore": round(aesthetic, 1),
        }
        candidates.append((score, finished, details, metrics))

    best_score, polished, best_details, best_metrics = choose_best_finished_candidate(candidates, before_score, scene)
    final_metrics = best_metrics
    after_score = quality_score_for_scene(polished, scene)

    if after_score < before_score - 4:
        polished = cv2.addWeighted(bgr, 0.28, polished, 0.72, 0)
        polished, final_metrics = tame_weird_output(bgr, polished, scene)
        after_score = quality_score_for_scene(polished, scene)

    protected_polished, high_quality_protected = protect_high_quality_original(bgr, polished, scene, before_score)
    if high_quality_protected:
        polished = protected_polished
        polished, final_metrics = tame_weird_output(bgr, polished, scene)
        after_score = quality_score_for_scene(polished, scene)
        reference_score = reference_similarity_score(polished, scene, best_details.get("profile"))
        final_metrics = {
            **final_metrics,
            "referenceScore": round(reference_score, 1),
            "afterQuality": after_score,
        }

    adjustments = {
        "scene": scene,
        "whiteBalance": f"{best_details['whiteBalanceMode']}，{'已校正' if best_details['cast'] > 0.04 else '轻微优化'}",
        "light": best_details["light"],
        "color": "自然增强",
        "reference": best_details["reference"],
        "guardrails": "已限制偏色/过饱和/过曝，并保护肤色与自然色",
        "tone": "已恢复暗部/高光层次",
        "noise": "智能降噪",
        "sharpness": "亮度细节锐化",
        "quality": f"{before_score} -> {after_score}",
        "aestheticScore": round(best_score, 1),
        "referenceScore": final_metrics.get("referenceScore", best_metrics.get("referenceScore")),
        "referenceFit": best_details.get("referenceFit"),
        "referenceIntensity": best_details.get("effectiveReferenceIntensity"),
        "highQualityProtected": high_quality_protected,
        "weirdness": final_metrics,
        "variant": best_details["variant"],
        "size": f"{original_w} x {original_h}",
        "sceneStats": scene_info,
    }
    return polished, adjustments


def save_outputs(job: Job, bgr: np.ndarray) -> None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.save(job.output_path, format="JPEG", quality=96, optimize=True, progressive=True, subsampling=0)

    preview = image.copy()
    preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    preview.save(job.preview_path, format="JPEG", quality=94, optimize=True, progressive=True, subsampling=0)


def save_original_preview(job: Job, bgr: np.ndarray) -> None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(rgb)
    preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    preview.save(job.original_preview_path, format="JPEG", quality=92, optimize=True, progressive=True, subsampling=0)


def process_job(job_id: str) -> None:
    job = jobs[job_id]
    try:
        if not job.upload_path.exists():
            set_job(job, status="failed", progress=100, message="处理失败", error="原始文件不存在，可能已被清空。")
            log_event("error", "原始文件不存在，跳过处理", jobId=job.id, file=job.original_name)
            return
        log_event("job", "开始处理", jobId=job.id, file=job.original_name)
        set_job(job, status="processing", progress=8, message="读取图片")
        bgr, metadata = load_image(job.upload_path)
        bgr, resize_note = resize_for_processing(bgr)
        if resize_note:
            log_event("job", "超大图片已降采样", jobId=job.id, file=job.original_name, note=resize_note)
        height, width = bgr.shape[:2]
        save_original_preview(job, bgr)
        set_job(job, progress=22, message="分析画面和选择参考", width=width, height=height)

        set_job(job, progress=38, message="生成多个修图候选")
        polished, adjustments = polish_image(bgr)
        adjustments["input"] = metadata["source"].upper()
        if metadata.get("fallback"):
            adjustments["rawFallback"] = metadata["fallback"]
        if resize_note:
            adjustments["sizeGuard"] = resize_note
        set_job(job, progress=82, message="导出高质量成品", adjustments=adjustments)

        save_outputs(job, polished)
        with jobs_lock:
            if job.id not in jobs or not job.upload_path.exists():
                log_event("job", "任务已被清空，丢弃输出", jobId=job.id, file=job.original_name)
                return
        set_job(
            job,
            status="done",
            progress=100,
            message="已完成",
            output_size=job.output_path.stat().st_size,
        )
        log_event("job", "处理完成", jobId=job.id, file=job.original_name, output=job.output_path.name)
    except Exception as exc:
        set_job(job, status="failed", progress=100, message="处理失败", error=str(exc))
        log_event("error", "处理失败", jobId=job.id, file=job.original_name, error=str(exc))


def enqueue_file(filename: str, content: bytes) -> Job:
    ensure_dirs()
    job_id = uuid.uuid4().hex[:12]
    original_name = safe_filename(filename)
    source_name = f"{job_id}_{original_name}"
    upload_path = UPLOAD_DIR / source_name
    original_preview_path = PREVIEW_DIR / f"{job_id}_original.jpg"
    output_path = OUTPUT_DIR / f"{Path(original_name).stem}_{job_id}_retouched.jpg"
    preview_path = PREVIEW_DIR / f"{job_id}.jpg"

    upload_path.write_bytes(content)
    job = Job(
        id=job_id,
        original_name=original_name,
        upload_path=upload_path,
        original_preview_path=original_preview_path,
        output_path=output_path,
        preview_path=preview_path,
        input_size=len(content),
    )
    with jobs_lock:
        jobs[job_id] = job
        save_jobs_locked()

    job_queue.put(job_id)
    log_event("job", "已加入队列", jobId=job.id, file=job.original_name, size=len(content))
    return job


def requeue_job(job_id: str) -> Job | None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return None
        job.status = "queued"
        job.progress = 0
        job.message = "等待重新处理"
        job.error = None
        job.adjustments = {}
        job.output_size = 0
        job.updated_at = now_ms()
        for path in (job.output_path, job.preview_path, job.original_preview_path):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        save_jobs_locked()
    job_queue.put(job_id)
    log_event("job", "已重新加入队列", jobId=job.id, file=job.original_name)
    return job


def inspect_raw_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": path.name,
        "size": path.stat().st_size if path.exists() else 0,
        "exists": path.exists(),
    }
    if not path.exists():
        return result
    data = path.read_bytes()[:16]
    result["header"] = data.hex(" ")
    result["tiffLike"] = data.startswith(b"II*\x00") or data.startswith(b"MM\x00*")
    if path.suffix.lower() in RAW_EXTENSIONS and rawpy is not None:
        try:
            with rawpy.imread(str(path)) as raw:
                result["rawOpen"] = True
                result["sizes"] = {
                    "width": raw.sizes.width,
                    "height": raw.sizes.height,
                    "rawWidth": raw.sizes.raw_width,
                    "rawHeight": raw.sizes.raw_height,
                }
        except Exception as exc:
            result["rawOpen"] = False
            result["rawError"] = str(exc)
        try:
            with rawpy.imread(str(path)) as raw:
                raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=False,
                    output_bps=8,
                    gamma=(2.222, 4.5),
                    user_flip=-1,
                    half_size=False,
                )
                result["fullDecode"] = True
        except Exception as exc:
            result["fullDecode"] = False
            result["fullDecodeError"] = str(exc)
        try:
            with rawpy.imread(str(path)) as raw:
                raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=False,
                    output_bps=8,
                    gamma=(2.222, 4.5),
                    user_flip=-1,
                    half_size=True,
                )
                result["halfDecode"] = True
        except Exception as exc:
            result["halfDecode"] = False
            result["halfDecodeError"] = str(exc)
        try:
            with rawpy.imread(str(path)) as raw:
                thumb = raw.extract_thumb()
                result["thumb"] = {
                    "format": str(getattr(thumb, "format", "")),
                    "bytes": len(getattr(thumb, "data", b"")),
                }
        except Exception as exc:
            result["thumbError"] = str(exc)
        try:
            embedded = extract_embedded_jpeg_from_raw(path)
            result["embeddedJpeg"] = {"width": int(embedded.shape[1]), "height": int(embedded.shape[0])}
        except Exception as exc:
            result["embeddedJpegError"] = str(exc)
    return result


def worker_loop() -> None:
    while True:
        try:
            job_id = job_queue.get(timeout=1)
        except Empty:
            continue
        try:
            if job_id in jobs:
                process_job(job_id)
        finally:
            job_queue.task_done()


class MultipartReader:
    def __init__(self, body: bytes, content_type: str):
        marker = "boundary="
        if marker not in content_type:
            raise ValueError("缺少 multipart boundary。")
        self.boundary = ("--" + content_type.split(marker, 1)[1].split(";", 1)[0].strip('"')).encode()
        self.body = body

    def files(self) -> list[tuple[str, bytes]]:
        items: list[tuple[str, bytes]] = []
        for part in self.body.split(self.boundary):
            part = part.strip(b"\r\n")
            if not part or part == b"--" or b"\r\n\r\n" not in part:
                continue
            header_blob, payload = part.split(b"\r\n\r\n", 1)
            headers = header_blob.decode("utf-8", "ignore")
            if "filename=" not in headers:
                continue
            filename = headers.split("filename=", 1)[1].split("\r\n", 1)[0].strip().strip('"')
            if payload.endswith(b"\r\n--"):
                payload = payload[:-4]
            elif payload.endswith(b"\r\n"):
                payload = payload[:-2]
            items.append((filename, payload))
        return items


class AppHandler(BaseHTTPRequestHandler):
    server_version = "AutoRetouch/1.2"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str, download_name: str | None = None) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/static/app.js":
            self.send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
            return
        if path == "/static/styles.css":
            self.send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
            return
        if path == "/api/jobs":
            bank, status = reference_snapshot()
            calibration = calibration_snapshot()
            with jobs_lock:
                payload = [job.public() for job in sorted(jobs.values(), key=lambda item: item.created_at, reverse=True)]
            self.send_json(
                {
                    "jobs": payload,
                    "rawAvailable": rawpy is not None,
                    "reference": {
                        "status": status,
                        "count": len(bank),
                        "onlineCount": sum(1 for profile in bank if profile.online),
                        "profiles": [profile.public() for profile in bank[:8]],
                    },
                    "calibration": calibration,
                    "events": recent_events(20),
                }
            )
            return
        if path.startswith("/api/calibration-source/"):
            filename = unquote(path.rsplit("/", 1)[-1])
            sample_path = REFERENCE_DIR / filename
            if not sample_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_file(sample_path, "image/jpeg")
            return
        if path.startswith("/api/calibration-output/"):
            filename = unquote(path.rsplit("/", 1)[-1])
            sample_path = REFERENCE_OUTPUT_DIR / filename
            if not sample_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_file(sample_path, "image/jpeg")
            return
        if path.startswith("/api/reference-preview/"):
            name = path.rsplit("/", 1)[-1]
            bank, _ = reference_snapshot()
            profile = next((item for item in bank if item.name == name), None)
            if not profile or not profile.preview_path:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_file(profile.preview_path, "image/jpeg")
            return
        if path.startswith("/api/original-preview/"):
            job_id = path.rsplit("/", 1)[-1]
            job = jobs.get(job_id)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_file(job.original_preview_path, "image/jpeg")
            return
        if path.startswith("/api/preview/"):
            job_id = path.rsplit("/", 1)[-1]
            job = jobs.get(job_id)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_file(job.preview_path, "image/jpeg")
            return
        if path.startswith("/api/download/"):
            job_id = path.rsplit("/", 1)[-1]
            job = jobs.get(job_id)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_file(job.output_path, "image/jpeg", job.output_path.name)
            return
        if path == "/api/download-all":
            self.download_all()
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self.upload()
            return
        if parsed.path == "/api/refresh-references":
            load_online_references()
            _, status = reference_snapshot()
            self.send_json({"ok": True, "status": status})
            return
        if parsed.path == "/api/clear":
            self.clear_jobs()
            return
        if parsed.path.startswith("/api/retry/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = requeue_job(job_id)
            if not job:
                self.send_json({"error": "没有找到这个任务。"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json({"job": job.public()})
            return
        if parsed.path.startswith("/api/inspect/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = jobs.get(job_id)
            if not job:
                self.send_json({"error": "没有找到这个任务。"}, HTTPStatus.NOT_FOUND)
                return
            report = inspect_raw_file(job.upload_path)
            log_event("inspect", "已检查文件", jobId=job.id, file=job.original_name, report=report)
            self.send_json({"report": report})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def upload(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self.send_json({"error": "没有收到文件。"}, HTTPStatus.BAD_REQUEST)
            return
        if length > 260 * 1024 * 1024:
            log_event("upload", "单次上传过大被拒绝", bytes=length)
            self.send_json({"error": "单次上传太大。请分批导入，每批 2 张以内的大 RAW 更稳定。"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return

        body = self.rfile.read(length)
        try:
            reader = MultipartReader(body, self.headers.get("Content-Type", ""))
            files = reader.files()
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        log_event("upload", "收到上传请求", bytes=length, files=len(files))
        for filename, content in files:
            if not filename or not content:
                continue
            if not is_supported(filename):
                rejected.append({"name": filename, "reason": "暂不支持这个扩展名"})
                log_event("upload", "文件扩展名不支持", file=filename)
                continue
            if len(content) > 140 * 1024 * 1024:
                rejected.append({"name": filename, "reason": "文件过大，建议先导出较小 RAW 或分辨率较低版本。"})
                log_event("upload", "单个文件过大被拒绝", file=filename, size=len(content))
                continue
            job = enqueue_file(filename, content)
            accepted.append(job.public())

        self.send_json({"accepted": accepted, "rejected": rejected, "rawAvailable": rawpy is not None})

    def clear_jobs(self) -> None:
        with jobs_lock:
            jobs.clear()
            save_jobs_locked()
        for directory in (UPLOAD_DIR, OUTPUT_DIR, PREVIEW_DIR):
            if directory.exists():
                shutil.rmtree(directory)
        ensure_dirs()
        while True:
            try:
                job_queue.get_nowait()
                job_queue.task_done()
            except Empty:
                break
        log_event("system", "已清空任务和输出")
        self.send_json({"ok": True})

    def download_all(self) -> None:
        ensure_dirs()
        zip_path = WORKSPACE_DIR / f"retouched_{int(time.time())}.zip"
        done_jobs = [job for job in jobs.values() if job.status == "done" and job.output_path.exists()]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for job in done_jobs:
                archive.write(job.output_path, job.output_path.name)
        self.send_file(zip_path, "application/zip", zip_path.name)


def run() -> None:
    ensure_dirs()
    load_jobs()
    recover_uploads_without_jobs()
    initialize_reference_bank()
    start_workers()
    enqueue_unfinished_jobs()
    port = int(os.environ.get("PORT", "8000"))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print(f"自动修图机已启动：http://127.0.0.1:{port}")
    print(f"RAW 支持：{'已启用 rawpy' if rawpy else '未启用，运行 install_dependencies.bat 可安装'}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
