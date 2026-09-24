"""OCR adapter â€” EasyOCR wrapper for extracting text from images.

Used by the web scraper to extract text from images in scraped articles.
Also usable standalone for E4 OCR benchmarking.

Includes intelligent image classification to skip decorative images
(logos, icons, photographs without text) and only run OCR on images
that likely contain text (charts, diagrams, screenshots, figures with labels).

Usage:
    from ipa.acquisition.ocr_adapter import OCRAdapter

    ocr = OCRAdapter(languages=["en", "es"], gpu=True)
    text = ocr.extract_text("path/to/image.png")
    # Or batch:
    results = ocr.extract_texts(["img1.png", "img2.png"])
    # Intelligent mode (skip decorative images):
    results = ocr.extract_texts_smart(["img1.png", "img2.png"])
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OCRResult:
    """Result of OCR on a single image."""
    image_path: str
    text: str
    confidence: float  # average confidence across detections
    detections: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None
    skipped: bool = False  # True if image was skipped by smart OCR
    skip_reason: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Image classifier â€” detect if an image likely contains text
# ---------------------------------------------------------------------------

class ImageClassifier:
    """Classify images as 'text-bearing' vs 'decorative' without OCR.

    Uses lightweight heuristics to avoid running EasyOCR on every image:
      1. File size: tiny images (<2KB) are likely icons/buttons
      2. Dimensions: very small images (<50px) are likely icons
      3. Aspect ratio: extremely wide/tall images are likely banners/decor
      4. Edge density: text-bearing images (charts, screenshots, diagrams)
         have high edge density compared to photographs
      5. Color variance: text images tend to have fewer unique colors
         than photographs (text is usually high-contrast on solid bg)

    No external dependencies beyond PIL/Pillow (already a transitive dep
    of EasyOCR).
    """

    # Thresholds (tuned for typical web article images)
    MIN_FILE_SIZE = 2048          # 2KB â€” skip tiny icons
    MIN_DIMENSION = 50            # px â€” skip tiny images
    MAX_ASPECT_RATIO = 10.0       # skip ultra-wide/tall banners
    # Max file size for OCR â€” images larger than this are downscaled or
    # skipped entirely to avoid GPU OOM (we saw 7GB allocations crash).
    MAX_FILE_SIZE = 20 * 1024 * 1024   # 20MB â€” skip huge images
    MAX_DIMENSION = 4000              # px â€” downscale before OCR if larger
    # Edge density: text on white bg has low average (0.01-0.15),
    # random noise has very high (>0.4), photographs are moderate (0.05-0.2)
    MIN_EDGE_DENSITY = 0.008      # below this â†’ photograph (smooth gradients)
    MAX_EDGE_DENSITY = 0.45       # above this â†’ likely noise, not text
    # Color count: text images have few unique colors (high-contrast text
    # on solid background). Photographs and noise have many.
    MAX_UNIQUE_COLORS_TEXT = 8000  # text-bearing images typically <8K colors

    def __init__(self) -> None:
        self._pil = None

    def _load_pil(self):
        if self._pil is None:
            from PIL import Image
            self._pil = Image
        return self._pil

    def classify(self, image_path: str | Path) -> tuple[bool, str]:
        """Classify an image. Returns (should_ocr, reason).

        should_ocr=True means the image likely contains text.
        reason explains the classification decision.
        """
        image_path = Path(image_path)
        Image = self._load_pil()

        # 1. File size check â€” skip tiny icons AND huge images (OOM protection)
        file_size = image_path.stat().st_size
        if file_size < self.MIN_FILE_SIZE:
            return False, f"file_too_small ({file_size}B)"
        if file_size > self.MAX_FILE_SIZE:
            return False, f"file_too_large ({file_size / 1024 / 1024:.1f}MB > {self.MAX_FILE_SIZE / 1024 / 1024:.0f}MB limit)"

        try:
            img = Image.open(image_path)
            w, h = img.size

            # 2. Dimension check
            if w < self.MIN_DIMENSION or h < self.MIN_DIMENSION:
                return False, f"dimensions_too_small ({w}x{h})"

            # 3. Aspect ratio check
            if w > 0 and h > 0:
                ratio = max(w, h) / min(w, h)
                if ratio > self.MAX_ASPECT_RATIO:
                    return False, f"extreme_aspect_ratio ({ratio:.1f})"

            # 4. Edge density (convert to grayscale, detect edges)
            gray = img.convert("L")
            pixels = list(gray.getdata())
            if len(pixels) < 100:
                return False, "too_few_pixels"

            # Simple edge detection: count pixels that differ significantly
            # from their horizontal neighbor
            w_gray = gray.size[0]
            edge_count = 0
            for i in range(len(pixels) - 1):
                if (i + 1) % w_gray == 0:  # skip row boundaries
                    continue
                if abs(pixels[i] - pixels[i + 1]) > 30:
                    edge_count += 1
            edge_density = edge_count / len(pixels)

            # 5. Color variance
            rgb_img = img.convert("RGB")
            colors = rgb_img.getcolors(maxcolors=self.MAX_UNIQUE_COLORS_TEXT + 1)
            unique_colors = len(colors) if colors else self.MAX_UNIQUE_COLORS_TEXT + 1

            # Decision logic:
            # 1. Very high edge density â†’ random noise, not text
            if edge_density > self.MAX_EDGE_DENSITY:
                return False, f"noise_like (edge={edge_density:.3f})"

            # 2. Low edge density â†’ smooth photograph/gradient
            if edge_density < self.MIN_EDGE_DENSITY:
                return False, f"photograph_like (edge={edge_density:.3f})"

            # 3. Moderate edge density + few colors â†’ text-bearing
            if unique_colors < self.MAX_UNIQUE_COLORS_TEXT:
                return True, f"text_like (edge={edge_density:.3f}, colors={unique_colors})"

            # 4. Moderate edge density + many colors â†’ uncertain
            # Could be a screenshot with anti-aliased text or a complex chart
            # Err on the side of OCR (better to run OCR and find nothing
            # than to miss text in a chart/diagram)
            return True, f"uncertain_ocr_anyway (edge={edge_density:.3f}, colors={unique_colors})"

        except Exception as e:
            # If we can't classify, err on the side of running OCR
            return True, f"classification_error ({e})"


class OCRAdapter:
    """EasyOCR wrapper with lazy model loading.

    The model is loaded on first use to avoid GPU/CPU initialization
    when OCR is not needed.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = True,
        detail: int = 1,
        paragraph: bool = False,
        model_storage_directory: str | None = None,
    ) -> None:
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.detail = detail
        self.paragraph = paragraph
        self.model_storage_directory = model_storage_directory
        self._reader = None  # lazy load

    def _load_reader(self) -> None:
        """Load EasyOCR reader (lazy)."""
        if self._reader is not None:
            return
        # Bajo MODEL_LOAD_LOCK: easyocr.Reader construye nn.Modules; si
        # corre concurrente con un from_pretrained (register_parameter
        # parcheado globalmente) quedan params en 'meta' (PM-007).
        from ipa.model_load_lock import MODEL_LOAD_LOCK
        with MODEL_LOAD_LOCK:
            if self._reader is not None:
                return
            import easyocr
            kwargs = {
                "lang_list": self.languages,
                "gpu": self.gpu,
            }
            if self.model_storage_directory:
                kwargs["model_storage_directory"] = self.model_storage_directory
            self._reader = easyocr.Reader(**kwargs)

    def extract_text(self, image_path: str | Path) -> OCRResult:
        """Extract text from a single image.

        Large images (>MAX_DIMENSION px) are downscaled before OCR to
        avoid GPU OOM.  The downscaled copy is temporary and deleted after.
        """
        start = time.monotonic()
        image_path = Path(image_path)
        if not image_path.exists():
            return OCRResult(
                image_path=str(image_path), text="", confidence=0.0,
                error=f"Image not found: {image_path}",
                elapsed_seconds=time.monotonic() - start,
            )

        # Downscale large images to avoid OOM
        ocr_path = image_path
        tmp_path = None
        try:
            from PIL import Image as PILImage
            img = PILImage.open(image_path)
            w, h = img.size
            if max(w, h) > ImageClassifier.MAX_DIMENSION:
                scale = ImageClassifier.MAX_DIMENSION / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), PILImage.LANCZOS)
                import tempfile, os
                fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="ocr_downscaled_")
                os.close(fd)
                img.save(tmp_path)
                ocr_path = Path(tmp_path)

            self._load_reader()
            results = self._reader.readtext(
                str(ocr_path),
                detail=self.detail,
                paragraph=self.paragraph,
            )
        except Exception as exc:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            return OCRResult(
                image_path=str(image_path), text="", confidence=0.0,
                error=str(exc), elapsed_seconds=time.monotonic() - start,
            )

        # Parse results
        detections: list[dict[str, Any]] = []
        texts: list[str] = []
        confidences: list[float] = []

        if self.paragraph:
            # paragraph mode returns list of strings
            for text in results:
                texts.append(text)
                detections.append({"text": text, "confidence": 1.0})
                confidences.append(1.0)
        else:
            # detail mode returns (bbox, text, confidence)
            for detection in results:
                if len(detection) >= 3:
                    bbox, text, conf = detection[0], detection[1], detection[2]
                    texts.append(text)
                    confidences.append(float(conf))
                    detections.append({
                        "text": text,
                        "confidence": float(conf),
                        "bbox": bbox if not isinstance(bbox, str) else None,
                    })

        full_text = "\n".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Clean up downscaled temp file
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

        return OCRResult(
            image_path=str(image_path),
            text=full_text,
            confidence=avg_conf,
            detections=detections,
            elapsed_seconds=time.monotonic() - start,
        )

    def extract_texts(
        self, image_paths: list[str | Path]
    ) -> list[OCRResult]:
        """Extract text from multiple images (sequential)."""
        return [self.extract_text(p) for p in image_paths]

    def extract_texts_smart(
        self, image_paths: list[str | Path],
        min_confidence: float = 0.4,
    ) -> list[OCRResult]:
        """Extract text from images, skipping decorative images.

        Uses ImageClassifier to pre-filter images. Only images classified
        as 'likely containing text' are sent to EasyOCR. Results with
        confidence below min_confidence are kept but marked.

        Returns OCRResult for every image (skipped ones have skipped=True).
        """
        classifier = ImageClassifier()
        results: list[OCRResult] = []

        for path in image_paths:
            start = time.monotonic()
            path = Path(path)

            if not path.exists():
                results.append(OCRResult(
                    image_path=str(path), text="", confidence=0.0,
                    error=f"Image not found: {path}",
                    elapsed_seconds=0.0,
                ))
                continue

            # Classify
            should_ocr, reason = classifier.classify(path)

            if not should_ocr:
                results.append(OCRResult(
                    image_path=str(path), text="", confidence=0.0,
                    skipped=True, skip_reason=reason,
                    elapsed_seconds=time.monotonic() - start,
                ))
                continue

            # Run OCR
            result = self.extract_text(path)
            # Filter by confidence
            if result.success and result.confidence < min_confidence and not result.text.strip():
                result.skipped = True
                result.skip_reason = f"low_confidence ({result.confidence:.2f})"
            results.append(result)

        return results

    def extract_text_simple(self, image_path: str | Path) -> str:
        """Extract text as a plain string (convenience)."""
        result = self.extract_text(image_path)
        return result.text if result.success else ""

    def close(self) -> None:
        """Release model resources."""
        if self._reader is not None:
            # EasyOCR doesn't have an explicit close, but we can release
            # the reference and let GC handle it
            self._reader = None

    def __enter__(self) -> "OCRAdapter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

