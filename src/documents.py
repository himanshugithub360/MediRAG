"""Uploaded files: lab reports, prescriptions (PDF) and photos (rash, report, medicine label).

- Text PDFs: text is extracted with pypdf.
- Scanned PDFs (no text layer): the page images are read by the vision model.
- Images: the vision model transcribes any text and describes what is visible, without
  diagnosing. That description is stored and used to answer questions about the image.
"""
import base64
import io
import os
import re
import uuid

from pypdf import PdfReader

from src import storage
from src.config import MAX_PDF_PAGES, MAX_SCANNED_PAGES, MAX_UPLOAD_MB
from src.config import MAX_UPLOAD_TEXT_CHARS as MAX_TEXT_CHARS

IMAGE_TYPES = {"png", "jpg", "jpeg", "webp"}

VISION_PROMPT = (
    "You are helping a medical-information app read an image a user uploaded. "
    "1) If the image contains text (lab report, prescription, discharge summary, medicine label), "
    "transcribe ALL readable text accurately: keep test names, values, units, reference ranges and "
    "flags (H/L), and use a Markdown table for test results. Mark anything unreadable as [unreadable]. "
    "2) If it shows a body part, skin, wound or rash, describe only what is visible: location, colour, "
    "size relative to surroundings, shape, borders, texture, pattern and any swelling, blistering or "
    "bleeding. Do NOT name a diagnosis or condition. "
    "3) Otherwise, briefly describe the image. "
    "Say clearly when the image is blurry or you are unsure. Never give medical advice."
)

IMAGE_NOTICE = ("AI image reading can be wrong, and a photo can't replace an examination. "
                "Don't rely on it for a diagnosis; show the image to a doctor "
                "(a dermatologist for skin problems).")


class UploadError(ValueError):
    """A problem with the uploaded file that should be shown to the user."""


def _jpeg_base64(image_bytes, max_side=1600):
    """Verify the bytes are a real image, shrink it, and return base64 JPEG."""
    from PIL import Image, UnidentifiedImageError

    try:
        Image.open(io.BytesIO(image_bytes)).verify()
        img = Image.open(io.BytesIO(image_bytes))
    except (UnidentifiedImageError, OSError) as e:
        raise UploadError("That file isn't a readable image.") from e
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return base64.b64encode(out.getvalue()).decode("ascii")


def _save(data, ext):
    os.makedirs(storage.upload_dir(), exist_ok=True)
    path = os.path.join(storage.upload_dir(), f"{uuid.uuid4().hex}.{ext}")
    with open(path, "wb") as f:
        f.write(data)
    return path


def process_upload(filename, data, describe_image):
    """Read an uploaded file. describe_image(base64_jpeg, prompt) -> text (the vision model).

    Returns dict(kind, text, pages, path). Raises UploadError for unusable files.
    """
    name = os.path.basename(filename or "upload")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise UploadError(f"Files can be at most {MAX_UPLOAD_MB} MB.")

    if ext == "pdf" or data[:5] == b"%PDF-":
        if data[:5] != b"%PDF-":
            raise UploadError("That file isn't a valid PDF.")
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise UploadError("This PDF is password-protected. Please upload an unlocked copy.")
            pages = reader.pages[:MAX_PDF_PAGES]
            texts = [(p.extract_text() or "").strip() for p in pages]
        except UploadError:
            raise
        except Exception as e:
            raise UploadError("Couldn't read that PDF. It may be damaged.") from e

        text = "\n\n".join(f"[Page {i + 1}]\n{t}" for i, t in enumerate(texts) if t)
        if sum(len(t) for t in texts) < 80 * len(pages):
            # Little or no text layer: probably a scan or phone photo saved as PDF
            scanned = []
            for i, page in enumerate(pages[:MAX_SCANNED_PAGES]):
                images = list(getattr(page, "images", []) or [])
                if not images:
                    continue
                largest = max(images, key=lambda im: len(im.data))
                try:
                    scanned.append(f"[Page {i + 1}, read from the scanned image]\n"
                                   + describe_image(_jpeg_base64(largest.data), VISION_PROMPT))
                except UploadError:
                    continue
            if scanned:
                text = "\n\n".join(scanned)
        if not text.strip():
            raise UploadError("Couldn't find any readable text in that PDF.")
        return {"kind": "pdf", "text": text[:MAX_TEXT_CHARS], "pages": len(reader.pages),
                "path": _save(data, "pdf")}

    if ext in IMAGE_TYPES:
        image_b64 = _jpeg_base64(data)
        text = describe_image(image_b64, VISION_PROMPT).strip()
        if not text:
            raise UploadError("Couldn't read anything from that image.")
        return {"kind": "image", "text": text[:MAX_TEXT_CHARS], "pages": None,
                "path": _save(data, "jpg" if ext == "jpeg" else ext)}

    raise UploadError("Please upload a PDF or an image (PNG, JPG or WebP).")


# "Haemoglobin  11.2  g/dL  13.0 - 17.0"  /  "HbA1c 7.1 % < 5.7"  /  "Vitamin D 32 ng/mL >30"
_NUM = r"\d+(?:\.\d+)?"
_RANGE_ROW = re.compile(
    rf"^(?P<name>[A-Za-z][\w ().,/%+-]*?)\s*[:]?\s+(?P<value>{_NUM})\s+(?P<unit>[^\d\s<>]*)\s*"
    rf"(?:(?P<lo>{_NUM})\s*(?:-|–|to)\s*(?P<hi>{_NUM})|(?P<op><=?|>=?)\s*(?P<lim>{_NUM}))")


def check_ranges(text):
    """Add the app's own range check after each "test value unit range" line, e.g.
    "Creatinine 0.9 mg/dL 0.7 - 1.3  [app check: within range]". Small LLMs often get these
    number comparisons wrong, so the answer is computed here and given to the model."""
    out = []
    for line in text.splitlines():
        m = _RANGE_ROW.search(line.strip())
        verdict = None
        if m:
            value = float(m.group("value"))
            if m.group("lo") is not None:
                lo, hi = float(m.group("lo")), float(m.group("hi"))
                if lo < hi:
                    verdict = "LOW" if value < lo else "HIGH" if value > hi else "within range"
            else:
                lim, op = float(m.group("lim")), m.group("op")
                ok = {"<": value < lim, "<=": value <= lim, ">": value > lim, ">=": value >= lim}[op]
                verdict = "within range" if ok else ("HIGH" if op.startswith("<") else "LOW")
        out.append(f"{line}  [app check: {verdict}]" if verdict else line)
    return "\n".join(out)


def split_text(text, size=1000, overlap=150):
    parts, start = [], 0
    while start < len(text):
        parts.append(text[start:start + size])
        start += size - overlap
    return parts


def relevant_upload_context(attachments, query, score, max_chars=7000):
    """Text from the user's uploads to give the LLM: everything if it's short, otherwise the
    parts that best match the question (score(query, texts) -> list of floats)."""
    if not attachments:
        return ""
    total = sum(len(a["text"]) for a in attachments)
    blocks = []
    if total <= max_chars:
        for a in attachments:
            blocks.append(f'--- File: {a["filename"]} ---\n{check_ranges(a["text"])}')
        return "\n\n".join(blocks)

    pieces = [(a["filename"], p) for a in attachments for p in split_text(check_ranges(a["text"]))]
    scores = score(query, [p for _, p in pieces])
    chosen, used = [], 0
    for (fname, piece), _ in sorted(zip(pieces, scores), key=lambda x: x[1], reverse=True):
        if used + len(piece) > max_chars:
            break
        chosen.append(f"--- From {fname} ---\n{piece}")
        used += len(piece)
    return "\n\n".join(chosen)
