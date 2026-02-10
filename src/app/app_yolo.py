from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO
import torch


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Detectare Defecte Sticlă (YOLO)",
    page_icon="🔍",
    layout="wide"
)

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# =========================
# PATH RESOLUTION (ROBUST)
# =========================
def find_project_root(start: Path) -> Path:
    """
    Caută root-ul proiectului urcând în sus până găsește repere clare.
    """
    current = start.resolve()
    for _ in range(8):  # max 8 nivele în sus
        if (current / "models").exists() and (current / "src").exists() and (current / "data").exists():
            return current
        if (current / "requirements.txt").exists():
            return current
        current = current.parent
    # fallback: păstrăm vechiul comportament
    return start.resolve().parents[2]


PROJECT_ROOT = find_project_root(Path(__file__).parent)
MODEL_PATH = PROJECT_ROOT / "models" / "yolo26" / "best.pt"


# =========================
# HELPERS
# =========================
@st.cache_resource
def load_yolo_model(model_path: str, model_mtime: float):
    # model_mtime este doar ca să invalideze cache-ul când se schimbă fișierul
    _ = model_mtime
    return YOLO(model_path)


def draw_boxes(image: Image.Image, boxes, names, conf_threshold: float):
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    detections = []

    for b in boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        label = names.get(cls_id, str(cls_id))

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        text = f"{label} {conf:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        y_text_top = max(0, y1 - th - 6)
        draw.rectangle([x1, y_text_top, x1 + tw + 8, y_text_top + th + 6], fill="red")
        draw.text((x1 + 4, y_text_top + 3), text, fill="white", font=font)

        detections.append({
            "label": label,
            "confidence": conf,
            "box": (x1, y1, x2, y2)
        })

    return img, detections


def ensure_model_exists():
    if not MODEL_PATH.exists():
        st.error(
            "Nu găsesc modelul YOLO.\n\n"
            f"**MODEL_PATH:** `{MODEL_PATH}`\n\n"
            "Asigură-te că fișierul `best.pt` există în:\n"
            "`models/yolo26/best.pt`"
        )
        st.stop()


# =========================
# UI
# =========================
st.title("Sistem de Detectare Defecte pe Sticlă (YOLO)")
st.write(
    "Aplicația utilizează un model YOLO antrenat pentru **detecția localizată** a defectelor "
    "(bounding box + etichetă + scor)."
)


ensure_model_exists()

model_mtime = MODEL_PATH.stat().st_mtime
model = load_yolo_model(str(MODEL_PATH), model_mtime)

with st.sidebar:
    st.header("Setări inferență")
    conf_th = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05)
    iou_th = st.slider("IoU threshold", 0.10, 0.90, DEFAULT_IOU, 0.05)
    imgsz = st.selectbox("Image size (imgsz)", [640, 768, 896, 1024], index=3)
    st.caption(f"Device: {DEVICE} (0=GPU, cpu=CPU)")
    st.caption("Dacă ai erori de memorie (CUDA OOM), scade imgsz.")

uploaded = st.file_uploader("Încărcați o imagine (JPG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Încarcă o imagine ca să rulezi detecția.")
    st.stop()

image = Image.open(uploaded).convert("RGB")

# =========================
# INFERENCE
# =========================
with st.spinner("Rulare inferență YOLO..."):
    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf_th,
        iou=iou_th,
        verbose=False,
        device=DEVICE
    )

res0 = results[0]
boxes = res0.boxes if res0.boxes is not None else []

annotated, dets = draw_boxes(image, boxes, res0.names, conf_th)

st.subheader("Rezultat detecție (YOLO)")
st.image(annotated, width=700, caption="Imagine cu defecte detectate")

st.divider()

# =========================
# DETECTIONS SUMMARY
# =========================
if len(dets) == 0:
    st.success("OK – Nu au fost detectate defecte peste pragul de confidence.")
else:
    st.error(f"Defecte detectate: {len(dets)}")
    dets = sorted(dets, key=lambda x: x["confidence"], reverse=True)

    counts = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    st.write("**Sumar pe clase:** " + ", ".join([f"{k}: {v}" for k, v in counts.items()]))

    for i, d in enumerate(dets, start=1):
        st.write(
            f"**{i}. {d['label']}** — confidence: **{d['confidence']*100:.2f}%** "
            f"— box: ({int(d['box'][0])}, {int(d['box'][1])}) → ({int(d['box'][2])}, {int(d['box'][3])})"
        )
