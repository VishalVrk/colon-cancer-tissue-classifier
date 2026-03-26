import gradio as gr
import io
import os
from huggingface_hub import InferenceClient

# ── Inference API Client (no torch / no local model download) ─────────────────
MODEL_ID = "polejowska/resnet-50-finetuned-nct-crc-he-45k"
client = InferenceClient(token=os.environ.get("HF_TOKEN"))

LABEL_DESCRIPTIONS = {
    "ADI":  "Adipose tissue",
    "BACK": "Background / glass",
    "DEB":  "Debris",
    "LYM":  "Lymphocytes",
    "MUC":  "Mucus",
    "MUS":  "Smooth muscle",
    "NORM": "Normal colon mucosa",
    "STR":  "Cancer-associated stroma",
    "TUM":  "Colorectal adenocarcinoma (Tumor)",
}


# ── Prediction function ───────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return {"error": "No image provided"}

    # Convert PIL image → bytes for Inference API
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    results = client.image_classification(buf, model=MODEL_ID)
    top = results[0]
    label = top.label

    return {
        "label":       label,
        "description": LABEL_DESCRIPTIONS.get(label, label),
        "confidence":  round(top.score * 100, 1),
        "is_cancer":   label == "TUM",
        "scores": [
            {
                "label":       r.label,
                "description": LABEL_DESCRIPTIONS.get(r.label, r.label),
                "score":       round(r.score * 100, 1),
            }
            for r in results
        ],
    }


# ── Gradio Interface ──────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="H&E Histopathology Patch (224×224)"),
    outputs=gr.JSON(label="Classification Result"),
    title="Colorectal Cancer Tissue Classifier",
    description=(
        "Upload an H&E stained tissue patch. "
        "Classifies into 9 CRC tissue types using a ResNet-50 trained on NCT-CRC-HE-45K."
    ),
    api_name="predict",
)

demo.launch()
