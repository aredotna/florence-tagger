import io
import os
import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageFile
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# make PIL tolerant of slightly truncated files
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------
# Config
# -----------------------------
S3_REGION = os.getenv("AWS_REGION", "us-east-1")  # you can unset this if buckets live in multiple regions
TOP_K_DEFAULT = int(os.getenv("TOP_K", "10"))
MODEL_ID = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-base")

# optional: allow TF32 on Ampere+ for perf (harmless if unsupported)
try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass


# -----------------------------
# S3 loader with clear errors
# -----------------------------
def load_image_from_s3(s3_uri: str) -> Image.Image:
    if not s3_uri.startswith("s3://"):
        raise HTTPException(status_code=400, detail="s3_uri must start with s3://")
    _, path = s3_uri.split("s3://", 1)
    try:
        bucket, key = path.split("/", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid s3_uri (missing key)")
    # NOTE: if you hit region errors, remove region_name to let boto3 resolve automatically
    s3 = boto3.client("s3", region_name=S3_REGION, config=Config(signature_version="s3v4"))
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response["Error"].get("Code", "UnknownError")
        msg = e.response["Error"].get("Message", str(e))
        raise HTTPException(status_code=400, detail=f"S3 error {code}: {msg}")
    try:
        return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")


# -----------------------------
# Florence backend
# -----------------------------
class FlorenceTagger:
    """
    Uses Florence-2 to produce:
      1) A vivid one-sentence caption
      2) A comma-separated tag string (short nouns/adjectives, specific first)
    Then converts tags -> Rekognition-like Labels.
    """

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        import spacy

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[boot] loading model {model_id} on {self.device} …")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)

        print("[boot] loading spaCy en_core_web_sm …")
        self.nlp = spacy.load("en_core_web_sm")

    # prompts
    @property
    def caption_prompt(self) -> str:
        return (
            "caption examples:\n"
            "photo of a dog playing in the park\n"
            "portrait of a woman wearing a red hat\n"
            "abstract painting with geometric shapes and bright colors\n"
            "Now write a caption for this image:"
        )

    @property
    def tags_prompt(self) -> str:
        return (
            "tag examples:\n"
            "dog, park, running, grass, playful\n"
            "woman, red hat, portrait, fashion, studio\n"
            "abstract, painting, geometric, colorful, modern\n"
            "Now write tags for this image:"
        )


    # shared generate helper with device/dtype casting (fixes float/half mismatch)
    def _gen(self, text, pil_img, max_new_tokens=96, num_beams=4):
        inputs = self.processor(text=text, images=pil_img, return_tensors="pt")
        device = self.device
        dtype = getattr(self.model, "dtype", None)

        casted = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if dtype is not None and hasattr(v, "dtype") and getattr(v, "dtype", None) and v.dtype.is_floating_point:
                    casted[k] = v.to(device=device, dtype=dtype)
                else:
                    casted[k] = v.to(device)
            else:
                casted[k] = v

        with self.torch.no_grad():
            out = self.model.generate(
                **casted,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                length_penalty=1.05,
            )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def generate_caption(self, pil_img):
        return self._gen(self.caption_prompt, pil_img, max_new_tokens=128, num_beams=5)

    def generate_tags_text(self, pil_img):
        return self._gen(self.tags_prompt, pil_img, max_new_tokens=64, num_beams=4)

    # parse/clean Florence tag string -> Labels[]
    def tags_to_labels(self, tag_str: str, top_k: int):
        raw = [t.strip().lower() for t in tag_str.split(",")]
        stop = {
            "a", "an", "the", "and", "or", "of", "in", "on", "with", "without",
            "to", "for", "by", "at", "from", "that", "this", "me", "photo", "image", "picture", "set"
        }
        cleaned = []
        seen = set()
        for t in raw:
            t = "".join(ch for ch in t if ch.isalnum() or ch == " ").strip()
            if not t or t in stop or len(t) < 2:
                continue
            # keep short, specific phrases
            if len(t.split()) <= 4 and t not in seen:
                seen.add(t)
                cleaned.append(t)

        # prefer shorter & lexicographic for stability (specific first already in model output)
        cleaned = sorted(cleaned, key=lambda s: (len(s.split()), s))

        labels = []
        for i, term in enumerate(cleaned[:top_k]):
            conf = max(55.0, 100.0 - i * (45.0 / max(1, top_k - 1)))  # 55–100 band
            labels.append({"Name": term.title(), "Confidence": round(conf, 1)})
        return labels

    # simple backup: mine noun-ish chunks from the caption if tags were weak
    def backup_labels_from_caption(self, caption: str, top_k: int):
        phrases = []
        doc = self.nlp(caption)
        for chunk in doc.noun_chunks:
            txt = "".join(ch for ch in chunk.text.lower().strip() if ch.isalnum() or ch == " ").strip()
            if len(txt) >= 2:
                phrases.append(txt)
        seen, kept = set(), []
        for p in phrases:
            if p not in seen:
                seen.add(p)
                kept.append(p)
        kept = sorted(kept, key=lambda s: (len(s.split()), s))
        labels = []
        for i, term in enumerate(kept[:top_k]):
            conf = max(50.0, 95.0 - i * (40.0 / max(1, top_k - 1)))
            labels.append({"Name": term.title(), "Confidence": round(conf, 1)})
        return labels

    def tag_image(self, pil_img, top_k: int):
        tag_text = self.generate_tags_text(pil_img)
        caption = self.generate_caption(pil_img)
        labels = self.tags_to_labels(tag_text, top_k)
        if len(labels) < max(3, top_k // 2):
            # fallback to caption-derived labels if Florence's tag string was too generic
            labels = self.backup_labels_from_caption(caption, top_k)
        return labels, caption


# -----------------------------
# API
# -----------------------------
app = FastAPI(title="Florence Keyword Tagger")

print("[boot] initializing FlorenceTagger …")
TAGGER = FlorenceTagger(MODEL_ID)
print("[boot] ready.")


class TagReq(BaseModel):
    s3_uri: str
    top_k: Optional[int] = None
    include_caption: Optional[bool] = False


@app.get("/health")
def health():
    import torch as _torch
    return {"ok": True, "backend": "florence", "device": "cuda" if _torch.cuda.is_available() else "cpu"}


@app.post("/tag")
def tag(req: TagReq):
    top_k = req.top_k or TOP_K_DEFAULT
    img = load_image_from_s3(req.s3_uri)
    labels, caption = TAGGER.tag_image(img, top_k=top_k)
    body = {"Labels": labels}
    if req.include_caption:
        body["Caption"] = caption
    return body
