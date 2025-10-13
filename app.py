import io, os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import boto3
from botocore.config import Config

S3_REGION = os.getenv("AWS_REGION", "us-east-1")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "10"))
MODEL_ID = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-base")

def load_image_from_s3(s3_uri: str) -> Image.Image:
    if not s3_uri.startswith("s3://"):
        raise HTTPException(status_code=400, detail="s3_uri must start with s3://")
    _, path = s3_uri.split("s3://", 1)
    bucket, key = path.split("/", 1)
    s3 = boto3.client("s3", region_name=S3_REGION, config=Config(signature_version="s3v4"))
    obj = s3.get_object(Bucket=bucket, Key=key)
    return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")

class FlorenceTagger:
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

    @property
    def caption_prompt(self) -> str:
        return "Describe this image in a concise sentence."

    def generate_caption(self, pil_img):
        inputs = self.processor(
            text=self.caption_prompt,
            images=pil_img,
            return_tensors="pt"
        )

        # >>> ensure device + dtype match the model (fixes float vs half error)
        device = self.device
        dtype = getattr(self.model, "dtype", None)
        casted = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if dtype is not None and hasattr(v, "dtype") and v.dtype.is_floating_point:
                    casted[k] = v.to(device=device, dtype=dtype)
                else:
                    casted[k] = v.to(device)
            else:
                casted[k] = v
        inputs = casted
        # <<<

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=3
            )
        caption = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return caption.strip()

    def tag_from_caption(self, caption: str, top_k: int):
        doc = self.nlp(caption)
        phrases = []
        for chunk in doc.noun_chunks:
            txt = "".join(ch for ch in chunk.text.lower().strip() if ch.isalnum() or ch == " ").strip()
            if len(txt) >= 2:
                phrases.append(txt)
        # dedupe, prefer shorter phrases
        seen, kept = set(), []
        for p in phrases:
            if p not in seen:
                seen.add(p)
                kept.append(p)
        kept.sort(key=lambda s: (len(s.split()), s))
        labels = []
        for i, term in enumerate(kept[:top_k]):
            conf = max(50.0, 100.0 - i * (40.0 / max(1, top_k - 1)))
            labels.append({"Name": term.title(), "Confidence": round(conf, 1)})
        return labels

    def tag_image(self, pil_img: Image.Image, top_k: int):
        caption = self.generate_caption(pil_img)
        return self.tag_from_caption(caption, top_k), caption

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
    import torch
    return {"ok": True, "backend": "florence", "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.post("/tag")
def tag(req: TagReq):
    top_k = req.top_k or TOP_K_DEFAULT
    img = load_image_from_s3(req.s3_uri)
    labels, caption = TAGGER.tag_image(img, top_k=top_k)
    body = {"Labels": labels}
    if req.include_caption:
        body["Caption"] = caption
    return body
