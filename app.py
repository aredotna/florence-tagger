# app.py
import io, os, re
from typing import Optional, List, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageFile
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ImageFile.LOAD_TRUNCATED_IMAGES = True

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "10"))

# Florence for caption only (you already had this working)
FLORENCE_MODEL_ID = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-large-ft")
TASK_CAPTION = "<DETAILED_CAPTION>"

# RAM: use the repo’s code + HF .pth files
RAM_VARIANT = os.getenv("RAM_VARIANT", "ram")  # "ram" or "ram++" (ram_plus)
RAM_WEIGHTS = os.getenv("RAM_WEIGHTS", "/models/ram/ram_swin_large_14m.pth")
RAM_TAG_EMB = os.getenv("RAM_TAG_EMB", "/models/ram/ram_tag_embedding_class_4585.pth")

STOPWORDS = {
    "a","an","the","and","or","of","in","on","with","without","to","for","by","at","from",
    "that","this","these","those","me","set","answering","unanswerable","text",
    "no extra text","n/a","none","tag","tags","image","photo","picture"
}

def s3_image(s3_uri: str) -> Image.Image:
    if not s3_uri.startswith("s3://"):
        raise HTTPException(status_code=400, detail="s3_uri must start with s3://")
    _, path = s3_uri.split("s3://", 1)
    try:
        bucket, key = path.split("/", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid s3_uri (missing key)")
    s3 = boto3.client("s3", region_name=AWS_REGION, config=Config(signature_version="s3v4"))
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

# ---------------- Florence captioner (unchanged except for small cleanup) ---------------
class FlorenceCaptioner:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[boot] loading captioner: {model_id} on {self.device} …")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def _to_device_match_dtype(self, batch):
        out = {}
        dtype = next(self.model.parameters()).dtype
        for k, v in batch.items():
            if hasattr(v, "to"):
                if hasattr(v, "dtype") and v.dtype.is_floating_point:
                    out[k] = v.to(device=self.device, dtype=dtype)
                else:
                    out[k] = v.to(device=self.device)
            else:
                out[k] = v
        return out

    def _generate(self, task_token: str, pil_img: Image.Image, max_new_tokens=96, num_beams=4) -> str:
        inputs = self.processor(text=task_token, images=pil_img, return_tensors="pt")
        inputs = self._to_device_match_dtype(inputs)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                length_penalty=1.05,
            )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def caption(self, pil_img: Image.Image) -> str:
        cap = self._generate(TASK_CAPTION, pil_img, max_new_tokens=128, num_beams=5)
        if not cap or cap.lower() in {"no","n/a"} or "<" in cap or "tag" in cap.lower():
            cap = self._generate("<CAPTION>", pil_img, max_new_tokens=96, num_beams=4)
        cap = re.sub(r"\s+", " ", cap.replace("\n", " ")).strip()
        return cap or "image"

# ---------------- RAM (base) tagger via official repo (no Transformers) -----------------
class RAMTagger:
    """
    Loads RAM using the official recognize-anything repo and local .pth checkpoints.
    This avoids the Transformers 'config.json' path entirely.
    """
    def __init__(self, variant: str, weights_path: str, tag_emb_path: str):
        import torch
        from ram import utils as ram_utils
        from ram import inference_ram, inference_ram_plus
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.variant = variant.lower()
        print(f"[boot] loading RAM ({self.variant}) from {weights_path} …")

        # Build preprocessing & tag list using repo utilities
        # Note: RAM uses a fixed tag list of 4,585 classes; the embedding file encodes them.
        self.transform = ram_utils.get_transform(image_size=384)
        self.tag_list = ram_utils.get_tag_list()           # list of strings
        self.tag_emb_path = tag_emb_path
        self.weights_path = weights_path

        # Prepare the actual inference function we’ll call
        # (The repo provides helpers for both RAM and RAM++)
        if self.variant == "ram++" or self.variant == "ram_plus" or self.variant == "ramplusplus":
            self._infer = lambda im: inference_ram_plus.infer_tags(
                image=im,
                model_ckpt=self.weights_path,
                tag_emb_path=self.tag_emb_path,
                device=self.device,
                image_size=384,
            )
        else:
            self._infer = lambda im: inference_ram.infer_tags(
                image=im,
                model_ckpt=self.weights_path,
                tag_emb_path=self.tag_emb_path,
                device=self.device,
                image_size=384,
            )

    def topk(self, pil_img: Image.Image, k: int) -> List[str]:
        """
        Calls the repo’s inference helper. It returns a comma-separated string of tags
        or a Python list depending on version; normalize to a list of lowercased terms.
        """
        tags = self._infer(pil_img)  # may return a single string or list
        if isinstance(tags, str):
            parts = re.split(r"[,;\n]+", tags)
        else:
            parts = list(tags)
        out, seen = [], set()
        for p in parts:
            t = re.sub(r"[^\w\s\-&]", "", str(p).strip().lower())
            t = re.sub(r"\s+", " ", t)
            t = re.sub(r"^(a|an|the)\s+", "", t)
            if not t or t in STOPWORDS:
                continue
            if 1 <= len(t.split()) <= 4 and t not in seen:
                seen.add(t)
                out.append(t)
            if len(out) >= k:
                break
        return out or ["unknown"]

# ---------------- Orchestrator -----------------
class Tagger:
    def __init__(self, captioner: FlorenceCaptioner, rammer: RAMTagger):
        self.captioner = captioner
        self.rammer = rammer

    def tag(self, pil_img: Image.Image, top_k: int) -> Tuple[List[dict], str]:
        names = self.rammer.topk(pil_img, top_k)
        labels = [{"Name": n.title(), "Confidence": 100.0} for n in names]  # keep shape
        cap = self.captioner.caption(pil_img)
        return labels, cap

# ---------------- FastAPI -----------------
app = FastAPI(title="Are.na Tagger (RAM) + Captioner (Florence)")

print("[boot] init …")
CAPTIONER = FlorenceCaptioner(FLORENCE_MODEL_ID)
RAMMER = RAMTagger(RAM_VARIANT, RAM_WEIGHTS, RAM_TAG_EMB)
RUNNER = Tagger(CAPTIONER, RAMMER)
print("[boot] ready.")

class TagReq(BaseModel):
    s3_uri: str
    top_k: Optional[int] = None
    include_caption: Optional[bool] = False

@app.get("/health")
def health():
    import torch
    return {"ok": True, "backend": f"{RAM_VARIANT}+florence", "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.post("/tag")
def tag(req: TagReq):
    img = s3_image(req.s3_uri)
    top_k = req.top_k or TOP_K_DEFAULT
    labels, cap = RUNNER.tag(img, top_k)
    body = {"Labels": labels}
    if req.include_caption:
        body["Caption"] = cap
    return body
