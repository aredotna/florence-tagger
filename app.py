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

# -------------------------------
# Config
# -------------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "10"))

# Models (override via env)
FLORENCE_MODEL_ID = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-large-ft")
FLORENCE_REVISION = os.getenv("FLORENCE_REVISION", None)  # optional pin to avoid remote code drift
RAM_MODEL_ID = os.getenv("RAM_MODEL_ID", "xinyu1205/recognize-anything-plus-model")  # RAM++

# Florence task token (caption only)
TASK_CAPTION = "<DETAILED_CAPTION>"  # try "<CAPTION>" for terser output

# Stopwords / style keepers used for light cleanup
STOPWORDS = {
    "a","an","the","and","or","of","in","on","with","without","to","for","by","at","from",
    "that","this","these","those","me","set","answering","unanswerable","text",
    "no extra text","n/a","none","tag","tags","image","photo","picture","we can see","in this image"
}

KEEP_STYLE = {
    "woodcut","engraving","etching","lithograph","collage","ink drawing","cyanotype",
    "screen print","printmaking","oil painting","watercolor","charcoal","pencil drawing"
}

# -------------------------------
# S3 -> PIL loader
# -------------------------------
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

# -------------------------------
# Florence (caption only)
# -------------------------------
class FlorenceCaptioner:
    def __init__(self, model_id: str, revision: Optional[str] = None):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[boot] loading captioner: {model_id} rev={revision or 'latest'} on {self.device} …")

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    def _to_device_match_dtype(self, batch):
        out = {}
        dtype = next(self.model.parameters()).dtype if any(True for _ in self.model.parameters()) else None
        for k, v in batch.items():
            if hasattr(v, "to"):
                if hasattr(v, "dtype") and v.dtype.is_floating_point and dtype is not None:
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
                no_repeat_ngram_size=3,
            )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def caption(self, pil_img: Image.Image) -> str:
        cap = self._generate(TASK_CAPTION, pil_img, max_new_tokens=128, num_beams=5)
        if not cap or cap.lower() in {"no", "n/a"} or "<" in cap or "tag" in cap.lower():
            cap = self._generate("<CAPTION>", pil_img, max_new_tokens=96, num_beams=4)

        # normalize / trim boilerplate
        cap = cap.replace("\n", " ")
        cap = re.sub(r"\s+", " ", cap).strip()
        cap = re.sub(r"(?i)\b(we can see|in this image|this is)\b[^.]*\.?\s*", "", cap).strip()

        # keep it punchy
        words = cap.split()
        if len(words) > 22:
            cap = " ".join(words[:22]).rstrip(",;:") + "."
        return cap or "image"

# -------------------------------
# RAM++ (Recognize Anything) tagger
# -------------------------------
class RAMTagger:
    """
    Uses RAM/RAM++ to directly emit comma-separated keywords.
    Requires: einops, timm (no flash-attn needed).
    """
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[boot] loading RAM tagger: {model_id} on {self.device} …")

        # RAM repos expose generate via remote code
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        # cache model dtype for safe casting
        try:
            self.model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self.model_dtype = self.torch.float16 if self.device == "cuda" else self.torch.float32

    def _to_device_match_dtype(self, batch):
        out = {}
        for k, v in batch.items():
            if hasattr(v, "to"):
                if hasattr(v, "dtype") and v.dtype.is_floating_point:
                    out[k] = v.to(device=self.device, dtype=self.model_dtype)
                else:
                    out[k] = v.to(device=self.device)
            else:
                out[k] = v
        return out

    def raw_tags(self, pil_img: Image.Image) -> str:
        with self.torch.no_grad():
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = self._to_device_match_dtype(inputs)
            # RAM++’s remote code supports generate; max_new_tokens ~64 is enough for keywords
            out = self.model.generate(**inputs, max_new_tokens=64)
            text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
            return text.strip()

    @staticmethod
    def _normalize_tag(t: str) -> str:
        t = t.strip().lower()
        t = re.sub(r"[^\w\s\-&]", "", t)
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"^(a|an|the)\s+", "", t)
        if not t:
            return ""
        if t in STOPWORDS:
            return ""
        # keep art/media terms from KEEP_STYLE if present
        if t in {"image","photo","picture","tag","tags"}:
            return ""
        return t

    def clean_tags(self, tag_str: str) -> List[str]:
        parts = re.split(r"[,;\n]+", tag_str)
        seen, tags = set(), []
        for p in parts:
            t = self._normalize_tag(p)
            if not t:
                continue
            if 1 <= len(t.split()) <= 4:
                if t not in seen:
                    seen.add(t)
                    tags.append(t)
        return tags

    def topk(self, pil_img: Image.Image, k: int) -> List[str]:
        raw = self.raw_tags(pil_img)
        c = self.clean_tags(raw)
        if not c:
            return ["unknown"]
        return c[:k]

# -------------------------------
# Orchestrator
# -------------------------------
class Tagger:
    def __init__(self, captioner: FlorenceCaptioner, rammer: RAMTagger):
        self.captioner = captioner
        self.rammer = rammer

    def tag(self, pil_img: Image.Image, top_k: int) -> Tuple[List[dict], str]:
        # 1) direct keyword tags from RAM/RAM++
        names = self.rammer.topk(pil_img, top_k)
        labels = [{"Name": n.title(), "Confidence": 100.0} for n in names]  # confidences kept for API compatibility

        # 2) caption from Florence
        cap = self.captioner.caption(pil_img)
        return labels, cap

# -------------------------------
# FastAPI wiring
# -------------------------------
app = FastAPI(title="Are.na Image Tagger (RAM++) + Captioner (Florence)")

print("[boot] init …")
CAPTIONER = FlorenceCaptioner(FLORENCE_MODEL_ID, FLORENCE_REVISION)
RAMMER = RAMTagger(RAM_MODEL_ID)
RUNNER = Tagger(CAPTIONER, RAMMER)
print("[boot] ready.")

class TagReq(BaseModel):
    s3_uri: str
    top_k: Optional[int] = None
    include_caption: Optional[bool] = False

@app.get("/health")
def health():
    import torch
    return {
        "ok": True,
        "backend": "ram++ + florence",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/tag")
def tag(req: TagReq):
    img = s3_image(req.s3_uri)
    top_k = req.top_k or TOP_K_DEFAULT
    labels, cap = RUNNER.tag(img, top_k)
    body = {"Labels": labels}
    if req.include_caption:
        body["Caption"] = cap
    return body
