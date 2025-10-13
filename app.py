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
FLORENCE_MODEL_ID = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-large-ft")  # instruction-tuned

# Florence task tokens (no prose!)
TASK_CAPTION = "<DETAILED_CAPTION>"  # try "<CAPTION>" if you prefer shorter lines
TASK_TAGS    = "<TAGS>"

STOPWORDS = {
    "a","an","the","and","or","of","in","on","with","without","to","for","by","at","from",
    "that","this","me","photo","image","picture","set","answering","unanswerable","text",
    "no extra text", "n/a", "none"
}

def s3_image(s3_uri: str) -> Image.Image:
    if not s3_uri.startswith("s3://"):
        raise HTTPException(status_code=400, detail="s3_uri must start with s3://")
    _, path = s3_uri.split("s3://", 1)
    try:
        bucket, key = path.split("/", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid s3_uri (missing key)")
    # If you hit region errors, remove region_name to let boto resolve automatically
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

class FlorenceRunner:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[boot] loading {model_id} on {self.device} …")

        # processor/model with remote code (Florence needs this)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)

        # perf knobs (harmless if CPU)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    def _generate(self, task_token: str, pil_img: Image.Image, max_new_tokens=96, num_beams=4) -> str:
        # IMPORTANT: pass the task token as the *text* (no examples/prose)
        inputs = self.processor(text=task_token, images=pil_img, return_tensors="pt")
        dtype = getattr(self.model, "dtype", None)

        # ensure device + dtype match model (fixes float vs half)
        casted = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if dtype is not None and hasattr(v, "dtype") and v.dtype.is_floating_point:
                    casted[k] = v.to(device=self.device, dtype=dtype)
                else:
                    casted[k] = v.to(device=self.device)
            else:
                casted[k] = v

        with self.torch.no_grad():
            out = self.model.generate(
                **casted,
                max_new_tokens=max_new_tokens,
                do_sample=False,        # deterministic
                num_beams=num_beams,    # a bit more thorough
                length_penalty=1.05,
            )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def caption(self, pil_img: Image.Image) -> str:
        cap = self._generate(TASK_CAPTION, pil_img, max_new_tokens=128, num_beams=5)
        # guard against weird echoes
        if not cap or cap.lower() in {"no","n/a"} or "<" in cap or "tag" in cap.lower():
            cap = self._generate("<CAPTION>", pil_img, max_new_tokens=96, num_beams=4)
        return cap.replace("\n", " ").strip()

    def tag_text(self, pil_img: Image.Image) -> str:
        txt = self._generate(TASK_TAGS, pil_img, max_new_tokens=64, num_beams=4)
        # guard against invalid output (echoed instructions)
        if "<" in txt or "example" in txt.lower() or len(txt) < 3:
            txt = self._generate("<TAGS>", pil_img, max_new_tokens=48, num_beams=4)
        return txt.strip()

    # cleaning helpers
    def clean_tags(self, tag_str: str) -> List[str]:
        # Split on commas and semicolons, also handle newlines
        raw = re.split(r"[,;\n]+", tag_str)
        cleaned, seen = [], set()
        for t in raw:
            t = t.strip().lower()
            t = "".join(ch for ch in t if ch.isalnum() or ch == " ").strip()
            if not t or t in STOPWORDS:
                continue
            # drop overly generic meta words
            if t in {"photo", "image", "picture", "tag", "tags"}:
                continue
            # short phrases only
            if 1 <= len(t.split()) <= 4 and t not in seen:
                seen.add(t)
                cleaned.append(t)
        return cleaned

    def labels_from_candidates(self, cands: List[str], top_k: int) -> List[dict]:
        # simple descending confidence by rank; you can swap for CLIP re-rank later
        labels = []
        for i, term in enumerate(cands[:top_k]):
            conf = max(55.0, 100.0 - i * (45.0 / max(1, top_k - 1)))
            labels.append({"Name": term.title(), "Confidence": round(conf, 1)})
        return labels

    def tag(self, pil_img: Image.Image, top_k: int) -> Tuple[List[dict], str]:
        tag_str = self.tag_text(pil_img)
        cap = self.caption(pil_img)
        cands = self.clean_tags(tag_str)

        # Backstop: harvest a few nounish bits from caption if Florence tags were thin
        if len(cands) < max(3, top_k // 2):
            extras = []
            # very light noun-ish extraction from caption
            for piece in re.split(r"[,:;—\-]+", cap.lower()):
                t = "".join(ch for ch in piece if ch.isalnum() or ch == " ").strip()
                if t and 1 <= len(t.split()) <= 3 and t not in STOPWORDS:
                    extras.append(t)
            # de-dup
            for e in extras:
                if e not in cands:
                    cands.append(e)

        return self.labels_from_candidates(cands, top_k), cap

app = FastAPI(title="Florence Keyword Tagger (task tokens)")

print("[boot] init …")
RUNNER = FlorenceRunner(FLORENCE_MODEL_ID)
print("[boot] ready.")

class TagReq(BaseModel):
    s3_uri: str
    top_k: Optional[int] = None
    include_caption: Optional[bool] = False

@app.get("/health")
def health():
    import torch
    return {"ok": True, "backend": "florence-ft", "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.post("/tag")
def tag(req: TagReq):
    img = s3_image(req.s3_uri)
    top_k = req.top_k or TOP_K_DEFAULT
    labels, cap = RUNNER.tag(img, top_k)
    body = {"Labels": labels}
    if req.include_caption:
        body["Caption"] = cap
    return body
