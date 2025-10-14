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
FLORENCE_MODEL_ID = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-large-ft")
SIGLIP_MODEL_ID = os.getenv("SIGLIP_MODEL_ID", "google/siglip-so400m-patch14-384")
ENABLE_OWL = os.getenv("ENABLE_OWL", "false").lower() in {"1", "true", "yes"}

# Florence task tokens (caption only)
TASK_CAPTION = "<DETAILED_CAPTION>"  # or "<CAPTION>" for shorter

STOPWORDS = {
    "a","an","the","and","or","of","in","on","with","without","to","for","by","at","from",
    "that","this","me","photo","image","picture","set","answering","unanswerable","text",
    "no extra text","n/a","none","tag","tags","we can see","in this image"
}

# Common art/media terms we DO allow as tags if they show up
KEEP_STYLE = {
    "woodcut","engraving","etching","lithograph","collage","ink drawing","cyanotype",
    "screen print","printmaking","oil painting","watercolor","charcoal","pencil drawing"
}

# -------------------------------
# S3 fetcher
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
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[boot] loading captioner: {model_id} on {self.device} …")

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    def _generate(self, task_token: str, pil_img: Image.Image, max_new_tokens=96, num_beams=4) -> str:
        inputs = self.processor(text=task_token, images=pil_img, return_tensors="pt")
        dtype = getattr(self.model, "dtype", None)

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
                do_sample=False,
                num_beams=num_beams,
                length_penalty=1.05,
                no_repeat_ngram_size=3,
            )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def caption(self, pil_img: Image.Image) -> str:
        cap = self._generate(TASK_CAPTION, pil_img, max_new_tokens=128, num_beams=5)
        if not cap or cap.lower() in {"no","n/a"} or "<" in cap or "tag" in cap.lower():
            cap = self._generate("<CAPTION>", pil_img, max_new_tokens=96, num_beams=4)
        # tighten to a single tidy sentence if model rambles
        cap = cap.replace("\n", " ").strip()
        cap = re.sub(r"\s+", " ", cap)
        # drop the "in this image we can see" boilerplate if present
        cap = re.sub(r"(?i)\b(in this image|we can see|this is)\b[^.]*\.?\s*", "", cap).strip()
        return cap or "image"

# -------------------------------
# SigLIP scorer (image-text cosine)
# -------------------------------
class SiglipScorer:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoProcessor, AutoModel
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[boot] loading SigLIP: {model_id} on {self.device} …")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        # discover the model's parameter dtype (fp16 on GPU)
        try:
            self.model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self.model_dtype = torch.float16 if self.device == "cuda" else torch.float32

    def _to_device_match_dtype(self, batch):
        """Move to device; cast only floating tensors to the model's dtype."""
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

    def image_emb(self, pil_img: Image.Image):
        with self.torch.no_grad():
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = self._to_device_match_dtype(inputs)
            out = self.model.get_image_features(**inputs)
            emb = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
            return emb  # [1, d]

    def text_emb(self, texts: List[str]):
        with self.torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = self._to_device_match_dtype(inputs)  # casts only floating tensors
            out = self.model.get_text_features(**inputs)
            emb = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
            return emb  # [n, d]

    def rank(self, pil_img: Image.Image, candidates: List[str]) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        img_e = self.image_emb(pil_img)
        txt_e = self.text_emb(candidates)
        sims = (img_e @ txt_e.T).squeeze(0)      # cosine since both are L2-normalized
        vals = sims.detach().float().cpu().tolist()
        paired = list(zip(candidates, vals))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired

# -------------------------------
# Candidate extraction
# -------------------------------
def candidate_tags_from_caption(caption: str) -> List[str]:
    """
    Very light noun-ish extraction from a caption to ensure we never return empty tags.
    """
    cap = caption.lower()
    # split on commas/colons/dashes/periods
    parts = re.split(r"[,\.:;—\-]+", cap)
    cands = []
    for p in parts:
        t = re.sub(r"[^\w\s\-&]", "", p).strip()
        if not t:
            continue
        # keep short noun-y phrases (1..4 tokens)
        toks = t.split()
        if 1 <= len(toks) <= 4:
            # drop boilerplate
            if any(sw in t for sw in STOPWORDS):
                continue
            cands.append(t)
    # allow style/media keeps
    for k in KEEP_STYLE:
        if k in cap and k not in cands:
            cands.append(k)
    # normalize/pad some common useful tokens
    cap_norm = " " + cap + " "
    if "black and white" in cap_norm and "black and white" not in cands:
        cands.append("black and white")
    return dedupe_keep_order([normalize_tag(x) for x in cands])

def normalize_tag(t: str) -> str:
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^(a|an|the)\s+", "", t)
    # drop too generic single words
    if t in {"image","photo","picture","painting","art","graphic","illustration","scene","view"}:
        return ""
    return t

def dedupe_keep_order(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def sanitize_topk(cands: List[str], top_k: int) -> List[str]:
    out = []
    for c in cands:
        c2 = re.sub(r"[^\w\s\-&]", "", c).strip()
        if not c2 or c2 in STOPWORDS:
            continue
        # keep only short phrases
        if 1 <= len(c2.split()) <= 4:
            out.append(c2)
        if len(out) >= top_k * 2:  # keep a small buffer before final cut
            break
    return dedupe_keep_order(out)

# -------------------------------
# Tagging orchestrator
# -------------------------------
class Tagger:
    def __init__(self, captioner: FlorenceCaptioner, scorer: SiglipScorer):
        self.captioner = captioner
        self.scorer = scorer

    def tag(self, pil_img: Image.Image, top_k: int) -> Tuple[List[dict], str]:
        # 1) caption
        cap = self.captioner.caption(pil_img)

        # 2) build candidate tags from caption
        cands = candidate_tags_from_caption(cap)

        # 3) (optional) OWL proposals could be added here and unioned
        # if ENABLE_OWL:
        #     cands = union_with_owl_proposals(pil_img, cands)

        # 4) sanitize and ensure we have something
        cands = sanitize_topk(cands, max(top_k * 3, 20))
        if not cands:
            cands = ["portrait","person","indoor","black and white"]  # last-resort fallbacks

        # 5) rank with SigLIP image-text similarity
        ranked = self.scorer.rank(pil_img, cands)

        # 6) final top_k
        top = [name for (name, _score) in ranked[:top_k]]

        # Rekognition-style output (you said you don't need confidences; we leave them in for compatibility)
        labels = [{"Name": t.title(), "Confidence": 100.0} for t in top]
        return labels, cap

# -------------------------------
# FastAPI
# -------------------------------
app = FastAPI(title="Are.na Image Caption+Tags (Florence+SigLIP)")

print("[boot] init …")
CAPTIONER = FlorenceCaptioner(FLORENCE_MODEL_ID)
SCORER = SiglipScorer(SIGLIP_MODEL_ID)
RUNNER = Tagger(CAPTIONER, SCORER)
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
        "backend": "florence+siglip",
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
