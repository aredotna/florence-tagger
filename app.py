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
FLORENCE_MODEL_ID = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-large-ft")  # <- instruction-tuned

# tiny stopword set for tag cleanup
STOPWORDS = {
    "a","an","the","and","or","of","in","on","with","without","to","for","by","at",
    "from","that","this","me","photo","image","picture","set","answering","text"
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

class FlorenceRunner:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        import spacy

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[boot] loading {model_id} on {self.device} …")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)

        # CLIP (open-clip) for re-ranking
        import open_clip
        self.oclip_model, _, self.oclip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
        )
        self.oclip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

        print("[boot] loading spaCy en_core_web_sm …")
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

        # perf knobs (harmless if no-cuda)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # Prompts: give examples → ask
    @property
    def caption_prompt(self) -> str:
        return (
            "caption examples:\n"
            "a young boy leaning against a tree in a sunlit field\n"
            "a black-and-white portrait of a woman on a spiral staircase\n"
            "a stained glass window depicting a crowned figure holding a scepter\n"
            "Now write a caption for this image:"
        )

    @property
    def tags_prompt(self) -> str:
        return (
            "tag examples:\n"
            "boy, tree, vintage, sunlight, outdoors\n"
            "woman, staircase, spiral, monochrome, architecture\n"
            "stained glass, crown, scepter, medieval, cathedral\n"
            "Now write tags for this image (comma-separated, lowercase, no extra text):"
        )

    def _generate(self, text: str, pil_img: Image.Image, max_new_tokens=96, num_beams=4) -> str:
        inputs = self.processor(text=text, images=pil_img, return_tensors="pt")
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
            )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def generate_caption(self, pil_img: Image.Image) -> str:
        cap = self._generate(self.caption_prompt, pil_img, max_new_tokens=128, num_beams=5)
        # guard against instruction echoing
        if len(cap) < 4 or cap.lower() in {"no", "n/a"} or "caption examples" in cap.lower():
            cap = self._generate("write a descriptive photo caption:", pil_img, max_new_tokens=96, num_beams=4)
        return cap.strip()

    def generate_tag_text(self, pil_img: Image.Image) -> str:
        txt = self._generate(self.tags_prompt, pil_img, max_new_tokens=64, num_beams=4)
        # guard against prompt echo
        if "tag examples" in txt.lower() or len(txt) < 3:
            txt = self._generate("tags (comma-separated):", pil_img, max_new_tokens=48, num_beams=4)
        return txt.strip()

    # cleaning & backup from caption
    def clean_tags(self, tag_str: str) -> List[str]:
        raw = [t.strip().lower() for t in tag_str.split(",")]
        cleaned, seen = [], set()
        for t in raw:
            t = "".join(ch for ch in t if ch.isalnum() or ch == " ").strip()
            if not t or t in STOPWORDS or len(t) < 2: 
                continue
            if len(t.split()) <= 4 and t not in seen:
                seen.add(t); cleaned.append(t)
        return cleaned

    def caption_candidates(self, caption: str) -> List[str]:
        doc = self.nlp(caption)
        cand = []
        for chunk in doc.noun_chunks:
            t = "".join(ch for ch in chunk.text.lower().strip() if ch.isalnum() or ch == " ").strip()
            if t and t not in STOPWORDS and len(t.split()) <= 4:
                cand.append(t)
        # also split on punctuation to catch adjectives
        for piece in re.split(r"[,:;/\-]+", caption.lower()):
            t = "".join(ch for ch in piece if ch.isalnum() or ch == " ").strip()
            if t and t not in STOPWORDS and 1 <= len(t.split()) <= 3:
                cand.append(t)
        # dedupe preserving order
        seen, out = set(), []
        for w in cand:
            if w not in seen:
                seen.add(w); out.append(w)
        return out

    # CLIP re-ranker
    def rerank_with_clip(self, pil_img: Image.Image, candidates: List[str], top_k: int) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        import torch
        with self.torch.no_grad():
            img_t = self.oclip_preprocess(pil_img).unsqueeze(0).to(self.device)
            img_feat = self.oclip_model.encode_image(img_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            texts = [f"a photo of {t}" for t in candidates]
            text_tokens = self.oclip_tokenizer(texts).to(self.device)
            text_feat = self.oclip_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            sims = (img_feat @ text_feat.T).squeeze(0)  # cosine sim
            vals, idxs = sims.topk(min(top_k, len(candidates)))
            out = []
            for v, i in zip(vals.tolist(), idxs.tolist()):
                out.append((candidates[i], float(v)))
            return out

    def tag(self, pil_img: Image.Image, top_k: int) -> Tuple[List[dict], str]:
        tag_text = self.generate_tag_text(pil_img)
        caption = self.generate_caption(pil_img)

        candidates = self.clean_tags(tag_text)
        # backup if Florence tag string is weak
        if len(candidates) < max(3, top_k // 2):
            candidates = list({*candidates, *self.caption_candidates(caption)})

        ranked = self.rerank_with_clip(pil_img, candidates, top_k=top_k)
        if not ranked:
            # worst case: fall back to cleaned candidates in order
            ranked = [(w, 0.0) for w in candidates[:top_k]]

        labels = []
        # map CLIP cosine (~0.2..0.35+) to a 55..100 confidence band
        def score_to_conf(s: float) -> float:
            # linear map from [0.15, 0.4] -> [55, 100]
            lo, hi = 0.15, 0.40
            s = max(lo, min(hi, s))
            return 55.0 + (s - lo) * (45.0 / (hi - lo))

        for term, sim in ranked[:top_k]:
            labels.append({"Name": term.title(), "Confidence": round(score_to_conf(sim), 1)})

        return labels, caption

app = FastAPI(title="Florence Keyword Tagger (FT + CLIP re-rank)")

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
    return {"ok": True, "backend": "florence-ft+clip", "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.post("/tag")
def tag(req: TagReq):
    img = s3_image(req.s3_uri)
    top_k = req.top_k or TOP_K_DEFAULT
    labels, cap = RUNNER.tag(img, top_k)
    body = {"Labels": labels}
    if req.include_caption: body["Caption"] = cap
    return body
