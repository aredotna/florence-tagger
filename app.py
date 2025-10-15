import os, io, re
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageFile
import boto3

ImageFile.LOAD_TRUNCATED_IMAGES = True

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
VLM_LOAD_8BIT = os.getenv("VLM_LOAD_8BIT", "false").lower() in {"1","true","yes"}

# ---------- S3 Loader ----------
s3 = boto3.client("s3", region_name=AWS_REGION)

def s3_image(s3_uri: str) -> Image.Image:
    if not s3_uri.startswith("s3://"):
        raise HTTPException(400, "s3_uri must start with s3://")
    try:
        bucket, key = s3_uri[5:].split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch/decode image: {e}")

# ---------- Qwen2.5-VL Captioner ----------
class QwenCaptioner:
    """
    Instruction VLM with a prompt & decoding recipe tuned for detailed but concise captions.
    """
    def __init__(self, model_id: str, load_8bit: bool = False):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        quant_kwargs = {}
        if load_8bit and self.device == "cuda":
            try:
                quant_kwargs = {"load_in_8bit": True, "device_map": "auto"}
                print("[boot] loading model in 8-bit with bitsandbytes")
            except Exception as e:
                print(f"[boot] 8-bit load failed: {e}; using fp16/fp32 instead")
                quant_kwargs = {}

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=(torch.float16 if self.device == "cuda" else torch.float32),
            **quant_kwargs
        )
        if not quant_kwargs:
            self.model = self.model.to(self.device)
        self.model.eval()
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # Strong instruction to elicit detailed, specific, single-sentence captions
    PROMPT = (
        "You are a precise captioner. Describe the image in ONE sentence with concrete details: "
        "number of people, approximate age (child/teen/adult), notable clothing colors, actions, "
        "room type and 1–3 distinctive objects (e.g., orchid, built-in shelves). "
        "Avoid hedging like 'maybe'/'appears' and avoid moral/judgemental words. "
        "Keep it to ~30–45 words."
    )

    def caption(self, pil_img: Image.Image) -> str:
        """
        Decoding tuned for specificity:
        - num_beams for coverage
        - low temperature for precision
        - no_repeat_ngram_size to avoid loops
        """
        import torch
        proc = self.processor

        # Qwen expects chat-style messages with an image
        messages = [
            {"role": "system", "content": "You are a helpful visual caption assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": self.PROMPT},
                {"type": "image", "image": pil_img}
            ]}
        ]
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
        pixel_inputs = proc(images=pil_img, return_tensors="pt")

        # move to device / match dtypes
        dtype = next(self.model.parameters()).dtype
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        pixel_inputs = {k: (v.to(self.device, dtype=dtype) if v.dtype.is_floating_point else v.to(self.device))
                        for k, v in pixel_inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=64,
            num_beams=5,
            do_sample=False,
            length_penalty=1.1,
            no_repeat_ngram_size=3,
        )

        with torch.no_grad():
            out = self.model.generate(
                **inputs, **pixel_inputs, **gen_kwargs
            )

        text = proc.batch_decode(out, skip_special_tokens=True)[0]
        text = postprocess_caption(text)
        return text

def postprocess_caption(text: str) -> str:
    # Clean up chat prefix artifacts or stray quotes
    t = re.sub(r"^\s*(assistant:|assistant|\"|“|”)+\s*", "", text.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    # enforce one sentence-ish: keep to ~45 words
    words = t.split()
    if len(words) > 45:
        t = " ".join(words[:45]).rstrip(",;:") + "."
    if not t.endswith((".", "!", "?")):
        t += "."
    # Capitalize first letter
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t

# ---------- FastAPI ----------
app = FastAPI(title="High-Detail Image Captioner (Qwen2.5-VL-7B)")

print("[boot] init …")
CAPTIONER = QwenCaptioner(VLM_MODEL_ID, VLM_LOAD_8BIT)
print("[boot] ready.")

class CaptionRequest(BaseModel):
    s3_uri: str
    detailed: Optional[bool] = True  # kept for compatibility; prompt already emphasizes detail

@app.get("/health")
def health():
    import torch
    return {
        "ok": True,
        "backend": "qwen2.5-vl-7b-instruct",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/caption")
def caption(req: CaptionRequest):
    try:
        img = s3_image(req.s3_uri)
        cap = CAPTIONER.caption(img)
        return {"caption": cap, "s3_uri": req.s3_uri}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, str(e))
