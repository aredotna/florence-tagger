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

# Simple image analysis without complex model loading

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

# ---------------- BLIP-2 Image Tagger (dynamic, no hardcoded lists) ---------------
class BLIPTagger:
    def __init__(self):
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[boot] loading BLIP-2 on {self.device}...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        print("[boot] BLIP-2 loaded successfully!")

    def caption(self, pil_img: Image.Image) -> str:
        """Generate a detailed caption using BLIP-2"""
        try:
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            
            with self.torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    temperature=0.7,
                    do_sample=True,
                )
            
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return caption or "an image"
            
        except Exception as e:
            print(f"[error] BLIP caption generation failed: {e}")
            return "an image"

    def extract_tags(self, pil_img: Image.Image, top_k: int = 10) -> List[str]:
        """Extract meaningful tags using BLIP-2 with dynamic prompts"""
        try:
            # Generate multiple captions with different prompts to get diverse tags
            prompts = [
                "Describe this image in detail:",
                "What objects are in this image?",
                "What is happening in this image?",
                "List the main subjects in this image:",
                "What can you see in this image?"
            ]
            
            all_tags = []
            
            for prompt in prompts:
                try:
                    inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt").to(self.device)
                    
                    with self.torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_length=30,
                            num_beams=3,
                            temperature=0.8,
                            do_sample=True,
                        )
                    
                    response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    
                    # Extract meaningful words from the response
                    words = self._extract_meaningful_words(response)
                    all_tags.extend(words)
                    
                except Exception as e:
                    print(f"[error] BLIP prompt '{prompt}' failed: {e}")
                    continue
            
            # Remove duplicates and filter
            unique_tags = []
            for tag in all_tags:
                if tag not in unique_tags and tag not in STOPWORDS and len(tag) > 2:
                    unique_tags.append(tag)
            
            return unique_tags[:top_k] if unique_tags else ["image", "photo"]
            
        except Exception as e:
            print(f"[error] BLIP tag extraction failed: {e}")
            return ["image", "photo"]

    def _extract_meaningful_words(self, text: str) -> List[str]:
        """Extract meaningful words from BLIP response"""
        import re
        
        # Clean up the text
        text = text.lower().strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "describe this image in detail:",
            "what objects are in this image?",
            "what is happening in this image?",
            "list the main subjects in this image:",
            "what can you see in this image?",
            "this image shows",
            "this is a",
            "this is an",
            "the image shows",
            "the image contains",
            "there is a",
            "there is an",
            "there are",
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Split into words and clean them
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        # Filter out common words and keep meaningful ones
        meaningful_words = []
        for word in words:
            if (word not in STOPWORDS and 
                len(word) > 2 and 
                word.isalpha() and
                word not in ["image", "photo", "picture", "shows", "contains", "there"]):
                meaningful_words.append(word)
        
        return meaningful_words

# ---------------- Orchestrator -----------------
class Tagger:
    def __init__(self, blip_tagger: BLIPTagger):
        self.blip_tagger = blip_tagger

    def tag(self, pil_img: Image.Image, top_k: int) -> Tuple[List[dict], str]:
        # Get tags and caption from BLIP-2
        tags = self.blip_tagger.extract_tags(pil_img, top_k)
        caption = self.blip_tagger.caption(pil_img)
        
        # Format tags as expected by the API
        labels = [{"Name": tag.title(), "Confidence": 100.0} for tag in tags]
        
        return labels, caption

# ---------------- FastAPI -----------------
app = FastAPI(title="Are.na Tagger (BLIP-2)")

print("[boot] init …")
BLIP_TAGGER = BLIPTagger()
RUNNER = Tagger(BLIP_TAGGER)
print("[boot] ready.")

class TagReq(BaseModel):
    s3_uri: str
    top_k: Optional[int] = None
    include_caption: Optional[bool] = False

@app.get("/health")
def health():
    import torch
    return {"ok": True, "backend": "blip-2", "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.post("/tag")
def tag(req: TagReq):
    img = s3_image(req.s3_uri)
    top_k = req.top_k or TOP_K_DEFAULT
    labels, cap = RUNNER.tag(img, top_k)
    body = {"Labels": labels}
    if req.include_caption:
        body["Caption"] = cap
    return body
