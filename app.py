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

# ---------------- Simple Image Analyzer (no complex model loading) ---------------
class SimpleImageAnalyzer:
    def __init__(self):
        print("[boot] initializing simple image analyzer...")
        print("[boot] ready - no complex models to load!")

    def caption(self, pil_img: Image.Image) -> str:
        """Generate a simple caption based on image properties"""
        width, height = pil_img.size
        
        # Basic image analysis
        aspect_ratio = width / height
        total_pixels = width * height
        
        # Determine orientation
        if aspect_ratio > 1.5:
            orientation = "wide landscape"
        elif aspect_ratio < 0.67:
            orientation = "tall portrait"
        elif aspect_ratio > 1.1:
            orientation = "landscape"
        elif aspect_ratio < 0.9:
            orientation = "portrait"
        else:
            orientation = "square"
        
        # Determine resolution
        if total_pixels > 2000000:  # > 2MP
            resolution = "high resolution"
        elif total_pixels > 500000:  # > 0.5MP
            resolution = "medium resolution"
        else:
            resolution = "low resolution"
        
        # Analyze dominant colors
        colors = self._get_dominant_colors(pil_img)
        
        # Create caption
        caption_parts = [f"{orientation} image", resolution]
        if colors:
            caption_parts.append(f"with {colors[0]} tones")
        
        return " ".join(caption_parts)

    def extract_tags(self, pil_img: Image.Image, top_k: int = 10) -> List[str]:
        """Extract tags based on image analysis"""
        width, height = pil_img.size
        tags = []
        
        # Orientation tags
        aspect_ratio = width / height
        if aspect_ratio > 1.5:
            tags.extend(["landscape", "wide", "panoramic"])
        elif aspect_ratio < 0.67:
            tags.extend(["portrait", "tall", "vertical"])
        elif aspect_ratio > 1.1:
            tags.extend(["landscape", "horizontal"])
        elif aspect_ratio < 0.9:
            tags.extend(["portrait", "vertical"])
        else:
            tags.extend(["square", "balanced"])
        
        # Resolution tags
        total_pixels = width * height
        if total_pixels > 2000000:
            tags.extend(["high resolution", "detailed", "sharp"])
        elif total_pixels > 500000:
            tags.extend(["medium resolution", "clear"])
        else:
            tags.extend(["low resolution", "small"])
        
        # Color analysis
        colors = self._get_dominant_colors(pil_img)
        if colors:
            tags.extend(colors[:3])  # Top 3 colors
        
        # Size categories
        if width > 2000 or height > 2000:
            tags.append("large")
        elif width < 500 or height < 500:
            tags.append("small")
        else:
            tags.append("medium")
        
        # Remove duplicates and limit
        unique_tags = []
        for tag in tags:
            if tag not in unique_tags and tag not in STOPWORDS:
                unique_tags.append(tag)
        
        return unique_tags[:top_k] if unique_tags else ["image", "photo"]

    def _get_dominant_colors(self, pil_img: Image.Image) -> List[str]:
        """Get dominant colors from the image"""
        try:
            # Resize for faster processing
            small_img = pil_img.resize((150, 150))
            
            # Convert to RGB if needed
            if small_img.mode != 'RGB':
                small_img = small_img.convert('RGB')
            
            # Get color data
            colors = small_img.getcolors(maxcolors=256*256*256)
            if not colors:
                return []
            
            # Sort by frequency
            colors.sort(key=lambda x: x[0], reverse=True)
            
            # Convert RGB to color names
            color_names = []
            for count, (r, g, b) in colors[:5]:  # Top 5 colors
                color_name = self._rgb_to_color_name(r, g, b)
                if color_name and color_name not in color_names:
                    color_names.append(color_name)
            
            return color_names
        except Exception:
            return []

    def _rgb_to_color_name(self, r: int, g: int, b: int) -> str:
        """Convert RGB values to color names"""
        # Simple color classification
        if r > 200 and g > 200 and b > 200:
            return "bright"
        elif r < 50 and g < 50 and b < 50:
            return "dark"
        elif r > g and r > b and r - max(g, b) > 50:
            return "red"
        elif g > r and g > b and g - max(r, b) > 50:
            return "green"
        elif b > r and b > g and b - max(r, g) > 50:
            return "blue"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r > 150 and g < 100 and b > 150:
            return "purple"
        elif r < 100 and g > 150 and b > 150:
            return "cyan"
        elif r > 150 and g > 100 and g < 150 and b < 100:
            return "orange"
        elif r > 100 and g > 100 and b > 100:
            return "light"
        else:
            return "neutral"

# ---------------- Orchestrator -----------------
class Tagger:
    def __init__(self, analyzer: SimpleImageAnalyzer):
        self.analyzer = analyzer

    def tag(self, pil_img: Image.Image, top_k: int) -> Tuple[List[dict], str]:
        # Get tags and caption from simple analyzer
        tags = self.analyzer.extract_tags(pil_img, top_k)
        caption = self.analyzer.caption(pil_img)
        
        # Format tags as expected by the API
        labels = [{"Name": tag.title(), "Confidence": 100.0} for tag in tags]
        
        return labels, caption

# ---------------- FastAPI -----------------
app = FastAPI(title="Are.na Tagger (Simple Image Analysis)")

print("[boot] init …")
ANALYZER = SimpleImageAnalyzer()
RUNNER = Tagger(ANALYZER)
print("[boot] ready.")

class TagReq(BaseModel):
    s3_uri: str
    top_k: Optional[int] = None
    include_caption: Optional[bool] = False

@app.get("/health")
def health():
    return {"ok": True, "backend": "simple-image-analyzer", "device": "cpu"}

@app.post("/tag")
def tag(req: TagReq):
    img = s3_image(req.s3_uri)
    top_k = req.top_k or TOP_K_DEFAULT
    labels, cap = RUNNER.tag(img, top_k)
    body = {"Labels": labels}
    if req.include_caption:
        body["Caption"] = cap
    return body
