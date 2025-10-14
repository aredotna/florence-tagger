import os
import io
from typing import Optional
from PIL import Image, ImageFile
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Enable truncated image loading
ImageFile.LOAD_TRUNCATED_IMAGES = True

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize S3 client
s3 = boto3.client('s3', region_name=AWS_REGION)

def s3_image(s3_uri: str) -> Image.Image:
    """Download image from S3 and return PIL Image"""
    try:
        bucket, key = s3_uri.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

# ---------------- Simple BLIP Captioner ---------------
class BLIPCaptioner:
    def __init__(self):
        print("[boot] Loading BLIP model...")
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[boot] Using device: {self.device}")
            
            # Use the base BLIP model (smaller, more reliable)
            model_name = "Salesforce/blip-image-captioning-base"
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            
            print("[boot] BLIP model loaded successfully!")
            
        except Exception as e:
            print(f"[boot] Error loading BLIP: {e}")
            raise e

    def caption(self, pil_img: Image.Image) -> str:
        """Generate a detailed caption for the image"""
        try:
            import torch
            
            # Try multiple prompts to get a longer description
            prompts = [
                "Describe this image in detail:",
                "What do you see in this image?",
                "Describe the scene:"
            ]
            
            all_captions = []
            
            for prompt in prompts:
                try:
                    # Process image with text prompt
                    inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt")
                    
                    if self.device == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Generate caption
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_length=80,
                            num_beams=4,
                            temperature=0.7,
                            do_sample=True,
                            early_stopping=True,
                        )
                    
                    # Decode caption
                    caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    # Clean up caption
                    caption = caption.strip()
                    if caption and caption.lower() not in ["", "image", "photo"]:
                        # Remove the prompt from the response
                        for p in prompts:
                            caption = caption.replace(p.lower(), "").strip()
                        all_captions.append(caption)
                        
                except Exception as e:
                    print(f"[error] Prompt '{prompt}' failed: {e}")
                    continue
            
            # If we got multiple captions, combine them
            if all_captions:
                # Remove duplicates and combine
                unique_captions = []
                for cap in all_captions:
                    if cap not in unique_captions and len(cap) > 10:
                        unique_captions.append(cap)
                
                if unique_captions:
                    # Combine the best captions
                    combined = ". ".join(unique_captions[:2])  # Take up to 2 best captions
                    return combined
            
            # Fallback: try without prompt
            inputs = self.processor(images=pil_img, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=60,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                )
            
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            caption = caption.strip()
            
            if not caption or caption.lower() in ["", "image", "photo"]:
                return "an image"
            
            return caption
            
        except Exception as e:
            print(f"[error] Caption generation failed: {e}")
            return "an image"

# ---------------- FastAPI App ---------------
app = FastAPI(title="Image Caption Service")

print("[boot] Initializing...")
CAPTIONER = BLIPCaptioner()
print("[boot] Ready!")

class CaptionRequest(BaseModel):
    s3_uri: str

@app.get("/health")
def health():
    return {"status": "ok", "service": "image-caption"}

@app.post("/caption")
def caption_image(req: CaptionRequest):
    """Generate a caption for an image from S3"""
    try:
        # Download image from S3
        img = s3_image(req.s3_uri)
        
        # Generate caption
        caption = CAPTIONER.caption(img)
        
        return {
            "caption": caption,
            "s3_uri": req.s3_uri
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoint for compatibility
@app.post("/tag")
def tag_image(req: CaptionRequest):
    """Legacy endpoint that returns caption in old format"""
    try:
        img = s3_image(req.s3_uri)
        caption = CAPTIONER.caption(img)
        
        return {
            "Labels": [{"Name": "Caption", "Confidence": 100.0}],
            "Caption": caption
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))