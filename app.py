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

# ---------------- Advanced Image Captioner ---------------
class AdvancedCaptioner:
    def __init__(self, model_type: str = "blip2"):
        print(f"[boot] Loading Advanced Image Captioner ({model_type})...")
        
        try:
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[boot] Using device: {self.device}")
            
            if model_type == "blip2":
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                # Use BLIP2 for much better detailed captions
                model_name = "Salesforce/blip2-opt-2.7b"
                
                self.processor = Blip2Processor.from_pretrained(model_name)
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
            elif model_type == "instructblip":
                from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
                # Use InstructBLIP for prompt-based detailed descriptions
                model_name = "Salesforce/instructblip-vicuna-7b"
                
                self.processor = InstructBlipProcessor.from_pretrained(model_name)
                self.model = InstructBlipForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            
            self.model_type = model_type
            print(f"[boot] {model_type.upper()} model loaded successfully!")
            
        except Exception as e:
            print(f"[boot] Error loading {model_type}: {e}")
            raise e

    def caption(self, pil_img: Image.Image, detailed: bool = True, custom_prompt: str = None) -> str:
        """Generate a detailed caption for the image"""
        try:
            import torch
            
            # Use different prompting strategies based on model type
            if self.model_type == "instructblip":
                if custom_prompt:
                    prompt = custom_prompt
                elif detailed:
                    prompt = "Describe this image in detail, including the setting, people, objects, colors, activities, and atmosphere."
                else:
                    prompt = "What is in this image?"
            else:  # blip2
                if detailed:
                    prompt = "Describe this image in detail, including the setting, people, objects, colors, and activities."
                else:
                    prompt = "A photo of"
            
            inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate caption with optimized parameters for detailed descriptions
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=150 if detailed else 80,  # Longer for detailed descriptions
                    num_beams=4,     # Good balance of quality and speed
                    temperature=0.7, # Slightly more creative
                    do_sample=True,  # Allow sampling for variety
                    early_stopping=True,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                )
            
            # Decode caption
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up caption - remove the prompt if it was added
            caption = caption.strip()
            if prompt in caption:
                caption = caption.replace(prompt, "").strip()
            
            if not caption or caption.lower() in ["", "image", "photo", "picture"]:
                return "an image"
            
            return caption
            
        except Exception as e:
            print(f"[error] Caption generation failed: {e}")
            return "an image"

# ---------------- FastAPI App ---------------
app = FastAPI(title="Image Caption Service")

print("[boot] Initializing...")
# Choose model type: "blip2" or "instructblip"
MODEL_TYPE = os.getenv("MODEL_TYPE", "blip2")
CAPTIONER = AdvancedCaptioner(model_type=MODEL_TYPE)
print("[boot] Ready!")

class CaptionRequest(BaseModel):
    s3_uri: str
    detailed: bool = True  # Default to detailed descriptions
    custom_prompt: Optional[str] = None  # Custom prompt for InstructBLIP

@app.get("/health")
def health():
    return {"status": "ok", "service": f"advanced-{MODEL_TYPE}-captioner"}

@app.post("/caption")
def caption_image(req: CaptionRequest):
    """Generate a caption for an image from S3"""
    try:
        # Download image from S3
        img = s3_image(req.s3_uri)
        
        # Generate caption
        caption = CAPTIONER.caption(img, detailed=req.detailed, custom_prompt=req.custom_prompt)
        
        return {
            "caption": caption,
            "s3_uri": req.s3_uri
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoint for compatibility
@app.post("/tag")
def tag_image(req: CaptionRequest):
    """Legacy endpoint that returns caption only"""
    try:
        img = s3_image(req.s3_uri)
        caption = CAPTIONER.caption(img, detailed=req.detailed, custom_prompt=req.custom_prompt)
        
        return {
            "caption": caption,
            "s3_uri": req.s3_uri
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))