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
            
            # Try to load the requested model, fallback to original BLIP if it fails
            success = False
            
            if model_type == "blip2":
                success = self._load_blip2()
            elif model_type == "instructblip":
                success = self._load_instructblip()
            
            # Fallback to original BLIP if advanced model fails
            if not success:
                print(f"[boot] {model_type} failed, falling back to original BLIP...")
                self._load_original_blip()
                self.model_type = "blip_original"
            else:
                self.model_type = model_type
            
            print(f"[boot] {self.model_type.upper()} model loaded successfully!")
            
        except Exception as e:
            print(f"[boot] Error loading models: {e}")
            print("[boot] Falling back to original BLIP...")
            self._load_original_blip()
            self.model_type = "blip_original"
    
    def _load_blip2(self) -> bool:
        """Load BLIP2 model with error handling"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            import torch
            
            # Try different models in order of preference - start with more stable ones
            models_to_try = [
                "Salesforce/blip2-flan-t5-large", # Smaller, more stable model first
                "Salesforce/blip2-flan-t5-xl",    # Alternative BLIP2 model
                "Salesforce/blip2-opt-2.7b",      # Original choice
                "Salesforce/blip2-flan-t5-xxl"    # Another alternative
            ]
            
            for model_name in models_to_try:
                try:
                    print(f"[boot] Trying BLIP2 model: {model_name}")
                    
                    # Clear any cached files that might be corrupted
                    import os
                    import shutil
                    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                    if os.path.exists(cache_dir):
                        # Try to clear corrupted cache
                        try:
                            for root, dirs, files in os.walk(cache_dir):
                                for d in dirs:
                                    if "blip2" in d.lower():
                                        cache_path = os.path.join(root, d)
                                        if os.path.exists(cache_path):
                                            shutil.rmtree(cache_path, ignore_errors=True)
                        except:
                            pass  # Ignore cache clearing errors
                    
                    self.processor = Blip2Processor.from_pretrained(
                        model_name,
                        force_download=False,
                        resume_download=True,
                        local_files_only=False
                    )
                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        force_download=False,
                        resume_download=True,
                        local_files_only=False
                    )
                    
                    if self.device == "cuda":
                        self.model = self.model.to(self.device)
                    
                    print(f"[boot] Successfully loaded BLIP2 model: {model_name}")
                    return True
                    
                except Exception as e:
                    print(f"[boot] Failed to load {model_name}: {e}")
                    continue
            
            return False
            
        except Exception as e:
            print(f"[boot] BLIP2 loading failed completely: {e}")
            return False
    
    def _load_instructblip(self) -> bool:
        """Load InstructBLIP model with error handling"""
        try:
            from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
            import torch
            
            model_name = "Salesforce/instructblip-vicuna-7b"
            
            print(f"[boot] Downloading InstructBLIP model: {model_name}")
            self.processor = InstructBlipProcessor.from_pretrained(
                model_name,
                force_download=False,
                resume_download=True
            )
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                force_download=False,
                resume_download=True
            )
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            
            return True
            
        except Exception as e:
            print(f"[boot] InstructBLIP loading failed: {e}")
            return False
    
    def _load_original_blip(self):
        """Load original BLIP model as fallback"""
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        
        model_name = "Salesforce/blip-image-captioning-large"
        
        print(f"[boot] Loading original BLIP model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        if self.device == "cuda":
            self.model = self.model.to(self.device)

    def caption(self, pil_img: Image.Image, detailed: bool = True, custom_prompt: str = None) -> str:
        """Generate a detailed caption for the image"""
        try:
            import torch
            
            # Handle different model types
            if self.model_type == "blip_original":
                return self._caption_original_blip(pil_img, detailed)
            elif self.model_type == "instructblip":
                return self._caption_instructblip(pil_img, detailed, custom_prompt)
            else:  # blip2
                return self._caption_blip2(pil_img, detailed)
            
        except Exception as e:
            print(f"[error] Caption generation failed: {e}")
            return "an image"
    
    def _caption_original_blip(self, pil_img: Image.Image, detailed: bool) -> str:
        """Caption using original BLIP model"""
        import torch
        
        # Original BLIP doesn't support text prompts, just image captioning
        inputs = self.processor(images=pil_img, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=60,
                num_beams=5,
                temperature=0.6,
                do_sample=False,
                early_stopping=True,
                repetition_penalty=1.1,
            )
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        caption = caption.strip()
        
        if not caption or caption.lower() in ["", "image", "photo", "picture"]:
            return "an image"
        
        return caption
    
    def _caption_blip2(self, pil_img: Image.Image, detailed: bool) -> str:
        """Caption using BLIP2 model"""
        import torch
        
        if detailed:
            prompt = "Describe this image in detail, including the setting, people, objects, colors, and activities."
        else:
            prompt = "A photo of"
        
        inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=120,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                early_stopping=True,
                repetition_penalty=1.2,
                length_penalty=1.0,
            )
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        caption = caption.strip()
        
        if prompt in caption:
            caption = caption.replace(prompt, "").strip()
        
        if not caption or caption.lower() in ["", "image", "photo", "picture"]:
            return "an image"
        
        return caption
    
    def _caption_instructblip(self, pil_img: Image.Image, detailed: bool, custom_prompt: str = None) -> str:
        """Caption using InstructBLIP model"""
        import torch
        
        if custom_prompt:
            prompt = custom_prompt
        elif detailed:
            prompt = "Describe this image in detail, including the setting, people, objects, colors, activities, and atmosphere."
        else:
            prompt = "What is in this image?"
        
        inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                early_stopping=True,
                repetition_penalty=1.2,
                length_penalty=1.0,
            )
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        caption = caption.strip()
        
        if prompt in caption:
            caption = caption.replace(prompt, "").strip()
        
        if not caption or caption.lower() in ["", "image", "photo", "picture"]:
            return "an image"
        
        return caption

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
    return {
        "status": "ok", 
        "service": f"advanced-{MODEL_TYPE}-captioner",
        "actual_model": CAPTIONER.model_type,
        "device": CAPTIONER.device
    }

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