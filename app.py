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

# ---------------- Florence 2 Image Captioner ---------------
class Florence2Captioner:
    def __init__(self):
        print("[boot] Loading Florence 2 Image Captioner...")
        
        try:
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[boot] Using device: {self.device}")
            
            # Load Florence 2 model for detailed descriptions
            success = self._load_florence2()
            
            if not success:
                print("[boot] Florence 2 failed, falling back to GIT...")
                success = self._load_git()
                if not success:
                    print("[boot] GIT failed, falling back to original BLIP...")
                    self._load_original_blip()
                    self.model_type = "blip_original"
                else:
                    self.model_type = "git"
            else:
                self.model_type = "florence2"
            
            print(f"[boot] {self.model_type.upper()} model loaded successfully!")
            
        except Exception as e:
            print(f"[boot] Error loading models: {e}")
            print("[boot] Falling back to original BLIP...")
            self._load_original_blip()
            self.model_type = "blip_original"
    
    def _load_florence2(self) -> bool:
        """Load Florence 2 model for detailed descriptions"""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch
            
            # Use Florence 2 model
            model_name = "microsoft/florence-2-base"
            
            print(f"[boot] Loading Florence 2 model: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            
            return True
            
        except Exception as e:
            print(f"[boot] Florence 2 loading failed: {e}")
            return False
    
    def _load_git(self) -> bool:
        """Load GIT model for detailed descriptions"""
        try:
            from transformers import GitProcessor, GitForCausalLM
            import torch
            
            # Try different GIT models in order of preference
            models_to_try = [
                "microsoft/git-large-coco",      # Original choice
                "microsoft/git-base-coco",       # Smaller alternative
                "microsoft/git-large-textcaps",  # Different training data
            ]
            
            for model_name in models_to_try:
                try:
                    print(f"[boot] Trying GIT model: {model_name}")
                    
                    self.processor = GitProcessor.from_pretrained(model_name)
                    self.model = GitForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    
                    if self.device == "cuda":
                        self.model = self.model.to(self.device)
                    
                    print(f"[boot] Successfully loaded GIT model: {model_name}")
                    return True
                    
                except Exception as e:
                    print(f"[boot] Failed to load {model_name}: {e}")
                    continue
            
            return False
            
        except Exception as e:
            print(f"[boot] GIT loading failed completely: {e}")
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
            if self.model_type == "florence2":
                return self._caption_florence2(pil_img, detailed)
            elif self.model_type == "git":
                return self._caption_git(pil_img, detailed)
            else:  # blip_original
                return self._caption_original_blip(pil_img, detailed)
            
        except Exception as e:
            print(f"[error] Caption generation failed: {e}")
            return "an image"
    
    def _caption_florence2(self, pil_img: Image.Image, detailed: bool) -> str:
        """Caption using Florence 2 model for detailed descriptions"""
        import torch
        
        # Florence 2 uses specific prompts for different tasks
        if detailed:
            prompt = "<DETAILED_CAPTION>"
        else:
            prompt = "<CAPTION>"
        
        inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=200 if detailed else 100,  # Longer for detailed descriptions
                num_beams=5,     # Good quality
                temperature=0.7, # Balanced creativity
                do_sample=True,  # Allow sampling
                early_stopping=True,
                repetition_penalty=1.2,
                length_penalty=1.1,
            )
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up the response - remove the prompt
        if prompt in caption:
            caption = caption.replace(prompt, "").strip()
        
        if not caption or caption.lower() in ["", "image", "photo", "picture"]:
            return "an image"
        
        return caption
    
    def _caption_git(self, pil_img: Image.Image, detailed: bool) -> str:
        """Caption using GIT model for detailed descriptions"""
        import torch
        
        # GIT uses image inputs directly
        inputs = self.processor(images=pil_img, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=200 if detailed else 120,  # Much longer for detailed descriptions
                num_beams=8,     # Higher quality beam search
                temperature=0.8, # More creative
                do_sample=True,  # Allow sampling
                early_stopping=True,
                repetition_penalty=1.3,  # Higher penalty for repetition
                length_penalty=1.3,      # Encourage longer descriptions
                no_repeat_ngram_size=2,   # Avoid repetitive phrases
                min_length=20 if detailed else 10,  # Minimum length
            )
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        caption = caption.strip()
        
        if not caption or caption.lower() in ["", "image", "photo", "picture"]:
            return "an image"
        
        return caption
    
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
                do_sample=False,
                early_stopping=True,
                repetition_penalty=1.1,
            )
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        caption = caption.strip()
        
        if not caption or caption.lower() in ["", "image", "photo", "picture"]:
            return "an image"
        
        return caption

# ---------------- FastAPI App ---------------
app = FastAPI(title="Image Caption Service")

print("[boot] Initializing...")
# Use Florence 2 model for detailed descriptions
CAPTIONER = Florence2Captioner()
print("[boot] Ready!")

class CaptionRequest(BaseModel):
    s3_uri: str
    detailed: bool = True  # Default to detailed descriptions
    custom_prompt: Optional[str] = None  # Custom prompt for InstructBLIP

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "service": "florence2-captioner",
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