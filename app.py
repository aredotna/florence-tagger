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

# GPT-OSS-120B configuration
USE_GPT_OSS = os.getenv("USE_GPT_OSS", "false").lower() in {"1","true","yes"}
GPT_OSS_MODEL_ID = os.getenv("GPT_OSS_MODEL_ID", "openai/gpt-oss-120b")
GPT_OSS_REASONING_LEVEL = os.getenv("GPT_OSS_REASONING_LEVEL", "medium")  # low, medium, high

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
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[boot] Loading Qwen2.5-VL model: {model_id}")
        print(f"[boot] Device: {self.device}")
        print(f"[boot] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[boot] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Load processor with fast processor disabled to avoid warnings
        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True,
            use_fast=False  # Disable fast processor to avoid warnings
        )

        # Configure quantization and loading
        quant_kwargs = {}
        if load_8bit and self.device == "cuda":
            try:
                quant_kwargs = {
                    "load_in_8bit": True, 
                    "device_map": "auto",
                    "low_cpu_mem_usage": True
                }
                print("[boot] Loading model in 8-bit with bitsandbytes")
            except Exception as e:
                print(f"[boot] 8-bit load failed: {e}; using fp16 instead")
                quant_kwargs = {"low_cpu_mem_usage": True}
        else:
            quant_kwargs = {"low_cpu_mem_usage": True}

        print("[boot] Starting model loading...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=(torch.float16 if self.device == "cuda" else torch.float32),
            **quant_kwargs
        )
        
        if not quant_kwargs.get("device_map"):
            print("[boot] Moving model to device...")
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Optimize CUDA settings
        if self.device == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                print("[boot] CUDA optimizations enabled")
            except Exception as e:
                print(f"[boot] CUDA optimization warning: {e}")
        
        print("[boot] Qwen2.5-VL model loaded successfully")

    # Professional OpenAI-style captioning prompt
    PROMPT = (
        "Audience: graphic designers, photographers, creative directors using text search. "
        "Task: Describe a single image in one stand-alone English sentence; ~30-50 words. "
        "Always include: Any clearly identifiable famous person, landmark, brand, artwork, product model (proper nouns, exact spelling). "
        "Stylistic or mood cues only if they are visually central. "
        "Always avoid: Guessing when uncertain; if identity is unclear, name the generic class rather than a brand. "
        "Camera metadata, hashtags, subjective opinions, filler, emojis."
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
    # enforce one sentence-ish: keep to ~50 words for professional captions
    words = t.split()
    if len(words) > 50:
        t = " ".join(words[:50]).rstrip(",;:") + "."
    if not t.endswith((".", "!", "?")):
        t += "."
    # Capitalize first letter
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t

# ---------- GPT-OSS-120B Caption Enhancer ----------
class GPTOSSCaptionEnhancer:
    """
    Uses GPT-OSS-120B to enhance image captions with better reasoning and detail.
    This is a text-only model that takes captions and improves them.
    """
    def __init__(self, model_id: str, reasoning_level: str = "medium"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reasoning_level = reasoning_level
        
        print(f"[boot] Loading GPT-OSS-120B model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Load model with appropriate settings for 80GB GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"[boot] GPT-OSS-120B loaded successfully on {self.device}")

    def enhance_caption(self, original_caption: str) -> str:
        """
        Enhance a caption using GPT-OSS-120B's reasoning capabilities.
        Uses standard transformers functionality without harmony package.
        """
        import torch
        
        # Create enhancement prompt based on reasoning level for professional captioning
        reasoning_prompts = {
            "low": "Refine this image caption for professional use by graphic designers and creative directors. Make it more precise and searchable:",
            "medium": "Enhance this image caption with specific brand names, proper nouns, and professional details suitable for creative industry text search:",
            "high": "Analyze and improve this image caption with professional reasoning. Add specific brand names, landmarks, artwork, product models, and proper nouns. Ensure it's optimized for creative industry search while maintaining accuracy:"
        }
        
        # Create a simple prompt without harmony format
        prompt = f"{reasoning_prompts[self.reasoning_level]}\n\nOriginal caption: {original_caption}\n\nEnhanced caption:"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate enhanced caption
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        # Decode response
        enhanced_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the enhanced caption part
        if "Enhanced caption:" in enhanced_text:
            enhanced_caption = enhanced_text.split("Enhanced caption:")[-1].strip()
        else:
            enhanced_caption = enhanced_text[len(prompt):].strip()
        
        # Clean up the response
        enhanced_caption = postprocess_caption(enhanced_caption)
        
        return enhanced_caption

# ---------- FastAPI ----------
app = FastAPI(title="High-Detail Image Captioner (Qwen2.5-VL-7B + GPT-OSS-120B)")

print("[boot] init …")

# Add memory management
import gc
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

try:
    CAPTIONER = QwenCaptioner(VLM_MODEL_ID, VLM_LOAD_8BIT)
    print("[boot] Qwen2.5-VL loaded successfully")
except Exception as e:
    print(f"[boot] Failed to load Qwen2.5-VL: {e}")
    print("[boot] This might be due to insufficient memory or disk space")
    raise e

# Initialize GPT-OSS enhancer if enabled
ENHANCER = None
if USE_GPT_OSS:
    try:
        print("[boot] Attempting to load GPT-OSS-120B...")
        ENHANCER = GPTOSSCaptionEnhancer(GPT_OSS_MODEL_ID, GPT_OSS_REASONING_LEVEL)
        print("[boot] GPT-OSS-120B enhancer ready.")
    except Exception as e:
        print(f"[boot] Failed to load GPT-OSS-120B: {e}")
        print("[boot] Continuing without GPT-OSS enhancement...")
        ENHANCER = None

print("[boot] ready.")

class CaptionRequest(BaseModel):
    s3_uri: str
    detailed: Optional[bool] = True  # kept for compatibility; prompt already emphasizes detail
    use_gpt_oss: Optional[bool] = None  # None = use global setting, True/False = override

@app.get("/health")
def health():
    import torch
    return {
        "ok": True,
        "backend": "qwen2.5-vl-7b-instruct",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpt_oss_enabled": USE_GPT_OSS and ENHANCER is not None,
        "gpt_oss_model": GPT_OSS_MODEL_ID if USE_GPT_OSS else None,
        "gpt_oss_reasoning_level": GPT_OSS_REASONING_LEVEL if USE_GPT_OSS else None
    }

@app.post("/caption")
def caption(req: CaptionRequest):
    try:
        img = s3_image(req.s3_uri)
        cap = CAPTIONER.caption(img)
        
        # Determine if we should use GPT-OSS enhancement
        should_enhance = False
        if req.use_gpt_oss is not None:
            should_enhance = req.use_gpt_oss
        else:
            should_enhance = USE_GPT_OSS
        
        # Enhance caption with GPT-OSS if requested and available
        enhanced_caption = cap
        enhancement_used = False
        if should_enhance and ENHANCER is not None:
            try:
                enhanced_caption = ENHANCER.enhance_caption(cap)
                enhancement_used = True
            except Exception as e:
                print(f"[warning] GPT-OSS enhancement failed: {e}")
                # Fall back to original caption
        
        return {
            "caption": enhanced_caption,
            "original_caption": cap if enhancement_used else None,
            "enhanced": enhancement_used,
            "s3_uri": req.s3_uri
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, str(e))
