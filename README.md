# Advanced Image Captioning Service with GPT-OSS-120B Enhancement

A self-hosted image captioning service that generates detailed descriptions similar to OpenAI's quality using state-of-the-art vision-language models, with optional GPT-OSS-120B enhancement for superior reasoning and detail.

## 🚀 Major Improvements

**Before (BLIP):** "there are two men fighting in a living room with a couch"

**After (BLIP2/InstructBLIP):** "Two young boys practice boxing in a spacious, modern living room; one is shirtless and reaching with a glove, while the other is in white shorts with a black belt, mid-throw. The room features neutral-colored furniture, a white orchid, and built-in shelves with decorative items."

## 🎯 Supported Models

### 1. Qwen2.5-VL-7B (Primary Vision Model)
- **Model:** `Qwen/Qwen2.5-VL-7B-Instruct`
- **Best for:** Professional image captioning with OpenAI-style quality
- **Quality:** Matches OpenAI's professional captioning standards
- **Memory:** ~7GB VRAM
- **Features:** Optimized for creative industry use cases

### 2. GPT-OSS-120B (Enhancement Model)
- **Model:** `openai/gpt-oss-120b`
- **Best for:** Enhancing captions with superior reasoning and professional details
- **Quality:** Adds brand names, proper nouns, and professional specificity
- **Memory:** ~80GB VRAM (requires H100/MI300X)
- **Features:** Configurable reasoning levels (low/medium/high)

## 🔧 Configuration

Configure the service via environment variables:

```bash
# Basic configuration
export AWS_REGION=us-east-1
export VLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
export VLM_LOAD_8BIT=false

# GPT-OSS-120B enhancement (optional)
export USE_GPT_OSS=true
export GPT_OSS_MODEL_ID=openai/gpt-oss-120b
export GPT_OSS_REASONING_LEVEL=medium  # low, medium, high
```

## 📡 API Usage

### Basic Caption Generation (Qwen2.5-VL only)

```bash
curl -X POST "http://localhost:8000/caption" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_uri": "s3://your-bucket/image.jpg",
    "detailed": true
  }'
```

### Enhanced Caption Generation (with GPT-OSS-120B)

```bash
curl -X POST "http://localhost:8000/caption" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_uri": "s3://your-bucket/image.jpg",
    "detailed": true,
    "use_gpt_oss": true
  }'
```

### Force GPT-OSS Enhancement (override global setting)

```bash
curl -X POST "http://localhost:8000/caption" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_uri": "s3://your-bucket/image.jpg",
    "use_gpt_oss": true
  }'
```

### Skip GPT-OSS Enhancement (use only Qwen2.5-VL)

```bash
curl -X POST "http://localhost:8000/caption" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_uri": "s3://your-bucket/image.jpg",
    "use_gpt_oss": false
  }'
```

### Response Format

**Without GPT-OSS enhancement:**
```json
{
  "caption": "Professional photographer using Canon EOS R5 camera with 24-70mm lens in modern studio with white backdrop and professional lighting setup.",
  "enhanced": false,
  "s3_uri": "s3://your-bucket/image.jpg"
}
```

**With GPT-OSS enhancement:**
```json
{
  "caption": "Professional photographer using Canon EOS R5 camera with RF 24-70mm f/2.8L IS USM lens in modern photography studio with white seamless backdrop, Profoto A1X flash units, and Elinchrom softbox lighting setup.",
  "original_caption": "Professional photographer using Canon EOS R5 camera with 24-70mm lens in modern studio with white backdrop and professional lighting setup.",
  "enhanced": true,
  "s3_uri": "s3://your-bucket/image.jpg"
}
```

## 🐳 Docker Deployment

### Basic Setup (Qwen2.5-VL only)
```bash
docker build -t florence-tagger .
docker run -p 8000:8000 \
  -e AWS_REGION=us-east-1 \
  -e VLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct \
  florence-tagger
```

### With GPT-OSS-120B Enhancement (requires 80GB+ GPU)
```bash
docker build -t florence-tagger .
docker run -p 8000:8000 \
  -e AWS_REGION=us-east-1 \
  -e VLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct \
  -e USE_GPT_OSS=true \
  -e GPT_OSS_MODEL_ID=openai/gpt-oss-120b \
  -e GPT_OSS_REASONING_LEVEL=medium \
  florence-tagger
```

### 8-bit Quantization (for lower VRAM)
```bash
docker run -p 8000:8000 \
  -e AWS_REGION=us-east-1 \
  -e VLM_LOAD_8BIT=true \
  -e USE_GPT_OSS=true \
  florence-tagger
```

## 💡 GPT-OSS Reasoning Levels

The GPT-OSS-120B model supports three reasoning levels for different use cases:

### Low Reasoning (Fast)
- **Use case:** Quick refinements for general use
- **Speed:** Fastest
- **Detail:** Basic professional improvements
- **Example:** "Canon camera" → "Canon EOS R5 camera"

### Medium Reasoning (Balanced)
- **Use case:** Professional captioning with brand specificity
- **Speed:** Moderate
- **Detail:** Adds specific model names, brands, proper nouns
- **Example:** "Professional camera setup" → "Canon EOS R5 with RF 24-70mm f/2.8L IS USM lens and Profoto lighting"

### High Reasoning (Detailed)
- **Use case:** Maximum detail for creative industry search
- **Speed:** Slower
- **Detail:** Comprehensive brand names, technical specifications, professional terminology
- **Example:** "Studio setup" → "Modern photography studio with Canon EOS R5, RF 24-70mm f/2.8L IS USM lens, white seamless backdrop, Profoto A1X flash units, Elinchrom softbox lighting, and Manfrotto tripod system"

## 🔄 Migration from Old Version

The API is backward compatible. Your existing code will work with improved descriptions:

```python
# Old code still works
response = requests.post("http://localhost:8000/caption", json={
    "s3_uri": "s3://bucket/image.jpg"
})

# New detailed descriptions automatically
print(response.json()["caption"])
```

## 📊 Performance Comparison

| Configuration | Description Quality | Speed | Memory | Professional Details |
|---------------|-------------------|-------|--------|---------------------|
| Qwen2.5-VL only | High | Fast | 7GB VRAM | ✅ |
| Qwen2.5-VL + GPT-OSS (low) | Very High | Medium | 87GB VRAM | ✅✅ |
| Qwen2.5-VL + GPT-OSS (medium) | Excellent | Slower | 87GB VRAM | ✅✅✅ |
| Qwen2.5-VL + GPT-OSS (high) | Outstanding | Slowest | 87GB VRAM | ✅✅✅✅ |

## 🛠️ Hardware Requirements

### Qwen2.5-VL Only
- **Minimum:** 8GB RAM, 8GB VRAM
- **Recommended:** 16GB RAM, 10GB+ VRAM
- **GPU:** NVIDIA GPU with CUDA support

### With GPT-OSS-120B Enhancement
- **Minimum:** 16GB RAM, 80GB VRAM
- **Recommended:** 32GB RAM, 80GB+ VRAM
- **GPU:** NVIDIA H100, AMD MI300X, or equivalent high-memory GPU

## 🔍 Health Check

```bash
curl http://localhost:8000/health
```

Returns:
```json
{
  "ok": true,
  "backend": "qwen2.5-vl-7b-instruct",
  "device": "cuda",
  "gpt_oss_enabled": true,
  "gpt_oss_model": "openai/gpt-oss-120b",
  "gpt_oss_reasoning_level": "medium"
}
```

## 🚀 Getting Started

1. **Clone and build:**
   ```bash
   git clone <repo>
   cd florence-tagger
   docker build -t florence-tagger .
   ```

2. **Run basic service (Qwen2.5-VL only):**
   ```bash
   docker run -p 8000:8000 \
     -e AWS_REGION=us-east-1 \
     florence-tagger
   ```

3. **Run with GPT-OSS enhancement (requires 80GB+ GPU):**
   ```bash
   docker run -p 8000:8000 \
     -e AWS_REGION=us-east-1 \
     -e USE_GPT_OSS=true \
     -e GPT_OSS_REASONING_LEVEL=medium \
     florence-tagger
   ```

4. **Test the service:**
   ```bash
   curl -X POST "http://localhost:8000/caption" \
     -H "Content-Type: application/json" \
     -d '{"s3_uri": "s3://your-bucket/test-image.jpg", "use_gpt_oss": true}'
   ```

## 🎯 Expected Results

With GPT-OSS-120B enhancement, you should see descriptions that are:
- **Professional-grade** with specific brand names and model numbers
- **Optimized for creative industry search** with proper nouns and technical details
- **Consistent with OpenAI's quality** in terms of specificity and accuracy
- **Configurable reasoning levels** for different speed/detail tradeoffs

**Example transformation:**
- **Basic:** "Professional photographer with camera in studio"
- **Enhanced:** "Professional photographer using Canon EOS R5 camera with RF 24-70mm f/2.8L IS USM lens in modern photography studio with white seamless backdrop, Profoto A1X flash units, and Elinchrom softbox lighting setup"
