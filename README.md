# Professional Image Captioning Service

A streamlined image captioning service using Qwen2.5-VL-7B-Instruct for professional-grade descriptions optimized for creative industry use.

## 🎯 Features

- **Professional Captioning**: Uses OpenAI-style prompts optimized for graphic designers, photographers, and creative directors
- **High-Quality Output**: 30-50 word descriptions with specific brand names, proper nouns, and technical details
- **Memory Efficient**: 8-bit quantization enabled by default for lower VRAM usage
- **FastAPI Service**: Simple REST API for easy integration

## 🚀 Quick Start

### Build and Run
```bash
# Build the Docker image
docker build -t florence-tagger .

# Run the service
docker run -p 8000:8000 \
  -e AWS_REGION=us-east-1 \
  -e VLM_LOAD_8BIT=true \
  florence-tagger
```

### Test the Service
```bash
# Health check
curl http://localhost:8000/health

# Generate caption
curl -X POST "http://localhost:8000/caption" \
  -H "Content-Type: application/json" \
  -d '{"s3_uri": "s3://your-bucket/image.jpg"}'
```

## 📡 API Usage

### Generate Caption
```bash
curl -X POST "http://localhost:8000/caption" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_uri": "s3://your-bucket/image.jpg",
    "detailed": true
  }'
```

### Response Format
```json
{
  "caption": "Professional photographer using Canon EOS R5 camera with RF 24-70mm f/2.8L IS USM lens in modern photography studio with white seamless backdrop and Profoto lighting setup.",
  "s3_uri": "s3://your-bucket/image.jpg",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct"
}
```

## 🔧 Configuration

Environment variables:
- `AWS_REGION`: AWS region for S3 access (default: us-east-1)
- `VLM_MODEL_ID`: Vision model to use (default: Qwen/Qwen2.5-VL-7B-Instruct)
- `VLM_LOAD_8BIT`: Enable 8-bit quantization (default: true)

## 🛠️ Hardware Requirements

- **Minimum**: 8GB RAM, 8GB VRAM
- **Recommended**: 16GB RAM, 10GB+ VRAM
- **GPU**: NVIDIA GPU with CUDA support

## 📊 Performance

With 8-bit quantization:
- **Memory Usage**: ~7GB VRAM (vs ~14GB without quantization)
- **Speed**: Fast inference with minimal quality loss
- **Quality**: Professional-grade captions matching OpenAI standards

## 🎯 Expected Results

The service generates captions optimized for creative industry search:
- Specific brand names and model numbers
- Proper nouns and technical details
- Professional terminology
- 30-50 words as specified

**Example:**
- **Input**: Image of photographer in studio
- **Output**: "Professional photographer using Canon EOS R5 camera with RF 24-70mm f/2.8L IS USM lens in modern photography studio with white seamless backdrop, Profoto A1X flash units, and Elinchrom softbox lighting setup."

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
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "8bit_quantization": true
}
```

## 🚀 Production Deployment

For production use:
```bash
docker run -d -p 8000:8000 \
  --name florence-tagger \
  --restart unless-stopped \
  --memory=16g \
  --memory-swap=20g \
  -e AWS_REGION=us-east-1 \
  -e VLM_LOAD_8BIT=true \
  florence-tagger
```

## 📝 Notes

- The service uses your professional OpenAI-style prompt for optimal results
- 8-bit quantization is enabled by default to reduce memory usage
- All GPT-OSS complexity has been removed for simplicity and reliability
- Focus on Qwen2.5-VL-7B-Instruct for consistent, high-quality results