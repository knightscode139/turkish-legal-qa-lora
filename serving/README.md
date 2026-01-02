# Turkish Legal QA API

FastAPI-based REST API for Turkish legal question answering using fine-tuned Trendyol-LLM-7b model with LoRA adapters.

## Model

- **Base Model**: `Trendyol/Trendyol-LLM-7b-chat-v0.1`
- **LoRA Adapters**: `knightscode139/trendyol-llm-7b-turkish-legal-lora`
- **Quantization**: 4-bit (NF4) for memory efficiency

## API Endpoints

### GET `/`
Root endpoint.

**Response:**
```json
{
  "message": "Turkish Legal QA API çalışıyor"
}
```

### GET `/health`
Health check endpoint for Kubernetes probes.

**Response:**
```json
{
  "status": "healthy"
}
```

### POST `/predict`
Generate answer for Turkish legal questions.

**Request:**
```json
{
  "text": "Trafik cezalarına itiraz süreci nasıl işler?"
}
```

**Response:**
```json
{
  "question": "Trafik cezalarına itiraz süreci nasıl işler?"
  "answer": "Generated legal explanation...",
  "model": "knightscode139/trendyol-llm-7b-turkish-legal-lora"
}
```

## Running Locally
```bash
# Install dependencies
uv sync

# Run the API
uv run uvicorn app:app --reload
```

API: `http://localhost:8000`

**Note:** First run downloads model from HuggingFace.

## Running with Docker
```bash
# Build
docker build -t turkish-legal-qa-api:v1 .

# Run (with GPU and cache mount)
docker run -d --gpus all -p 8080:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name legal-qa-api turkish-legal-qa-api:v1

# Logs
docker logs legal-qa-api -f

# Stop & Remove
docker stop legal-qa-api && docker rm legal-qa-api
```

## Testing
```bash
# Root
curl http://localhost:8000/

# Health
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Trafik cezalarına itiraz süreci nasıl işler?"}'
```
