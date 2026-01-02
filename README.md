# Turkish Legal Question-Answering System

Production-ready Turkish legal Q&A system using fine-tuned 7B LLM with MLOps pipeline.

## Overview

- **Model**: Trendyol-LLM-7b fine-tuned with QLoRA on 14.9K Turkish legal Q&A pairs
- **Serving**: FastAPI REST API
- **Infrastructure**: Docker containerization

## Project Structure
```
turkish-legal-qa/
├── training/          # LoRA fine-tuning
├── serving/           # FastAPI API + Docker
└── README.md
```

See [training/README.md](training/README.md) and [serving/README.md](serving/README.md) for details.

## Status

- ✅ Fine-tuning (QLoRA, 4-bit quantization, 0.24% trainable params)
- ✅ FastAPI serving + Docker GPU support
- ⏳ Kubernetes deployment
- ⏳ Gradio UI

## Tech Stack

PyTorch • Transformers • PEFT • FastAPI • Docker • Kubernetes
