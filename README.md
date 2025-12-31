# Turkish Legal QA - LoRA Fine-tuning

Fine-tuning Trendyol LLM 7B on Turkish legal question-answering dataset using QLoRA.

## Overview

This project fine-tunes a 7B parameter language model on ~15K Turkish legal Q&A pairs using parameter-efficient LoRA adapters.

## Model

- **Base Model**: [Trendyol/Trendyol-LLM-7b-chat-v0.1](https://huggingface.co/Trendyol/Trendyol-LLM-7b-chat-v0.1)
- **Fine-tuned Model**: [knightscode139/trendyol-llm-7b-turkish-legal-lora](https://huggingface.co/knightscode139/trendyol-llm-7b-turkish-legal-lora)
- **Dataset**: [Renicames/turkish-law-chatbot](https://huggingface.co/datasets/Renicames/turkish-law-chatbot) (13,354 train / 1,500 test)

## Results

| Metric | Value |
|--------|-------|
| Training Loss | 0.818 |
| Test Loss | 0.7859 |
| Perplexity | 2.19 |
| Model Size | 33MB (LoRA adapters only) |
| Training Time | ~4.7 hours (RTX 5060 Laptop 8GB) |

## Installation

```bash
git clone https://github.com/knightscode139/turkish-legal-qa-lora
cd turkish-legal-qa-lora
uv sync
```

## Training

Open and run the Jupyter notebook:

```bash
jupyter notebook turkish_legal_lora_finetuning.ipynb
```

The notebook includes:
- Model loading with 4-bit quantization (QLoRA)
- Dataset preparation and formatting
- LoRA adapter configuration
- Training with optimized memory settings
- Evaluation on test set

## Training Configuration

- **Method**: QLoRA (4-bit NF4 quantization + LoRA)
- **Trainable Parameters**: 16.7M (0.24% of total)
- **Epochs**: 2
- **Batch Size**: 2 (effective: 8 with gradient accumulation)
- **Learning Rate**: 2e-4
- **LoRA Rank (r)**: 16
- **LoRA Alpha**: 32
- **Target Modules**: q_proj, v_proj, k_proj, o_proj
- **Optimizer**: paged_adamw_8bit