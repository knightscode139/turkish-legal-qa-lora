---
language: tr
license: apache-2.0
base_model: Trendyol/Trendyol-LLM-7b-chat-v0.1
tags:
- legal
- turkish
- lora
- peft
- question-answering
datasets:
- Renicames/turkish-law-chatbot
metrics:
- perplexity
model-index:
- name: trendyol-llm-7b-turkish-legal-lora
  results:
  - task:
      type: text-generation
    dataset:
      type: Renicames/turkish-law-chatbot
      name: Turkish Law Chatbot
    metrics:
    - type: perplexity
      value: 2.19
    - type: loss
      value: 0.7859
---

# Turkish Legal QA - LoRA Fine-tuned Model

LoRA adapters for Trendyol LLM 7B fine-tuned on Turkish legal question-answering dataset.

## Model Details

- **Base Model**: [Trendyol/Trendyol-LLM-7b-chat-v0.1](https://huggingface.co/Trendyol/Trendyol-LLM-7b-chat-v0.1)
- **Dataset**: [turkish-law-chatbot](https://huggingface.co/datasets/Renicames/turkish-law-chatbot) (13,354 training examples)
- **Method**: QLoRA (4-bit quantization + LoRA)
- **Adapter Size**: 33MB
- **Training Time**: ~4.7 hours on RTX 5060 Laptop (8GB VRAM)
- **Trainable Parameters**: 16.7M (0.24% of total)

## Performance

| Metric | Value |
|--------|-------|
| Test Loss | 0.7859 |
| Perplexity | 2.19 |
| Training Loss | 0.818 |

## Training Configuration

- **Epochs**: 2
- **Batch Size**: 2 (per device)
- **Gradient Accumulation**: 4 (effective batch size = 8)
- **Learning Rate**: 2e-4
- **LoRA Rank (r)**: 16
- **LoRA Alpha**: 32
- **Target Modules**: q_proj, v_proj, k_proj, o_proj
- **Optimizer**: paged_adamw_8bit

## Usage

### Installation

Clone the training repository and install dependencies:

```bash
git clone https://github.com/knightscode139/turkish-legal-qa-lora
cd turkish-legal-qa-lora
uv sync
```

### Load Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Trendyol/Trendyol-LLM-7b-chat-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "knightscode139/trendyol-llm-7b-turkish-legal-lora")
tokenizer = AutoTokenizer.from_pretrained("Trendyol/Trendyol-LLM-7b-chat-v0.1")

# Generate response
messages = [{"role": "user", "content": "Trafik cezalarına itiraz süreci nasıl işler?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Example Output

**Question**: Trafik cezalarına itiraz süreci nasıl işler?

**Answer**: Trafik cezalarına itiraz süreci, cezanın tebliğinden itibaren 15 gün içinde sulh ceza hakimliğine başvurarak yapılabilir.

## Training Details

The model was fine-tuned using QLoRA technique which combines:
- **4-bit NF4 quantization** for memory efficiency
- **LoRA adapters** for parameter-efficient fine-tuning
- **Gradient checkpointing** to reduce memory usage

This allows fine-tuning a 7B parameter model on consumer hardware (8GB VRAM).

## Limitations

- Model is specialized for Turkish legal questions
- May not perform well on general conversation
- Responses are based on training data and should be verified
- Not a replacement for professional legal advice

## Citation

```bibtex
@misc{trendyol-turkish-legal-lora,
  author = {knightscode139},
  title = {Turkish Legal QA LoRA Fine-tuned Model},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/knightscode139/trendyol-llm-7b-turkish-legal-lora}
}
```

## License

This model is released under Apache 2.0 license, consistent with the base model [Trendyol/Trendyol-LLM-7b-chat-v0.1](https://huggingface.co/Trendyol/Trendyol-LLM-7b-chat-v0.1).
