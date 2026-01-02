from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

app = FastAPI(title="Turkish Legal QA API")

# Global variables
tokenizer = None
model = None


@app.on_event("startup")
def load_model():
    global tokenizer, model
    print("Loading model...")

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
    model = PeftModel.from_pretrained(
        base_model,
        "knightscode139/trendyol-llm-7b-turkish-legal-lora"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Trendyol/Trendyol-LLM-7b-chat-v0.1")

    print("Model loaded successfully!")


class Question(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Turkish Legal QA API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(question: Question):
    # Prepare message in chat template format
    messages = [{"role": "user", "content": question.text}]
    text = tokenizer.apply_chat_template(
           messages,
           tokenize=False,
           add_generation_prompt=True)

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024)
        # Decode only the newly generated tokens (remove the input prompt)
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
    
    return {
        "question": question.text,
        "answer": response,
        "model": "knightscode139/trendyol-llm-7b-turkish-legal-lora"
    }
