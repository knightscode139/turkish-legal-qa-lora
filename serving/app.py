from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Turkish Legal QA API")

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
    # Şimdilik mock response - sonra gerçek modeli yükleyeceğiz
    return {
        "question": question.text,
        "answer": "Model henüz yüklenmedi - test endpoint",
        "model": "knightscode139/trendyol-llm-7b-turkish-legal-lora"
    }
