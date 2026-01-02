"""
Upload LoRA adapters to Hugging Face Hub
"""
from huggingface_hub import HfApi, create_repo

# Configuration
username = "knightscode139"
model_name = "trendyol-llm-7b-turkish-legal-lora"
repo_name = f"{username}/{model_name}"
local_path = "./trendyol-turkish-law-lora-final"

print(f"Uploading model to: https://huggingface.co/{repo_name}")

# Create repository
try:
    create_repo(repo_name, repo_type="model", exist_ok=True)
    print("✓ Repository created/verified")
except Exception as e:
    print(f"Repository creation: {e}")

# Upload model files
api = HfApi()
try:
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_name,
        repo_type="model",
        commit_message="Add LoRA adapters for Turkish legal QA"
    )
    print(f"\n✓ Model uploaded successfully!")
    print(f"View at: https://huggingface.co/{repo_name}")
except Exception as e:
    print(f"Upload failed: {e}")
