import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL = "unsloth/phi3-mini-4k-instruct"
ADAPTERS_DIR = "./phi3_doctor_lora"
MERGED_DIR = "./phi3_doctor_merged"

def merge_adapters():
    print(f"loading base model: {BASE_MODEL}...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )

    tokenizer = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    print(f"applying LoRA adapters from: {ADAPTERS_DIR}...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTERS_DIR)

    print("merging weights...")
    merged_model = peft_model.merge_and_unload()

    print(f"saving merged model to: {MERGED_DIR}...")
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretraiend(MERGED_DIR)
    tokenizer.save_pretraiend(MERGED_DIR)

    print("model is merged and ready to use!")

if __name__ == "__main__":
    merge_adapters()
