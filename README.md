# Medical AI Chatbot: Fine-Tuned Phi-3

# Overview
A lightweight and accurate medical assistant powered by Microsoft's `Phi-3-mini-4k-instruct`.
This model was fine-tuned on curated medical Q&A pairs using LoRA adapters, and subsequently quantized to **INT8 using Intel OpenVINO**.
This allows the large language model to run fast on standard local CPUs without requiring an expensive GPU.

# Project Workflow & Repository Structure
This project was built in sequantial stages, documented in the `Notebooks/` directory:
1. **`01_EDA_and_Plots.ipynb`**: Exploratory Data Analysis of the HealthCareMagic dataset, analyzing token lengths and question distribution.
2. **`02_FineTuning_LoRA.ipynb`**: Utilized Unsloth to apply 4-bit quantozation and LoRA adapters to train the model efficiently on a free-tier Google Colab GPU.
3. **`03_OpenVINO_Quantization.ipynb`**: Merged the LoRA adapters and compressed the model into OpenVINO INT8 format for CPU infernece.
4. **`04_Testing_and_Evaluation.ipynb`**: Automated testing of model responses and plotting of CPu inference speeds (tokens/sec).

**Core Scripts:**
- `streamlit.py`: the Streamlit Web application.
- `cli_chat.py`: a lightweight command-line interface.
- `merge.py`: a helper script for developers to merge the raw LoRA weights into the base Phi-3 model.

# Why LoRA (Low-Rank Adaptation)?
Fine-Tuning a Large Language Model like Phi-3 (3.8 Billion parameters) traditionally requires massive compute clusters and hundreds of gigabytes of VRAM. to accomplish this on a single consumer-grade GPU (Google Colab Free Tier), I utilized **LoRA** combined with **4-bit quantozation (QLoRA)** via the Unsloth library.

LoRA freezes the base model and injects tiny, trainable "rank-decomposition matrices" into the transformer layers (Targeting: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

The resulting LoRA adapter weights are incredibly small (106MB), making them easy to store, share and swap, rather than saving a massive 8GB model for every iteration.

By keeping the base weights forzen, the model retains the incredible logic and reasoning capabilities Microsoft originally trained into Phi-3 while the LoRA adapters effectively steer its behavior and vocabulary toward the medical domain.


## How To Run

### 1. CLone the Repository
```bash
git clone https://github.com/Lord-Mahdi1383/Medical_AI_Chatbot.git
cd Medical_AI_Chatbot
```

### 2. Install Dependencies
```bash
pip install -r Requirements.txt
```

### 3. Download the Model Weights
- For Standard Users (recommended):
donwload the compiled OpenVINO INT8 model [https://drive.google.com/uc?export=download&id=1vfqIgmZtFaU55coy7OE3aQw3PpVx79-6]
and extract it into a folder named model
  
- for Developers (LoRA Adapters):
If you wish to merge the weights yourself download the adapter folder from [https://drive.google.com/uc?export=download&id=1Srg7xUM9GFaCtg8JPbonX9Ilq6Awoio7] and unzip it in a folder named model then run:
```bash
python merge.py
```


### 4. Launch the App!
**Option A:** Streamlit Web Interface:
```bash
streamlit run streamlit.py
```

**Option B:** Terminal CLI Interface:
```bash
python cli_chat.py
```


# Medical Disclaimer
This AI model is an experimental research project and is for educational purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment. Always seel the advice of a qualified physician or health provider
