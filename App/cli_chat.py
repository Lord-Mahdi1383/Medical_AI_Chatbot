import sys
import os
from transformers import AutoTokenizer, TextIteratorStreamer
from optimum.intel import OVModelForCausalLM
from threading import Thread

def main():
    print("\n" + "="*50)
    print(" MEDICAL AI TERMINAL CHAT")
    print(" Powered by OpenVINO INT8 (CPU Optimized)")
    print("="*50)

    # find and load model
    # Dynamically find the model path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.abspath(os.path.join(script_dir, "..")) 
    model_dir = os.path.join(project_root, "model", "phi3_doctor_openvino_int8")

    print("\n[System] Loading Medical AI into RAM...")
    
    if not os.path.exists(model_dir):
        print(f"\n[Error] Could not find the model folder at: {model_dir}")
        print("Please make sure you unzipped the model into the 'model' directory!")
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = OVModelForCausalLM.from_pretrained(model_dir)
        print("[System] Load Complete! AI is ready.\n")
    except Exception as e:
        print(f"\n[Error] Failed to load the model. Error details: {e}")
        sys.exit(1)

    print("--- Type 'quit' or 'exit' to end the conversation. ---\n")

    # chat loop
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("\n[System] Shutting down. Stay healthy! \n")
            break
            
        if not user_input.strip():
            continue

        prompt = f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt")

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs, 
            streamer=streamer, 
            max_new_tokens=150, 
            temperature=0.3,
            do_sample=True
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        print("AI: ", end="", flush=True)

        for new_text in streamer:
            print(new_text, end="", flush=True)
            
        print("\n" + "-"*50) 

if __name__ == "__main__":
    main()