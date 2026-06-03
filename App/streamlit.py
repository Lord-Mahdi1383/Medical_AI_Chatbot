import streamlit as st
import os
from transformers import AutoTokenizer, TextIteratorStreamer
from optimum.intel import OVModelForCausalLM
from threading import Thread

# configuration
st.set_page_config(page_title="Medical AI Chatbot", page_icon="", layout="centered")

# model loading
@st.cache_resource
def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Gets the /app folder
    project_root = os.path.abspath(os.path.join(script_dir, "..")) # Goes up to root
    model_dir = os.path.join(project_root, "model", "phi3_doctor_openvino_int8")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = OVModelForCausalLM.from_pretrained(model_dir)
    return model, tokenizer

with st.spinner("Loading Medical AI into memory..."):
    model, tokenizer = load_model()

# ui layout
st.title("Medical AI Chatbot")
st.caption("Powered by Fine-Tuned Phi-3 & Intel OpenVINO INT8")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input and generation
if prompt := st.chat_input("Please describe your symptoms or ask a medical question..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    prompt_template = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt_template, return_tensors="pt")

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

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        for new_text in streamer:
            full_response += new_text
            response_container.markdown(full_response + "▌") 
            
        response_container.markdown(full_response) 
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})