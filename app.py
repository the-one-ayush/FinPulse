import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

st.set_page_config(page_title="FinPulse", page_icon="📈")

st.title("📈 FinPulse – Financial News Analyzer")
st.write("Analyze financial news and predict market impact.")

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_MODEL = "bitstormer/FinPulse"   # your HuggingFace adapter repo


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    model = PeftModel.from_pretrained(base_model, LORA_MODEL)

    return model, tokenizer


model, tokenizer = load_model()


news = st.text_area(
    "Enter financial news headline",
    placeholder="Example: Reliance announces expansion of renewable energy projects."
)

if st.button("Analyze") and news.strip():

    prompt = f"""
### Instruction:
Analyze financial news and predict market impact

### News:
{news}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with st.spinner("Analyzing..."):

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("Prediction")
    st.write(result)