from unsloth import FastLanguageModel

finpulse_model, finpulse_tokenizer = FastLanguageModel.from_pretrained(
    "financial_lora",
    max_seq_length=512,
    load_in_4bit=True
)

base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct",
    load_in_4bit=True
)

FastLanguageModel.for_inference(finpulse_model)
FastLanguageModel.for_inference(base_model)


def generate_prediction(model, tokenizer, news):

    prompt = f"""
### Instruction:
Analyze financial news and predict market impact

### News:
{news}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def analyze_news_using_FinPulse(news):
    return generate_prediction(finpulse_model, finpulse_tokenizer, news)


def analyze_news_using_base_model(news):
    return generate_prediction(base_model, base_tokenizer, news)


news = "Tesla reports record quarterly earnings driven by strong EV sales"

print("FinPulse Prediction:")
print(analyze_news_using_FinPulse(news))

print("\nBase Model Prediction:")
print(analyze_news_using_base_model(news))