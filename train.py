from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout = 0,
    bias = "none"
)

dataset = load_dataset("json", data_files="financial_sentiment.json")["train"]

def format_prompt(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### News:
{example['input']}

### Response:
{example['output']}"""
    }

dataset = dataset.map(format_prompt)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=2e-4,
        output_dir="outputs"
    ),
)

trainer.train()

model.save_pretrained("financial_lora")
tokenizer.save_pretrained("financial_lora")